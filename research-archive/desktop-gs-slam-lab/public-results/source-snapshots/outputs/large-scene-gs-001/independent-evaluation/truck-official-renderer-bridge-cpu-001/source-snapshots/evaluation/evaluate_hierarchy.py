"""Author .hier hard cuts with the frozen gsplat baseline renderer and metrics.

This evaluates a compressed, unoptimized hierarchy derived from final_state.pt.
It does NOT implement official continuous interpolation, hierarchy optimization,
chunk training, streaming, or a paper-scale reproduction. GPU mode needs the
training owner to release the device; --cpu-only performs no CUDA operation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

import evaluate_baseline as baseline
from read_hierarchy import read_hierarchy, official_size_proxy_cpu, tau_to_threshold

ROOT = baseline.ROOT
TRACK = baseline.TRACK
OUTPUT = baseline.OUTPUT
REAL = TRACK / 'hierarchy-build/real/truck-baseline-final-hierarchy-001'
EXTENSIONS = TRACK / 'hierarchy-build/python-extensions/cu128'
FIXED = {'torch':'2.7.1+cu128','torchvision':'0.22.1+cu128','numpy':'1.26.4',
         'gsplat':'1.5.3','torchmetrics':'1.7.1'}
SETTINGS = [('leaves',None),('tau3',3.),('tau6',6.),('tau15',15.)]
save, require, digest, read_json, summarize = baseline.save, baseline.require, baseline.digest, baseline.read_json, baseline.summarize

def array_digest(a, dtype=None):
    """Canonical C-order bytes, no shape metadata; shape/dtype stored separately."""
    h = hashlib.sha256()
    for i in range(0,len(a),65536):
        h.update(np.ascontiguousarray(a[i:i+65536],dtype=dtype).tobytes())
    return h.hexdigest()

class Topology:
    """Reusable vectorized root-to-leaf coverage; no tree mutation."""
    def __init__(self,h):
        self.h=h
        n=h.nodes.astype(np.int64)
        self.depth,self.parent,self.start,self.leaves,self.merged=n[:,0],n[:,1],n[:,2],n[:,3],n[:,4]
        total=self.leaves+self.merged
        positive=np.flatnonzero(total>0)
        order=positive[np.argsort(self.start[positive],kind='stable')]
        require(np.array_equal(self.start[order],np.cumsum(total[order])-total[order]),'Storage is not one exclusive dense range')
        self.owner=np.repeat(order,total[order]).astype(np.int32)
        require(len(self.owner)==len(h.positions),'Owner range does not cover Gaussian storage')
        slots=np.arange(len(h.positions),dtype=np.int64)
        self.leaf_storage=np.flatnonzero(slots<self.start[self.owner]+self.leaves[self.owner]).astype(np.int32)
        self.layers=[np.flatnonzero(self.depth==d) for d in range(int(self.depth.max()),-1,-1)]
        require(self.parent[0]==-1 and np.all(self.depth[self.parent[1:]]>self.depth[1:]),'Topology is not depth ordered')

    def audit(self,indices,node_ids):
        indices=np.asarray(indices,dtype=np.int32);node_ids=np.asarray(node_ids,dtype=np.int32)
        require(indices.shape==node_ids.shape,'Node IDs/selection shape mismatch')
        require(np.all(indices>=0) and np.all(indices<len(self.owner)),'Selected Gaussian outside storage')
        require(np.array_equal(node_ids,self.owner[indices]),'Selected Gaussian does not belong to claimed node')
        chosen=np.zeros(len(self.owner),dtype=np.bool_);chosen[indices]=True
        require(np.count_nonzero(chosen)==len(indices),'Duplicate selected Gaussian')
        is_leaf=indices<self.start[node_ids]+self.leaves[node_ids]
        selected_summaries=np.bincount(node_ids[~is_leaf],minlength=len(self.start))
        require(np.all((selected_summaries==0)|(selected_summaries==self.merged)),'Partial summary group selected')
        summary=selected_summaries>0
        ancestors=np.zeros(len(self.start),dtype=np.int16)
        for layer in self.layers:
            layer=layer[self.parent[layer]>=0]
            p=self.parent[layer];ancestors[layer]=ancestors[p]+summary[p]
        local_nodes=self.owner[self.leaf_storage]
        counts=ancestors[local_nodes]+summary[local_nodes]+chosen[self.leaf_storage]
        values,frequencies=np.unique(counts,return_counts=True)
        return dict(selected_gaussians=len(indices),selected_leaves=int(is_leaf.sum()),
                    selected_merged_gaussians=int((~is_leaf).sum()),stored_leaves=len(self.leaf_storage),
                    root_to_leaf_coverage_histogram={str(int(k)):int(v) for k,v in zip(values,frequencies)},
                    every_leaf_exactly_once=bool(len(counts)) and bool(np.all(counts==1)),
                    unrepresented_leaves=int(np.count_nonzero(counts==0)),multiply_represented_leaves=int(np.count_nonzero(counts>1)),
                    selected_summary_node_count=int(summary.sum()),scope='Leaf topology representation only, not image equivalence')

    def reference_counts(self,camera,theta):
        sizes=official_size_proxy_cpu(self.h,camera)
        active=sizes>=np.float32(theta)
        parent_active=np.zeros(len(active),dtype=np.bool_)
        parent_active[1:]=active[self.parent[1:]]
        crossing=(~active)&parent_active
        counts=np.where(active,self.leaves,0)+np.where(crossing,self.leaves+np.where(self.depth!=0,self.merged,0),0)
        return counts,sizes

def author_cpu_load(h,output):
    import torch
    sys.path.insert(0,str(EXTENSIONS))
    import gaussian_hierarchy._C as gh
    require(Path(gh.__file__).resolve().is_relative_to(EXTENSIONS.resolve()),'Unexpected hierarchy extension location')
    started=time.perf_counter()
    tensors=gh.load_hierarchy(str(h.path))
    names=('positions','sh','opacities','log_scales','rotations','nodes','boxes')
    mapped=(h.positions,h.sh,h.opacities[:,None],h.log_scales,h.rotations,h.nodes,h.boxes)
    arrays=[]
    for name,t,ref in zip(names,tensors,mapped):
        require(t.device.type=='cpu','Author load unexpectedly returned GPU tensor')
        require(tuple(t.shape)==ref.shape,'Author loader shape mismatch')
        dtype=np.int32 if name=='nodes' else np.float32
        actual=t.numpy();expected_hash=array_digest(ref,dtype)
        require(array_digest(actual)==expected_hash,f'Author loader differs from independent read-only decoder: {name}')
        arrays.append(dict(name=name,shape=list(t.shape),dtype=str(t.dtype),sha256_canonical_bytes=expected_hash))
    leaves=gh.expand_to_target(tensors[5],0)
    require(leaves.device.type=='cpu','Author target traversal unexpectedly returned GPU tensor')
    save(output/'author-loader-audit.json',dict(status='passed_author_CPU_loader_vs_independent_memmap',
        module_path=gh.__file__,module_sha256=digest(gh.__file__),load_and_comparison_seconds=time.perf_counter()-started,
        arrays=arrays,cuda_called=False,leaf_traversal='Author CPU expand_to_target(nodes,0)',
        memory_maps='Read-only r mode; normalized compressed node array is in-memory only'))
    return gh,tensors,leaves.numpy().copy()

def compressed_position_control(h,topology,splats,output):
    require(len(topology.leaf_storage)==len(splats['means']),'Hierarchy leaf count differs from final baseline')
    # Sorting a position multiset preserves duplicate-position Gaussians; it does
    # not pretend hierarchy storage order supplies original Gaussian identities.
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4')])
    source=np.ascontiguousarray(splats['means'].numpy()).view(dtype).reshape(-1)
    leaf=np.ascontiguousarray(h.positions[topology.leaf_storage],dtype=np.float32).view(dtype).reshape(-1)
    source_sorted=np.sort(source);leaf_sorted=np.sort(leaf)
    require(np.array_equal(source_sorted,leaf_sorted),'Hierarchy leaf position multiset differs from exact final baseline')
    result=dict(status='passed_leaf_position_multiset',leaves=len(leaf),baseline_gaussians=len(source),
        sorted_xyz_canonical_sha256=array_digest(source_sorted),
        stage='Unoptimized Creator hierarchy from post-update 30000-step final_state.pt',
        baseline_stage='Exact float32 saved baseline state, independently evaluated earlier',
        leaf_stage='Same position multiset, reordered by Creator; rotations normalized by author PLY loader; compressed float16 rotations/log scales/activated opacity/SH',
        identity_limit='Position multiset and bound source receipts establish origin; no original per-Gaussian IDs exported by Creator',
        implication='Leaf control separates representation/compression changes from tau cut changes; it is not assumed pixel-identical to exact baseline')
    save(output/'leaf-source-control.json',result)
    return result

def hierarchy_tensor_audit(h,tensors,output):
    import torch
    positions,sh,opacity,logscale,rotation,nodes,boxes=tensors
    count=len(positions);fields={}
    for name,t in zip(('positions','sh','activated_opacities','log_scales','rotations_wxyz'),tensors[:5]):
        require(t.dtype==torch.float32 and torch.isfinite(t).all().item(),f'Invalid hierarchy tensor {name}')
        fields[name]=dict(shape=list(t.shape),dtype=str(t.dtype),min=float(t.min()),max=float(t.max()),bytes=t.numel()*t.element_size())
    qnorm=torch.linalg.vector_norm(rotation,dim=1)
    require((qnorm>0).all().item(),'Zero hierarchy quaternion')
    require((opacity>=0).all().item(),'Negative activated hierarchy opacity')
    scales=torch.exp(logscale)
    require(torch.isfinite(scales).all().item() and (scales>0).all().item(),'Invalid exp(log scale)')
    result=dict(status='passed_hierarchy_tensor_storage_audit',gaussian_count=count,tensors=fields,
        wxyz_quaternions='Author storage is wxyz; gsplat internally normalizes; no reordering',
        scale_decode='torch.exp(log_scales)',opacity_decode='Already activated linear alpha/aggregate weight; passed unchanged, no sigmoid and no input clamp',
        activated_opacity_gt_one_count=int((opacity>1).sum()),activated_opacity_max=float(opacity.max()),
        renderer_alpha_policy='Existing fixed gsplat per-pixel alpha=min(0.999,opacity*exp(-sigma)); this is disclosed renderer behavior, not an input-parameter clamp',
        storage='Compressed .hier: float32 xyz; float16 rotation/log-scale/activated-opacity/SH/boxes; author loader yields float32 CPU',
        file_sha256=digest(h.path),file_bytes=h.path.stat().st_size)
    save(output/'hierarchy-state-audit.json',result)
    return result

def cpu_checks(args,output):
    import torch
    training=TRACK/'training'/args.training_run
    configuration=read_json(training/'configuration.json');outcome=read_json(training/'outcome.json')
    source,split,original,image_rows,input_report=baseline.input_audit()
    save(output/'input-audit.json',input_report)
    parsed,geometry=baseline.geometry_audit(training,configuration['config'],original,split)
    save(output/'geometry-audit.json',geometry)
    splats,state=baseline.state_audit(training,configuration,outcome,30000,False)
    save(output/'exact-baseline-state-audit.json',state)
    prepare=read_json(REAL/'preparation.json');decoded=read_json(REAL/'hierarchy-decoding.json')
    require(state['final_state_sha256']==prepare['final_state_sha256'],'Hierarchy source final-state hash differs')
    require(digest(training/'parser_geometry.npz')==prepare['parser_geometry_sha256'],'Hierarchy/source parser frame mismatch')
    require(digest(REAL/'post-update-creator-degree3.ply')==prepare['layout']['output_sha256'],'Creator PLY source receipt mismatch')
    hierarchy=Path(args.hierarchy).resolve()
    require(hierarchy.is_file() and digest(hierarchy)==decoded['sha256'],'Actual hierarchy SHA differs from audited Creator output')
    h=read_hierarchy(hierarchy);topology=Topology(h)
    require(h.summary()['source_commit']=='677c8553dc64dfd62c272eca94a291a277733113','Unexpected hierarchy contract')
    gh,tensors,leaves=author_cpu_load(h,output)
    leaf_node_ids=topology.owner[leaves]
    control=topology.audit(leaves,leaf_node_ids)
    require(control['every_leaf_exactly_once'],'Author leaf traversal lacks exclusive complete coverage')
    require(np.array_equal(np.sort(leaves),topology.leaf_storage),'Author leaf traversal differs from decoder storage')
    leaf_source=compressed_position_control(h,topology,splats,output)
    state_audit=hierarchy_tensor_audit(h,tensors,output)
    del splats
    baseline_dir=OUTPUT/args.baseline_evaluation_run
    previous=read_json(baseline_dir/'metrics.json');previous_provenance=read_json(baseline_dir/'provenance.json')
    require(previous['status']=='completed_independent_post_update_heldout_evaluation','Prior exact baseline evaluation incomplete')
    require(previous_provenance['final_state_sha256']==state['final_state_sha256'],'Comparison baseline state differs')
    require(previous_provenance['input_manifest_sha256']==input_report['input_manifest_sha256'] and previous_provenance['split_sha256']==input_report['split_sha256'],'Comparison baseline split/input differs')
    require([r['name'] for r in previous['per_image']]==split['test_names'],'Comparison camera order differs')
    for row in previous['per_image']:
        require(row['target_sha256']==digest(row['target_path']),'Comparison target image changed')
    camera=int(parsed['test_indices'][0]);probe=[]
    for setting,tau in SETTINGS:
        if tau is None:probe.append(dict(setting=setting,**control));continue
        theta=tau_to_threshold(tau,float(parsed['K'][camera,0,0]));counts,sizes=topology.reference_counts(parsed['camtoworlds'][camera,:3,3],theta)
        owners=np.repeat(np.arange(len(counts),dtype=np.int32),counts)
        # Dense per-node starts allow offsets without a Python loop.
        repeated=np.repeat(np.cumsum(counts)-counts,counts)
        selected=(topology.start[owners]+np.arange(int(counts.sum()))-repeated).astype(np.int32)
        checked=topology.audit(selected,owners)
        probe.append(dict(setting=setting,tau=tau,threshold=theta,root_size_proxy=float(sizes[0]) if np.isfinite(sizes[0]) else 'positive-infinity',
                          root_fallback_failure=not checked['every_leaf_exactly_once'],**checked))
    save(output/'cpu-first-camera-cuts.json',dict(camera_name=split['test_names'][0],probe=probe,cuda_used=False,
         scope='Fast independent source-predicate CPU audit; GPU author cuts are compared per frame only during authorized GPU mode'))
    provenance=dict(hierarchy_path=str(hierarchy),hierarchy_sha256=digest(hierarchy),hierarchy_summary=h.summary(),
        creator_preparation_sha256=digest(REAL/'preparation.json'),creator_decoding_sha256=digest(REAL/'hierarchy-decoding.json'),
        final_state_sha256=state['final_state_sha256'],parser_geometry_sha256=digest(training/'parser_geometry.npz'),
        input_manifest_sha256=input_report['input_manifest_sha256'],split_sha256=input_report['split_sha256'],
        baseline_metrics_path=str(baseline_dir/'metrics.json'),baseline_metrics_sha256=digest(baseline_dir/'metrics.json'),
        evaluation_script_sha256=digest(__file__),independent_decoder_sha256=digest(ROOT/'scripts/large_scene_gs/read_hierarchy.py'),
        metrics_reference_script_sha256=digest(ROOT/'scripts/large_scene_gs/evaluate_baseline.py'),
        protocol='Leaves + tau 3/6/15 hard cuts, same gsplat renderer; no author continuous interpolation or post-optimization',
        holdout_scope='GS photometric optimization only; full-scene public SfM initialization; Creator filtered with 219 training pose archive',
        frame='Same saved normalized global SfM Sim3; no evaluation renormalization; units not metres',
        settings=[dict(name=n,tau=t) for n,t in SETTINGS],leaf_control=leaf_source)
    save(output/'provenance.json',provenance)
    return configuration['config'],parsed,split,image_rows,h,topology,gh,tensors,leaves,previous,state_audit

def save_selection(path,indices,parents,node_ids):
    np.savez(path,indices=np.asarray(indices,dtype=np.int32),parent_gaussian_indices=np.asarray(parents,dtype=np.int32),
             node_ids=np.asarray(node_ids,dtype=np.int32))
    return dict(path=str(path),file_sha256=digest(path),selected_index_bytes_sha256=array_digest(indices,np.int32),
                node_id_bytes_sha256=array_digest(node_ids,np.int32),parent_index_bytes_sha256=array_digest(parents,np.int32),
                dtype='int32',count=len(indices),order='Unchanged author traversal/selection order')

def gpu_renders(cfg,parsed,split,image_rows,h,topology,gh,tensors,leaves,output):
    import torch
    from gsplat.rendering import rasterization
    require(torch.cuda.is_available(),'CUDA unavailable')
    versions={k:importlib.metadata.version(k) for k in FIXED}
    require(versions==FIXED,'Package versions differ from frozen baseline protocol')
    # Metrics are instantiated after ALL settings render, so their allocations
    # cannot enter any setting's render timing/allocator readings.
    pre_allocated=torch.cuda.memory_allocated();pre_reserved=torch.cuda.memory_reserved()
    gpu=[t.to('cuda') for t in tensors]
    xyz,sh,alpha,logs,quats,nodes,boxes=gpu
    cams=torch.from_numpy(parsed['camtoworlds'].astype(np.float32)).to('cuda')
    Ks=torch.from_numpy(parsed['K'].astype(np.float32)).to('cuda')
    indices_buf=torch.empty(len(xyz),dtype=torch.int32,device='cuda');parents_buf=torch.empty_like(indices_buf);node_buf=torch.empty_like(indices_buf)
    leaf_gpu=torch.from_numpy(leaves).to('cuda')
    viewdir_cpu=torch.zeros(3,dtype=torch.float32)
    torch.cuda.synchronize()
    resident=dict(full_gaussian_count=len(xyz),full_node_count=len(nodes),allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),pre_load_allocated_bytes=pre_allocated,pre_load_reserved_bytes=pre_reserved,
        model_and_tree_tensors_bytes=sum(t.numel()*t.element_size() for t in gpu),
        scope='All hierarchy Gaussians, nodes, boxes, cameras and reusable output buffers resident on GPU. No streaming or model eviction; smaller subsets do not establish resident-memory savings.')
    save(output/'full-resident-memory.json',resident)
    target_paths=[Path(cfg['data_dir'])/f"images_{cfg['data_factor']}"/image_rows[n]['resized_name'] for n in split['test_names']]
    targets=[]
    for p in target_paths:
        with Image.open(p) as im:im.load();targets.append(np.asarray(im).copy())
    tests=parsed['test_indices'].astype(int);results=[]

    @torch.no_grad()
    def select(index,tau):
        if tau is None:return leaf_gpu,None
        theta=tau_to_threshold(tau,float(parsed['K'][index,0,0]))
        count=gh.expand_to_size(nodes,boxes,theta,cams[index,:3,3].contiguous(),viewdir_cpu,indices_buf,parents_buf,node_buf)
        require(0<=count<=len(xyz),'Author cut returned invalid count')
        return indices_buf[:count],theta

    @torch.no_grad()
    def subset(index,ids):
        ix=ids.long()
        return dict(means=xyz[ix],quats=quats[ix],scales=torch.exp(logs[ix]),opacities=alpha[ix,0],colors=sh[ix],
                    viewmats=torch.linalg.inv(cams[index:index+1]),Ks=Ks[index:index+1])

    @torch.no_grad()
    def raster(index,height,width,params):
        return rasterization(**params,width=width,height=height,sh_degree=cfg['sh_degree'],near_plane=cfg['near_plane'],
            far_plane=cfg['far_plane'],packed=cfg['packed'],absgrad=cfg['strategy']['absgrad'],sparse_grad=cfg['sparse_grad'],
            rasterize_mode='antialiased' if cfg['antialiased'] else 'classic',camera_model=cfg['camera_model'],
            distributed=False,with_ut=cfg['with_ut'],with_eval3d=cfg['with_eval3d'])

    for setting,tau in SETTINGS:
        folder=output/setting;folder.mkdir();selection_folder=folder/'selections';selection_folder.mkdir()
        predictions=[];rows=[];shared_leaf_receipt=None
        if tau is None:
            owners=topology.owner[leaves];p=topology.parent[owners]
            parent_start=np.where(p>=0,topology.start[np.maximum(p,0)],-1).astype(np.int32)
            shared_leaf_receipt=save_selection(selection_folder/'fixed-author-leaves.npz',leaves,parent_start,owners)
        # Three warmups per setting, no metrics, CPU coverage or disk I/O.
        for j in range(3):
            index=int(tests[j]);hh,ww=targets[j].shape[:2];ids,_=select(index,tau)
            if len(ids):
                params=subset(index,ids);warm=raster(index,hh,ww,params);torch.cuda.synchronize();del warm,params
        torch.cuda.synchronize()
        for j,index in enumerate(tests):
            index=int(index);height,width=targets[j].shape[:2]
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            ids,theta=select(index,tau);torch.cuda.synchronize();cut_seconds=time.perf_counter()-began
            cut_peak=torch.cuda.max_memory_allocated();cut_reserved=torch.cuda.max_memory_reserved()
            selected=ids.cpu().numpy().copy()
            if tau is None:
                parents=parent_start;nodeids=owners;receipt=shared_leaf_receipt
            else:
                parents=parents_buf[:len(ids)].cpu().numpy().copy();nodeids=node_buf[:len(ids)].cpu().numpy().copy()
                receipt=save_selection(selection_folder/f'{j:02d}-{Path(split["test_names"][j]).stem}.npz',selected,parents,nodeids)
            checked=topology.audit(selected,nodeids)
            reference_match=True;root_size=None;root_fallback=False
            if tau is not None:
                expected_counts,sizes=topology.reference_counts(parsed['camtoworlds'][index,:3,3],theta)
                actual_counts=np.bincount(nodeids,minlength=len(expected_counts))
                reference_match=bool(np.array_equal(actual_counts,expected_counts))
                p=topology.parent[nodeids];expected_parent=np.where(p>=0,topology.start[np.maximum(p,0)],-1)
                reference_match=reference_match and bool(np.array_equal(parents,expected_parent))
                root_size=float(sizes[0]) if np.isfinite(sizes[0]) else 'positive-infinity'
                root_fallback=bool(sizes[0]<np.float32(theta)) and not checked['every_leaf_exactly_once']
            row=dict(name=split['test_names'][j],test_index=index,setting=setting,tau=tau,threshold=theta,
                selection=receipt,coverage=checked,author_GPU_vs_CPU_predicate_match=reference_match,
                root_size_proxy=root_size,root_fallback_failure=root_fallback,
                cut_seconds=cut_seconds if tau is not None else 0.,fixed_leaf_index_reuse_seconds=cut_seconds if tau is None else None,
                cut_peak_allocated_bytes=cut_peak,cut_peak_reserved_bytes=cut_reserved,
                opacity_gt_one_selected_count=int(np.count_nonzero(h.opacities[selected]>1)),
                opacity_input_transform='none; activated alpha passed unchanged')
            # Failures are preserved per frame without changing the author cut,
            # inventing a root fallback, or pooling a partial set of views.
            if not checked['every_leaf_exactly_once'] or not reference_match or not len(selected):
                row.update(status='invalid_cut_no_render',failure='Coverage, exact-source cut or nonempty-cut check failed')
                rows.append(row);save(folder/'per-frame-audit.json',rows);continue
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            params=subset(index,ids);torch.cuda.synchronize();subset_seconds=time.perf_counter()-began
            subset_peak=torch.cuda.max_memory_allocated();subset_reserved=torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
            colors,alphas,info=raster(index,height,width,params);torch.cuda.synchronize();raster_seconds=time.perf_counter()-began
            raster_peak=torch.cuda.max_memory_allocated();raster_reserved=torch.cuda.max_memory_reserved()
            require(tuple(colors.shape)==(1,height,width,3) and torch.isfinite(colors).all().item(),'Invalid GS rendered RGB')
            require(torch.isfinite(alphas).all().item(),'Nonfinite GS rendered alpha')
            raw=colors[0].cpu().numpy().copy();predictions.append(np.clip(raw,0,1))
            row.update(status='valid_cut_and_finite_render',subset_decode_seconds=subset_seconds,gsplat_raster_seconds=raster_seconds,
                sum_measured_GPU_stage_seconds=(cut_seconds if tau is not None else 0.)+subset_seconds+raster_seconds,
                subset_peak_allocated_bytes=subset_peak,subset_peak_reserved_bytes=subset_reserved,
                raster_peak_allocated_bytes=raster_peak,raster_peak_reserved_bytes=raster_reserved,
                unclipped_RGB_min=float(raw.min()),unclipped_RGB_max=float(raw.max()),clipped_RGB_fraction=float(np.mean((raw<0)|(raw>1))),
                render_resolution=[width,height])
            rows.append(row);del colors,alphas,info,params,raw
        valid=[r for r in rows if r['status']=='valid_cut_and_finite_render']
        complete=len(valid)==len(tests)==32
        timing=dict(status='all_32_frames_valid' if complete else 'incomplete_invalid_cut_preserved_no_pooled_metrics',
            warmup_count=3,total_frames=len(rows),valid_frames=len(valid),invalid_frames=len(rows)-len(valid),
            stages={},full_resident_memory=resident,
            timing_boundary='CUDA sync surrounds separate author GPU cut, subset gather/exp/view inverse, and gsplat raster stages; allocation inside each stage included. CPU audits, selection transfers/writes, prediction transfers/clamp and metrics excluded.',
            total_definition='Sum of independently synchronized stage durations; extra synchronizations and excluded CPU audit gaps mean this is not continuous playback throughput.',
            hard_cut='No parent/child parameter interpolation and no official per-pixel alpha/sibling correction',
            streaming=False)
        if complete:
            for key in ('cut_seconds','subset_decode_seconds','gsplat_raster_seconds','sum_measured_GPU_stage_seconds'):
                timing['stages'][key]=summarize([r[key] for r in valid])
            timing['selected_count']=summarize([r['selection']['count'] for r in valid])
            timing['max_allocated_bytes']=max(max(r[k] for k in ('cut_peak_allocated_bytes','subset_peak_allocated_bytes','raster_peak_allocated_bytes')) for r in valid)
            timing['max_reserved_bytes']=max(max(r[k] for k in ('cut_peak_reserved_bytes','subset_peak_reserved_bytes','raster_peak_reserved_bytes')) for r in valid)
            np.savez(folder/'predictions.npz',image_names=np.array(split['test_names']),rgb=np.stack(predictions))
        save(folder/'per-frame-audit.json',rows);save(folder/'timings.json',timing)
        results.append(dict(setting=setting,tau=tau,complete=complete,folder=folder,predictions=predictions,rows=rows,timing=timing))
        print(json.dumps(dict(setting=setting,valid_frames=len(valid),selected=timing.get('selected_count')),ensure_ascii=False),flush=True)
    del gpu,xyz,sh,alpha,logs,quats,nodes,boxes,cams,Ks,indices_buf,parents_buf,node_buf,leaf_gpu
    torch.cuda.synchronize()
    return results,targets,target_paths,versions

def score(results,targets,target_paths,split,previous,output,versions):
    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    import torchmetrics.functional.image.lpips as lpips_source
    paths=[Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',Path(lpips_source.__file__).parent/'lpips_models/alex.pth']
    require(all(p.is_file() for p in paths),'Previously installed LPIPS weights missing; do not download')
    weight_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    require(weight_files==previous['weight_files'],'Perceptual weights differ from exact baseline')
    ssim=StructuralSimilarityIndexMeasure(data_range=1.).to('cuda').eval()
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to('cuda').eval()
    prev_by_name={r['name']:r for r in previous['per_image']};summary=[]
    for result in results:
        folder=result['folder'];pairs=folder/'paired';preds=folder/'predicted';pairs.mkdir();preds.mkdir();rows=[]
        if not result['complete']:
            summary.append(dict(setting=result['setting'],status='invalid_frames_preserved_no_average',valid_frames=result['timing']['valid_frames']));continue
        for j,(name,pred,truth_uint8) in enumerate(zip(split['test_names'],result['predictions'],targets)):
            truth=truth_uint8.astype(np.float64)/255.;mse=float(np.mean((pred.astype(np.float64)-truth)**2))
            require(mse>0 and np.isfinite(mse),'Nonfinite/perfect PSNR representation unsupported')
            psnr=float(-10*np.log10(mse));pt=torch.from_numpy(pred).permute(2,0,1)[None].to('cuda')
            gt=torch.from_numpy(truth_uint8).permute(2,0,1)[None].to(device='cuda',dtype=torch.float32)/255.
            with torch.no_grad():sv=float(ssim(pt,gt).cpu());lv=float(lpips(pt,gt).cpu())
            ssim.reset();lpips.reset();require(np.isfinite(sv+lv),'Nonfinite metric')
            quantized=(pred*255).astype(np.uint8);pfile=preds/f'{j:02d}-{Path(name).stem}.png';pair=pairs/pfile.name
            Image.fromarray(quantized,'RGB').save(pfile);Image.fromarray(np.concatenate([truth_uint8,quantized],axis=1),'RGB').save(pair)
            original=prev_by_name[name]
            rows.append(dict(name=name,width=truth_uint8.shape[1],height=truth_uint8.shape[0],mse_numpy_float64=mse,
                psnr_numpy_db=psnr,ssim_torchmetrics=sv,lpips_alex_torchmetrics=lv,
                minus_exact_baseline=dict(psnr_db=psnr-original['psnr_numpy_db'],ssim=sv-original['ssim_torchmetrics'],lpips=lv-original['lpips_alex_torchmetrics']),
                target_path=str(target_paths[j]),target_sha256=digest(target_paths[j]),prediction_path=str(pfile),prediction_sha256=digest(pfile),paired_path=str(pair),paired_sha256=digest(pair)))
            del pt,gt
        metrics=dict(status='complete_32_valid_hard_cut_frames',setting=result['setting'],tau=result['tau'],test_images=32,
            psnr_numpy_mean_per_image_db=float(np.mean([r['psnr_numpy_db'] for r in rows])),
            ssim_mean_per_image=float(np.mean([r['ssim_torchmetrics'] for r in rows])),lpips_alex_mean_per_image=float(np.mean([r['lpips_alex_torchmetrics'] for r in rows])),
            psnr_aggregation=previous['psnr_aggregation'],photometric_scope=previous['photometric_scope'],
            ssim_protocol=previous['ssim_protocol'],lpips_protocol=previous['lpips_protocol'],package_versions=versions,weight_files=weight_files,
            per_image=rows,render_timing_summary=result['timing'])
        metrics['minus_exact_baseline']={k:metrics[k]-previous[k] for k in ('psnr_numpy_mean_per_image_db','ssim_mean_per_image','lpips_alex_mean_per_image')}
        save(folder/'metrics.json',metrics);save(folder/'per-image-metrics.json',rows)
        summary.append({k:v for k,v in metrics.items() if k not in {'per_image','weight_files'}})
    report=dict(status='complete_four_settings_32_valid_views_each' if all(x['complete'] for x in results) else 'incomplete_one_or_more_settings_failed',
        exact_baseline={k:previous[k] for k in ('psnr_numpy_mean_per_image_db','ssim_mean_per_image','lpips_alex_mean_per_image')},settings=summary,
        comparison_scope='Exact final float32 baseline vs compressed unoptimized author hierarchy hard cuts, fixed cameras/targets/renderer/metrics. Leaf representation changes and tau approximation are separate stages.',
        method_limit='No continuous parent/child parameter or alpha interpolation, no hierarchy post-optimization, no four-chunk merge, no streaming; tau is pixel-linked dimensionless threshold based on saved fx.',
        opacity_policy='Author activated hierarchy opacity passed unchanged, including values >1; no sigmoid or input clamp. Fixed gsplat internally caps each pixel alpha at .999.',
        memory_policy='Full hierarchy resident on GPU throughout renders. Per-frame peak is PyTorch allocator scope; no claim of reduced streaming resident memory.')
    save(output/'metrics.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-run',required=True)
    p.add_argument('--training-run',default='truck-baseline-001')
    p.add_argument('--baseline-evaluation-run',default='truck-baseline-heldout-001')
    p.add_argument('--hierarchy',default=str(REAL/'official-creator/hierarchy.hier'))
    p.add_argument('--cpu-only',action='store_true')
    args=p.parse_args()
    require(all(x.replace('-','').replace('_','').isalnum() for x in (args.evaluation_run,args.training_run,args.baseline_evaluation_run)),'Invalid run name')
    output=OUTPUT/args.evaluation_run;output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();outcome=dict(status='running',started_utc=baseline.now(),cpu_only=args.cpu_only)
    save(output/'outcome.json',outcome)
    try:
        cfg,parsed,split,image_rows,h,topology,gh,tensors,leaves,previous,state=cpu_checks(args,output)
        if args.cpu_only:
            outcome.update(status='passed_CPU_loader_geometry_leaf_source_and_probe_no_GPU',cuda_used=False)
        else:
            results,targets,target_paths,versions=gpu_renders(cfg,parsed,split,image_rows,h,topology,gh,tensors,leaves,output)
            metrics=score(results,targets,target_paths,split,previous,output,versions)
            outcome.update(status=metrics['status'],cuda_used=True,settings=[dict(name=r['setting'],complete=r['complete']) for r in results])
            lines=['# 同渲染器未优化层级硬切对照','',
                '这是完成 30,000 次更新后的单地图 baseline 派生层级，使用作者 .hier 加载与 GPU cut，再由相同 gsplat 渲染器评测固定 32 个留出相机。没有进行块训练、层级后优化或官方连续参数／每像素 alpha 插值。',
                '',f"精确 baseline：PSNR {previous['psnr_numpy_mean_per_image_db']:.6f} dB，SSIM {previous['ssim_mean_per_image']:.6f}，LPIPS {previous['lpips_alex_mean_per_image']:.6f}。",'']
            for s in metrics['settings']:
                if s['status']!='complete_32_valid_hard_cut_frames':lines.append(f"{s['setting']} 未完整通过，保留逐帧失败，未汇总指标。");continue
                stages=s['render_timing_summary']['stages']
                lines.append(f"{s['setting']}：PSNR {s['psnr_numpy_mean_per_image_db']:.6f} dB，SSIM {s['ssim_mean_per_image']:.6f}，LPIPS {s['lpips_alex_mean_per_image']:.6f}；cut／subset+decode／gsplat p50 分别 {stages['cut_seconds']['p50']*1000:.4f}／{stages['subset_decode_seconds']['p50']*1000:.4f}／{stages['gsplat_raster_seconds']['p50']*1000:.4f} ms。")
            lines+=['','leaves 保留同一位置多重集，但经过作者加载四元数归一化与 .hier float16 属性压缩，不能假定等于精确 baseline；tau3／6／15 再增加硬切近似。',
                '', '已激活的不透明度直接传入，允许聚合值超过 1，未二次 sigmoid 或静默裁切；固定 gsplat 的每像素 alpha 上限 .999 是渲染器原有行为。',
                '', '计时分列作者 GPU cut、subset 参数提取／exp／相机求逆和光栅化。完整层级一直驻留 GPU，未做 streaming；不能把选点减少写成常驻显存减少。各阶段单独同步后的时间和也不等于连续播放吞吐率。',
                '', '逐帧选点／nodeids／哈希、叶路径覆盖、失败、真实配对图和指标均保留；只在每组 32 帧全部有效后求平均。相机标定来自全场景 SfM，留出针对图像优化，长度单位不是米。']
            (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        require(digest(h.path)==state['file_sha256'],'Read-only hierarchy unexpectedly changed')
        outcome.update(ended_utc=baseline.now(),elapsed_seconds=time.perf_counter()-started,evaluation_script_sha256=digest(__file__))
        save(output/'outcome.json',outcome);print(json.dumps(outcome,ensure_ascii=False),flush=True)
    except Exception as e:
        outcome.update(status='failed_preserved_evaluation',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),ended_utc=baseline.now())
        save(output/'outcome.json',outcome);raise

if __name__=='__main__':main()
