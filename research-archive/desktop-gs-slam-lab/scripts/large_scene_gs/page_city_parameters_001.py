"""Bounded GPU parameter cache using frozen camera cuts, with exact RGB verification.

This deliberately isolates host-to-device parameter paging. It is not online tree
selection or disk streaming, and timing includes synchronous CPU cache bookkeeping.
"""
from pathlib import Path
from types import SimpleNamespace
import json, sys, time, traceback
import numpy as np
import evaluate_official_hierarchy as core

OLD=core.TRACK/'independent-evaluation/small-city006-heldout-1404-001'
OUT=core.TRACK/'research/city-parameter-paging-023'
SAVE=core.save

def main():
    OUT.mkdir(exist_ok=False)
    SAVE(OUT/'plan.json',dict(status='registered_before_measurement',views='All 117 frozen views, ascending original order, tau=3; one cold-to-warm pass',
        cache='Capacity rounded up to 65536 rows from largest per-view union of children and interpolation parents. Skybox permanently resident. Evict least recently used rows not required by current view.',
        unchanged='Original model, SH, property decoders, camera, rasterizer, interpolation weights, Gaussian rendering order',
        limits=['Selection and weights replay frozen verified archives; no online tree traversal','Full raw model retained in CPU RAM; no disk paging','No asynchronous prefetch','No peak memory bound for native rasterizer'],script_sha256=core.digest(Path(__file__))))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes());core.source_audit(OUT)
    protocol=core.read_json(OLD/'protocol.json');inp=Path(protocol['independent_input_npz_path'])
    assert core.digest(inp)==protocol['independent_input_npz_sha256']
    inputs=np.load(inp);names=inputs['names'];H,W=inputs['rgb'].shape[1:3]
    records=core.read_json(OLD/'merged-post/tau3/per-frame-audit.json')
    requirements=[];bindings=[]
    def load_case(j):
        row=records[j];assert row['name']==names[j]
        sel=Path(row['selected']['path']);assert core.digest(sel)==row['selected']['file_sha256']
        with np.load(sel) as a:ids=a['indices'].copy();parents=a['parent_gaussian_indices'].copy()
        assert ids.min()>=0 and parents.min()>=0
        stem=f'{j:02d}-{Path(str(names[j])).stem}'
        weights_path=OLD/'merged-post/tau3/selections'/f'{stem}-interpolation.npz'
        with np.load(weights_path) as a:w=a['child_weights'].copy();k=a['num_parent_children'].copy()
        assert core.array_digest(w)==row['interpolation_weight_bytes_sha256']
        assert core.array_digest(k)==row['sibling_count_bytes_sha256']
        return ids,parents,w,k,stem
    for j in range(len(names)):
        ids,parents,w,k,stem=load_case(j);required=np.unique(np.concatenate((ids,parents)))
        requirements.append(len(required));bindings.append(dict(index=j,name=str(names[j]),selected=len(ids),required=len(required),
            selected_sha256=core.array_digest(ids),parents_sha256=core.array_digest(parents),weights_sha256=core.array_digest(w)))
    capacity=int(np.ceil(max(requirements)/65536)*65536)
    SAVE(OUT/'input-bindings.json',dict(camera_archive_sha256=core.digest(inp),capacity_tree_rows=capacity,cases=bindings))
    desc=next(d for d in protocol['models'] if d['name']=='merged-post')
    sky_audit=core.read_json(OLD/'shared-skybox-audit.json')
    sky,_=core.skybox_PLY_CPU(Path(sky_audit['ply_path']),sky_audit['count'],OUT,sky_audit['source_metadata'])
    assert core.digest(Path(sky_audit['ply_path']))==sky_audit['ply_sha256']
    root,h,topology,gh,tensors,leaves,audit=core.model_cpu(desc,{},sky,OUT)
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(core.REPO));sys.path.insert(0,str(core.EXT))
    from scene.gaussian_model import GaussianModel
    import gaussian_renderer as renderer
    class CachedModel(GaussianModel):
        @property
        def get_features(self):return self.features_cached
        @property
        def get_scaling(self):return self.scales_cached
        @property
        def get_rotation(self):return self.rotations_cached
        @property
        def get_opacity(self):return self.opacity_cached
    sky_count=audit['background_G'];N=capacity+sky_count;tree_count=audit['tree_G']
    row_raw_bytes=sum(t[0].numel()*t.element_size() for t in tensors[:5]);row_cached_bytes=32
    free,total=torch.cuda.mem_get_info()
    bound=core.memory_plan(N,0,len(leaves),max(b['selected'] for b in bindings),sky_count,1)
    assert bound['conservative_Python_tensor_peak_upper_bytes']+1024**3<free
    SAVE(OUT/'memory-preflight.json',dict(free_bytes=free,total_bytes=total,bound=bound,capacity=capacity,raw_bytes_per_row=row_raw_bytes))
    pc=CachedModel(3);pc.active_sh_degree=3;pc.skybox_points=sky_count;pc.pretrained_exposures=None;pc.opacity_activation=torch.abs
    gpu=[torch.zeros((N,*t.shape[1:]),dtype=t.dtype,device='cuda') for t in tensors[:5]]
    pc._xyz,pc.features_cached,pc._opacity,pc._scaling,pc._rotation=gpu
    pc._features_dc=pc.features_cached[:,:1];pc._features_rest=pc.features_cached[:,1:]
    pc.scales_cached=torch.zeros((N,3),device='cuda');pc.rotations_cached=torch.zeros((N,4),device='cuda');pc.opacity_cached=torch.zeros((N,1),device='cuda')
    def upload(global_ids,slots):
        si=torch.from_numpy(slots.astype(np.int64)).cuda();gi=torch.from_numpy(global_ids.astype(np.int64))
        # Only missing raw rows cross host/device; decoded values are computed on the GPU.
        for target,source in zip(gpu,tensors[:5]):target[si]=source[gi].to('cuda')
        pc.scales_cached[si]=torch.exp(pc._scaling[si])
        pc.rotations_cached[si]=torch.nn.functional.normalize(pc._rotation[si],dim=1)
        pc.opacity_cached[si]=torch.abs(pc._opacity[si])
    upload(np.arange(tree_count,tree_count+sky_count),np.arange(capacity,N))
    global_to_slot=np.full(tree_count,-1,np.int32);slot_to_global=np.full(capacity,-1,np.int32);last_use=np.full(capacity,-1,np.int64)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    rows=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    @torch.no_grad()
    def one(j):
        ids,parents,w,k,stem=load_case(j)
        required=np.unique(np.concatenate((ids,parents)));assert required.max()<tree_count
        torch.cuda.synchronize();start=time.perf_counter()
        occupied=global_to_slot[required];missing=required[occupied<0]
        protected=np.zeros(capacity,bool);protected[occupied[occupied>=0]]=True
        candidates=np.flatnonzero(~protected)
        available=candidates[np.argsort(last_use[candidates],kind='stable')[:len(missing)]]
        old=slot_to_global[available];evicted=old[old>=0]
        assert len(available)==len(missing) and not np.isin(evicted,required).any()
        global_to_slot[evicted]=-1;global_to_slot[missing]=available;slot_to_global[available]=missing
        upload(missing,available)
        last_use[global_to_slot[required]]=j
        local_ids=global_to_slot[ids];local_parents=global_to_slot[parents]
        assert np.array_equal(slot_to_global[local_ids],ids) and np.array_equal(slot_to_global[local_parents],parents)
        torch.cuda.synchronize();paging_ms=1000*(time.perf_counter()-start)
        camera=core.make_official_camera_CPU(inputs['K'][j],inputs['c2w'][j],W,H,str(names[j]))
        fields=vars(camera).copy()
        for key in ('world_view_transform','projection_matrix','full_proj_transform','camera_center'):fields[key]=fields[key].cuda()
        ii=torch.from_numpy(local_ids).cuda();pp=torch.from_numpy(local_parents).cuda()
        ww=torch.zeros(len(w)+sky_count,device='cuda');ww[:len(w)]=torch.from_numpy(w).cuda()
        kk=torch.ones(len(k)+sky_count,device='cuda',dtype=torch.int32);kk[:len(k)]=torch.from_numpy(k).cuda()
        torch.cuda.synchronize();start=time.perf_counter()
        result=renderer.render_post(SimpleNamespace(**fields),pc,pipe,bg,render_indices=ii,parent_indices=pp,interpolation_weights=ww,num_node_kids=kk,interp_python=True,use_trained_exp=False)
        torch.cuda.synchronize();render_ms=1000*(time.perf_counter()-start)
        rgb=result['render'].permute(1,2,0).cpu().numpy();reference=np.load(OLD/'merged-post/tau3/float-frames'/f'{stem}.npy')
        exact=bool(np.array_equal(rgb,reference));diff=float(np.max(np.abs(rgb-reference)))
        row=dict(index=j,name=str(names[j]),required_tree_rows=len(required),selected_tree_rows=len(ids),cache_hits=len(required)-len(missing),cache_misses=len(missing),evicted_rows=len(evicted),occupied_tree_rows=int((slot_to_global>=0).sum()),raw_parameter_transfer_bytes=len(missing)*row_raw_bytes,
            paging_host_ms=paging_ms,render_host_ms=render_ms,torch_allocated_bytes=torch.cuda.memory_allocated(),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),rgb_bitwise_equal=exact,rgb_max_abs_difference=diff)
        rows.append(row);SAVE(OUT/'measurements.json',rows)
        assert exact and np.isfinite(rgb).all(),f'Pixel mismatch at {j}: {diff}'
        if j%10==0:print(json.dumps(row),flush=True)
    for j in range(len(names)):one(j)
    assert core.digest(Path(desc['path']))==audit['source_sha256']
    result=dict(status='completed_bounded_parameter_cache_exact_RGB',views=len(rows),all_RGB_bitwise_equal=True,
        tree_capacity=capacity,skybox_rows=sky_count,total_model_rows=audit['total_G'],
        resident_parameter_bytes=N*(row_raw_bytes+row_cached_bytes),whole_model_parameter_bytes=audit['total_G']*(row_raw_bytes+row_cached_bytes),
        total_raw_parameter_transfer_bytes=sum(r['raw_parameter_transfer_bytes'] for r in rows)+sky_count*row_raw_bytes,
        total_evicted_rows=sum(r['evicted_rows'] for r in rows),total_cache_hits=sum(r['cache_hits'] for r in rows),total_cache_misses=sum(r['cache_misses'] for r in rows),
        paging_median_ms=float(np.median([r['paging_host_ms'] for r in rows])),render_median_ms=float(np.median([r['render_host_ms'] for r in rows])),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_model_parameters_uploaded_at_once=False,
        limitations=['Frozen precomputed cuts and interpolation weights; no online selector','CPU model fully resident; no disk streaming','Synchronous LRU transfer; no prefetch','Parameter bytes exclude temporary raster and cache bookkeeping','Single pass timing is descriptive, not a speed comparison'])
    SAVE(OUT/'metrics.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
