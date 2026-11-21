"""Independent CPU ledger for real006 Creator anchors, exposure and formal post.

Read-only maps; no Torch/CUDA, renderer, native dispatch or old005 edits.
Creator-only evidence and four I15000 post completion have distinct statuses.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'independent-evaluation'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
BINDING=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
BINDING_SHA='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'
CONFIG_SHA='562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47'
NATIVE_SHA='a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6'
OLD=BASE/'official-hierarchy/small-city-hierarchy-005'
CODEC=OLD/'helper-snapshots/read_hierarchy.py'
TRAIN=OLD/'helper-snapshots/run_small_city_stage.py'
BINDER=ROOT/'scripts/large_scene_gs/bind_small_city_lineage_006.py'
HELPER=OLD/'helper-snapshots/run_official_hierarchy.py'

def require(c,m):
    if not c:raise RuntimeError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def pin(p,d):require(sha(p)==d,'Actual bound source changed: '+str(p))
def set_record(ids):
    a=np.asarray(ids,dtype='<i8');return dict(count=len(a),sorted_int64_le_sha256=hashlib.sha256(a.tobytes()).hexdigest(),first_ids=a[:16].tolist())

def mapped(path,codec,tail=0):
    # Exact compressed writer schema, independently vector-validated. Maps are
    # mode=r and nodes alone are copied/normalized into CPU int32 arrays.
    with path.open('rb') as f:signed=struct.unpack('<i',f.read(4))[0]
    require(signed<0,'Expect actual compressed author writer')
    g=-signed;off=4+124*g
    with path.open('rb') as f:f.seek(off);n=struct.unpack('<i',f.read(4))[0]
    require(g>tail>=0 and n>0 and path.stat().st_size==off+4+36*n,'Compressed payload length/tail contract')
    fields=[('positions','<f4',4,(g,3)),('rotations','<f2',4+12*g,(g,4)),('log_scales','<f2',4+20*g,(g,3)),('opacities','<f2',4+26*g,(g,)),('sh','<f2',4+28*g,(g,16,3))]
    arrays={name:np.memmap(path,dtype=d,mode='r',offset=o,shape=shape) for name,d,o,shape in fields}
    dtype=np.dtype([('parent','<i4'),('start','<i4'),('children_start','<i4'),('packed','<i2',(4,))])
    p=np.memmap(path,dtype=dtype,mode='r',offset=off+4,shape=(n,))
    nodes=np.empty((n,7),np.int32)
    nodes[:,0]=p['packed'][:,0];nodes[:,1]=p['parent'];nodes[:,2]=p['start'];nodes[:,3]=p['packed'][:,2];nodes[:,4]=p['packed'][:,3];nodes[:,5]=p['children_start'];nodes[:,6]=p['packed'][:,1]
    boxes=np.memmap(path,dtype='<f2',mode='r',offset=off+4+20*n,shape=(n,2,4))
    require(nodes[0,1]==-1 and np.all(nodes[:,[0,2,3,4,6]]>=0) and np.all((nodes[1:,1]>=0)&(nodes[1:,1]<n)),'Node metadata/parents')
    require(np.all(nodes[1:,0]<nodes[nodes[1:,1],0]),'Strict parent-depth ancestry')
    counts=nodes[:,3].astype(np.int64)+nodes[:,4];require(int(counts.sum())==g-tail,'Tree storage count/tail')
    positive=np.flatnonzero(counts);order=positive[np.argsort(nodes[positive,2],kind='stable')]
    require(np.array_equal(nodes[order,2],np.cumsum(counts[order])-counts[order]),'Exclusive dense storage ranges')
    cc=nodes[:,6];cs=nodes[:,5];require(int(cc.sum())==n-1 and np.all((cc==0)|((cs>=1)&(cs.astype(np.int64)+cc<=n))),'Child ranges/edge count')
    prefix=np.cumsum(cc,dtype=np.int64)-cc
    children=(np.repeat(cs.astype(np.int64)-prefix,cc)+np.arange(n-1,dtype=np.int64)).astype(np.int32)
    owners=np.repeat(np.arange(n,dtype=np.int32),cc)
    require(np.array_equal(nodes[children,1],owners) and np.array_equal(np.sort(children),np.arange(1,n,dtype=np.int32)),'Actual child-parent partition')
    del children,owners,positive,order,prefix
    stats={};alpha_gt1=0
    for name,a in arrays.items():
        lo,hi=float('inf'),float('-inf')
        for b in range(0,g,65536):
            v=np.asarray(a[b:b+65536],np.float32);require(np.isfinite(v).all(),'Nonfinite actual stored '+name);lo=min(lo,float(v.min()));hi=max(hi,float(v.max()))
            if name=='rotations':require(np.all(np.linalg.norm(v,axis=1)>0),'Zero actual quaternion')
            if name=='log_scales':
                with np.errstate(over='ignore',under='ignore'):scale=np.exp(v)
                require(np.isfinite(scale).all() and np.all(scale>0),'Scale exp invalid')
            if name=='opacities':require(np.all(v>=0),'Negative activated alpha');alpha_gt1+=int((v>1).sum())
        stats[name]=dict(min=lo,max=hi,finite=True)
    for b in range(0,n,65536):
        v=boxes[b:b+65536];require(np.isfinite(v[:,:,:3]).all() and np.all(v[:,0,:3]<=v[:,1,:3]) and not np.isnan(v[:,:,3]).any() and np.all(v[:,0,3]>=0),'Actual AABB numeric condition')
    h=codec.Hierarchy(path,True,arrays['positions'],arrays['rotations'],arrays['log_scales'],arrays['opacities'],arrays['sh'],nodes,boxes,tail)
    return h,dict(fields=stats,activated_alpha_above_one=alpha_gt1,storage_exclusive=True,actual_parent_child_tree_valid=True,expected_unowned_tail=tail)

def classify(h,anchors,output,prefix):
    p=len(h.positions);n=len(h.nodes);nodes=h.nodes
    require(len(np.unique(anchors))==len(anchors),'Duplicate actual anchor index')
    anchored=np.zeros(p,np.bool_);anchored[anchors]=True
    counts=nodes[:,3].astype(np.int64)+nodes[:,4];starts=nodes[:,2].astype(np.int64)
    cumulative=np.concatenate([np.zeros(1,np.int64),np.cumsum(anchored,dtype=np.int64)])
    membership=cumulative[starts+counts]-cumulative[starts]
    require(np.all((membership==0)|(membership==counts)),'Partial positive owned node range in anchors')
    node_anchored=(counts>0)&(membership==counts)
    inherited=np.zeros(n,np.bool_)
    for depth in range(int(nodes[0,0])-1,-1,-1):
        indices=np.flatnonzero(nodes[:,0]==depth);parents=nodes[indices,1]
        inherited[indices]=inherited[parents]|node_anchored[parents]
    bad=np.flatnonzero(inherited&(counts>0)&~node_anchored)
    require(len(bad)==0,'Actual anchored positive-G ancestor has unanchored descendant')
    leafcounts=nodes[:,3].astype(np.int64);lcount=int(leafcounts.sum());offset=np.cumsum(leafcounts)-leafcounts
    leafids=np.repeat(starts-offset,leafcounts)+np.arange(lcount,dtype=np.int64)
    leaf=np.zeros(p,np.bool_);leaf[leafids]=True;require(int(leaf.sum())==lcount,'Actual leaf storage overlaps')
    intersection=np.flatnonzero(anchored&leaf);missing=np.flatnonzero(leaf&~anchored);summary=np.flatnonzero(anchored&~leaf)
    frontier=np.flatnonzero(node_anchored&~inherited)
    proof=output/(prefix+'-actual-anchor-sets.npz')
    np.savez_compressed(proof,anchors_sorted=np.flatnonzero(anchored),leaves_sorted=np.flatnonzero(leaf),anchor_intersection_leaf=intersection,leaf_minus_anchor=missing,anchor_minus_leaf=summary,positive_anchored_node_ids=np.flatnonzero(node_anchored),positive_anchor_frontier_node_ids=frontier)
    record=dict(status='actual_node_range_and_descendant_anchor_contract_passed',G=p,nodes=n,A=set_record(np.flatnonzero(anchored)),L=set_record(np.flatnonzero(leaf)),A_intersection_L=set_record(intersection),L_minus_A=set_record(missing),A_minus_L=set_record(summary),positive_owned_anchored_nodes=int(node_anchored.sum()),descendant_closure_violations=len(bad),positive_anchor_frontier=set_record(frontier),all_leafs_anchored_measured=len(missing)==0,anchors_equal_leaves_measured=bool(np.array_equal(anchored,leaf)),whole_owned_node_ranges=True,actual_set_npz_path=str(proof),actual_set_npz_sha256=sha(proof),source_contract='writeAnchors bottomRec(seen), andBelowRec(marked), all local count_leafs+count_merged in bottom-and-below nodes. Original seen/marked are not saved; frontier inferred from observable positive storage. Zero-G node membership itself is unobservable; ancestry propagates through it.',scope='Actual model-local Gaussian storage IDs, not original SfM track IDs, visibility/RGB quality or universal all-leaf freeze')
    return record,leaf

def xyz_multiset(source,tree):
    require(source.dtype==tree.dtype==np.float32 and source.shape==tree.shape,'Actual source/leaf XYZ shape differs')
    def ordered(a):
        b=np.ascontiguousarray(a).view(np.uint32).reshape(-1,3);order=np.lexsort((b[:,2],b[:,1],b[:,0]));return b[order]
    a,b=ordered(source),ordered(tree);same=bool(np.array_equal(a,b));require(same,'Actual Creator leaf XYZ multiset differs from source minus100k sky')
    return dict(actual_raw_float32_XYZ_bit_multiset_equal=True,rows=len(source),sorted_uint32_xyz_sha256=hashlib.sha256(a.tobytes()).hexdigest(),scope='Exact XYZ multiplicities only; not raw quaternion/scale-axis equality. Author Creator can permute equivalent ellipsoid axes and signs; compressed fields quantized.')

def leaf_changes(before,after,leaf):
    require(len(after.positions)==len(before.positions)+100000,'Only single appended post sky tail expected')
    fields={}
    for name in ('positions','rotations','log_scales','opacities','sh'):
        old,new=getattr(before,name),getattr(after,name);rows=0;changed=0;maximum=0.
        for b in range(0,len(leaf),65536):
            choose=leaf[b:b+65536];a=np.asarray(old[b:b+65536][choose],np.float32);c=np.asarray(new[b:b+65536][choose],np.float32)
            require(np.isfinite(a).all() and np.isfinite(c).all(),'Nonfinite measured pre/post leaf field')
            if len(a):
                changed+=int(np.any((a!=c).reshape(len(a),-1),axis=1).sum());maximum=max(maximum,float(np.max(np.abs(a-c))));rows+=len(a)
        fields[name]=dict(actual_leaf_rows=rows,changed_leaf_rows=changed,max_abs_decoded_difference=maximum,decoded_equal=changed==0)
    return dict(fields=fields,all_leaf_decoded_fields_equal=all(r['decoded_equal'] for r in fields.values()),scope='Measured chunk leaf endpoints; not assumed final merged topology/pixels or RGB quality.')

def creator_only(chunk,output,config,binding,codec,training,binder):
    source=next(r for r in binding['ordered_chunks'] if r['chunk']==chunk);directory=LOCK/'trained_chunks'/chunk
    receipt=read(directory/'creator-outcome.json');path=directory/'hierarchy.hier';anchorspath=directory/'anchors.bin'
    protected={str(p):sha(p) for p in (directory/'creator-outcome.json',path,anchorspath,directory/'exposure.json',Path(source['model']['path']),Path(source['exposure']['path']))}
    require(receipt['status']=='completed' and receipt['exit_code']==receipt['actual_child_poll_exit_code']==0 and receipt['source_kind']=='unoptimized' and receipt['configuration_sha256']==CONFIG_SHA and receipt['training_lineage_binding_sha256']==BINDING_SHA and receipt['formal_observation_reclassification'] is False,'Actual006 Creator origin/terminal receipt')
    require(receipt['training_origin']==source and receipt['source_training_receipt']==read(source['training_receipt']['path']) and receipt['input_model']['sha256']==source['model']['sha256']==sha(source['model']['path']),'Actual original model/receipt lineage differs')
    require(sha(path)==receipt['hierarchy_sha256'] and sha(anchorspath)==receipt['anchor_sha256'],'Actual Creator tree/anchors changed')
    archive=LOCK/'creator-camera-archives'/chunk;ar=read(archive/'manifest.json')
    for row in ar['files']:require(sha(row['path'])==row['sha256'],'Actual train-only pose archive changed')
    names=read(BASE/'official-inputs/small-city-training-002/manifest.json');split=next(c for c in names['chunks'] if c['name']==chunk)
    require(ar['train_only_pose_count']==len(split['train_names']) and ar['heldout_pose_count']==0 and ar['pose_64_byte_blocks_preserved_exactly'] and ar['camera_bytes_preserved_exactly'],'Train-only actual archive scope')
    e=read(directory/'exposure.json');original=read(source['exposure']['path']);values=np.asarray(list(e.values()),np.float64)
    require(e==original and set(e)==set(split['train_names']) and not set(e)&set(names['test_names']) and values.shape==(len(e),3,4) and np.isfinite(values).all(),'Actual exposure matrix/camera values differ')
    require(sha(directory/'exposure.json')==source['exposure']['sha256']==receipt['copied_exposure_sha256']==receipt['source_exposure']['sha256'],'Original→Creator actual exposure bytes edge')
    scaffold=Path(binding['coarse']['model']['path']).parent;saved=Path(source['model']['path']).parent
    sky=binder.sky_bits(scaffold,saved,training,codec)
    h,numeric=mapped(path,codec);anchors=codec.read_anchors(anchorspath,len(h.positions));require(len(anchors)==receipt['anchor_count'],'Actual anchor header differs from receipt')
    classification,leaf=classify(h,anchors,output,chunk)
    v=training.ply_vertices(source['model']['path']);xyz=np.stack([v[n][100000:] for n in ('x','y','z')],axis=1)
    matching=xyz_multiset(xyz,np.asarray(h.positions[leaf],np.float32));del xyz,v
    summary=h.summary();require(summary==receipt['hierarchy'],'Actual independently decoded Creator summary differs')
    require(all(sha(p)==d for p,d in protected.items()),'Original actual Creator/source bytes changed while audited')
    return dict(chunk=chunk,training_source_kind=source['training_source_kind'],actual_Gaussian_Adam_calls=source['actual_Gaussian_Adam_calls'],derived_Gaussian_call_upper_bound=source['derived_Gaussian_call_upper_bound'],formal_training_run=source['formal_training_run'],Creator_completed_only=True,formal_post_completed=False,source_model_sha256=source['model']['sha256'],actual_tree_sha256=sha(path),actual_anchor_sha256=sha(anchorspath),actual_hierarchy=summary,numeric_storage=numeric,actual_anchor_classification=classification,actual_leaf_source_XYZ=matching,actual_single_source_sky_raw_uint32=sky,actual_original_and_Creator_exposure_sha256=source['exposure']['sha256'],actual_exposure_count=len(e),actual_exposure_nonidentity_count=int(np.any(values!=np.eye(3,4),axis=(1,2)).sum()),post_exposure_edge='not_checked_no_formal_post_yet',protected_sources=protected,all_read_source_bytes_unchanged=True),h,leaf

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--mode',choices=['creator','formal-four'],required=True);p.add_argument('--chunk',choices=['0_0','0_1','1_0','1_1'],default='0_0');a=p.parse_args()
    require(a.run.replace('-','').isalnum() and len(a.run)<=110,'Unsafe output namespace');output=OUT/a.run;output.mkdir(exist_ok=False);shutil.copyfile(__file__,output/'source-snapshot.py')
    start=time.perf_counter();outcome=dict(status='running_independent_CPU006_anchor_ledger',CUDA_called=False,Torch_imported=False,mode=a.mode)
    save(output/'outcome.json',outcome)
    try:
        for path,digest in [(BINDING,BINDING_SHA),(LOCK/'configuration.json',CONFIG_SHA),(LOCK/'driver-source.py',NATIVE_SHA),(CODEC,'583c96f50b6f486d73bbbbaf9faf053ed15d486a7c7b48e15ef8e3ebc0964538'),(TRAIN,'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704'),(BINDER,'c72fea79bb3b083ead1dfa58e9cf7e7d0134842d764dd3c5482162448dc4db9e')]:pin(path,digest)
        binding,config=read(BINDING),read(LOCK/'configuration.json');require(binding['formal_observation_reclassification'] is False,'Observed source reclassified')
        if a.mode=='formal-four':
            required=[LOCK/'trained_chunks'/c/n for c in config['chunk_runs'] for n in ('creator-outcome.json','post-outcome.json','hierarchy.hier','hierarchy.hier_opt','anchors.bin','exposure.json')]
            require(all(x.is_file() for x in required),'Actual four006 I15000 post families missing; reject before models and never substitute pilot64')
        codec=load('ledger006_codec',CODEC);training=load('ledger006_raw_training_cpu',TRAIN);binder=load('ledger006_sky_uint32_cpu',BINDER)
        chunks=[a.chunk] if a.mode=='creator' else list(config['chunk_runs']);records=[]
        for chunk in chunks:
            r,before,leaf=creator_only(chunk,output,config,binding,codec,training,binder)
            if a.mode=='formal-four':
                directory=LOCK/'trained_chunks'/chunk;post=read(directory/'post-outcome.json')
                require(post['status']=='completed' and post['exit_code']==0 and post['source_kind']=='post_optimized' and post['requested_iterations']==15000 and post['optimizer_update_calls']==14999 and post['configuration_sha256']==CONFIG_SHA and post['training_lineage_binding_sha256']==BINDING_SHA,'Only actual I15000 same-source post allowed; pilot64 refused')
                require(post['exposure_optimizer_updates']==0 and not post['depth_loss_present_in_unchanged_post'] and post['frozen_exposure']['sha256']==r['actual_original_and_Creator_exposure_sha256']==sha(directory/'exposure.json'),'Post frozen original exposure edge differs')
                native=load('ledger006_post_argv_only',LOCK/'driver-source.py');native.ROOT=ROOT;native.OUTPUT=BASE;native.REPO=ROOT/'repos/hierarchical-3d-gaussians';native.TRAINING_PYTHON=ROOT/'.venv-large-gs/Scripts/python.exe'
                manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json');scaffold=Path(binding['coarse']['model']['path']).parent
                expected=native.post_command(directory,BASE/'official-inputs'/config['input']/'chunks'/chunk,scaffold,manifest,15000)
                require(post['command_argv']==expected and post.get('actual_child_poll_exit_code')==0,'Actual formal post argv/terminal differs from immutable official source')
                pin(HELPER,'325e3efd7a9a525fc88b556b2127e32f6f8b7c9628356aeb365f846ec673a029');sys.modules['read_hierarchy']=codec;helper=load('ledger006_actual_anchor_freeze',HELPER)
                after,numeric=mapped(directory/'hierarchy.hier_opt',codec,100000);require(sha(after.path)==post['hierarchy_sha256'],'Actual post bytes differ')
                anchors=codec.read_anchors(directory/'anchors.bin',len(before.positions));freeze=helper.post_freeze_audit(before,after,anchors,Path(binding['coarse']['model']['path']).parent,100000)
                require(freeze['status']=='frozen_anchor_and_skybox_fields_verified','Actual formal anchor/sky freeze failed')
                r.update(formal_post_completed=True,actual_formal_post_sha256=post['hierarchy_sha256'],actual_post_storage=numeric,actual_anchor_and_single_tail_freeze=freeze,actual_pre_post_leaf_fields=leaf_changes(before,after,leaf),post_exposure_edge='original→Creator→formalpost byte-identical',formal_post_optimizer_call_scope='14999 from unchanged unconditional source post loop, not a new runtime instrument')
            records.append(r);del before,leaf
        require('torch' not in sys.modules,'CPU ledger imported Torch')
        report=dict(status='completed_actual006_Creator_anchor_source_exposure_CPU_measurement_only' if a.mode=='creator' else 'completed_actual_four006_I15000_post_anchor_exposure_CPU_ledger',mode=a.mode,records=records,training_lineage_binding_sha256=BINDING_SHA,configuration_sha256=CONFIG_SHA,native_source_sha256=NATIVE_SHA,formal_observation_reclassification=False,actual_four_formal_posts_complete=a.mode=='formal-four',actual_merger_or1404_complete=False,CUDA_called=False,Torch_imported=False,units='Published SfM aligned units, not GT and not metres',source_sha256=sha(__file__))
        save(output/'report.json',report);outcome.update(status=report['status'],report_sha256=sha(output/'report.json'),elapsed_seconds=time.perf_counter()-start,actual_four_formal_posts_complete=a.mode=='formal-four');save(output/'outcome.json',outcome)
        print(json.dumps(dict(status=report['status'],records=[dict(chunk=r['chunk'],A_intersection_L=r['actual_anchor_classification']['A_intersection_L']['count'],L_minus_A=r['actual_anchor_classification']['L_minus_A']['count'],A_minus_L=r['actual_anchor_classification']['A_minus_L']['count']) for r in records],report_sha256=sha(output/'report.json'),CUDA_called=False,Torch_imported=False)),flush=True)
    except Exception as exc:
        outcome.update(status='failed_preserved_independent_CPU006_anchor_ledger',error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,Torch_imported='torch' in sys.modules,actual_four_formal_posts_complete=False);save(output/'outcome.json',outcome);raise

if __name__=='__main__':main()
