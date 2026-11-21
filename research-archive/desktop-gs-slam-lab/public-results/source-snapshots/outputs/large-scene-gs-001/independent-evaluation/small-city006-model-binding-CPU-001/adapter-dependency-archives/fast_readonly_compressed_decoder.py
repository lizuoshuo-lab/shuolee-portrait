"""CPU-only official SmallCity source/frame/cameras/masks/memory readiness."""
from pathlib import Path
import json,sys,struct,time,traceback
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
import evaluate_official_hierarchy as evaluator
from read_hierarchy import Hierarchy,official_size_proxy_cpu
from plyfile import PlyData
save,digest,require=evaluator.save,evaluator.digest,evaluator.require
OUTPUT=Path(__file__).resolve().parent
TRACK=evaluator.TRACK

def mapped_tree(path):
    with path.open('rb') as stream:signed=struct.unpack('<i',stream.read(4))[0]
    require(signed<0,'This readiness expects verified compressed author model')
    g=-signed;node_offset=4+124*g
    with path.open('rb') as stream:stream.seek(node_offset);n=struct.unpack('<i',stream.read(4))[0]
    require(path.stat().st_size==node_offset+4+36*n,'Independent compressed payload length mismatch')
    position=np.memmap(path,dtype='<f4',mode='r',offset=4,shape=(g,3))
    rotation=np.memmap(path,dtype='<f2',mode='r',offset=4+12*g,shape=(g,4))
    logs=np.memmap(path,dtype='<f2',mode='r',offset=4+20*g,shape=(g,3))
    alpha=np.memmap(path,dtype='<f2',mode='r',offset=4+26*g,shape=(g,))
    sh=np.memmap(path,dtype='<f2',mode='r',offset=4+28*g,shape=(g,16,3))
    dtype=np.dtype([('parent','<i4'),('start','<i4'),('children_start','<i4'),('packed','<i2',(4,))])
    packed=np.memmap(path,dtype=dtype,mode='r',offset=node_offset+4,shape=(n,))
    nodes=np.empty((n,7),np.int32)
    nodes[:,0]=packed['packed'][:,0];nodes[:,1]=packed['parent'];nodes[:,2]=packed['start']
    nodes[:,3]=packed['packed'][:,2];nodes[:,4]=packed['packed'][:,3];nodes[:,5]=packed['children_start'];nodes[:,6]=packed['packed'][:,1]
    boxes=np.memmap(path,dtype='<f2',mode='r',offset=node_offset+4+20*n,shape=(n,2,4))
    h=Hierarchy(path,True,position,rotation,logs,alpha,sh,nodes,boxes)
    require(nodes[0,1]==-1 and np.all((nodes[1:,1]>=0)&(nodes[1:,1]<n)),'Invalid parent')
    require(np.all(nodes[1:,0]<nodes[nodes[1:,1],0]),'Parent depth must strictly increase')
    count=nodes[:,3].astype(np.int64)+nodes[:,4]
    require(np.all(count>=0) and int(count.sum())==g,'Unowned sky tail or wrong storage count')
    positive=np.flatnonzero(count);order=positive[np.argsort(nodes[positive,2],kind='stable')]
    require(np.array_equal(nodes[order,2],np.cumsum(count[order])-count[order]),'Overlapping or gapped dense Gaussian ranges')
    childcounts=nodes[:,6];require(np.all(childcounts>=0) and int(childcounts.sum())==n-1,'Child edge count wrong')
    childstarts=nodes[:,5];require(np.all((childcounts==0)|((childstarts>=1)&(childstarts.astype(np.int64)+childcounts<=n))),'Child range invalid')
    prefix=np.cumsum(childcounts,dtype=np.int64)-childcounts
    children=(np.repeat(childstarts.astype(np.int64)-prefix,childcounts)+np.arange(n-1,dtype=np.int64)).astype(np.int32)
    owners=np.repeat(np.arange(n,dtype=np.int32),childcounts)
    require(np.array_equal(nodes[children,1],owners),'Child-parent ownership disagrees')
    require(np.array_equal(np.sort(children),np.arange(1,n,dtype=np.int32)),'Missing/duplicate child edges')
    del children,owners,count,positive,order,prefix
    stats={name:dict(min=None,max=None,rows=0) for name in ('position','rotation','log_scales','activated_alpha','SH')}
    alpha_gt1=0
    for begin in range(0,g,65536):
        end=begin+65536
        for name,array in zip(stats,(position,rotation,logs,alpha,sh)):
            a=np.asarray(array[begin:end],np.float32);require(np.isfinite(a).all(),'Nonfinite '+name)
            row=stats[name];lo=float(a.min());hi=float(a.max());row['min']=lo if row['min'] is None else min(row['min'],lo);row['max']=hi if row['max'] is None else max(row['max'],hi);row['rows']+=len(a)
        q=np.asarray(rotation[begin:end],np.float32);require(np.all(np.linalg.norm(q,axis=1)>0),'Zero raw quaternion')
        scales=np.exp(np.asarray(logs[begin:end],np.float32));require(np.isfinite(scales).all() and np.all(scales>0),'Invalid scale exp')
        require(np.all(alpha[begin:end]>=0),'Negative stored activated alpha');alpha_gt1+=int(np.count_nonzero(alpha[begin:end]>1))
    for begin in range(0,n,65536):
        box=boxes[begin:begin+65536];require(np.isfinite(box[:,:,:3]).all() and np.all(box[:,0,:3]<=box[:,1,:3]),'Invalid AABB')
        require(not np.isnan(box[:,:,3]).any() and np.all(box[:,0,3]>=0),'Invalid AABB size metadata')
    save(OUTPUT/'independent-model-storage-audit.json',dict(status='passed_vectorized_independent_compressed_storage_tree_audit',
        source_path=str(path),source_sha256=digest(path),summary=h.summary(),fields=stats,activated_alpha_gt_one=alpha_gt1,
        topology='Every non-root exactly one parent and child edge, strictly decreasing depth, root connected, dense exclusive Gaussian storage, no unowned tail',
        memorymap='Read-only r mode; no tensor GPU/author load; compressed nodes normalized in CPU only',cuda_called=False))
    return h

def main():
    started=time.perf_counter();outcome=dict(status='running_CPU_only',cuda_used=False)
    save(OUTPUT/'outcome.json',outcome)
    try:
        evaluator.source_audit(OUTPUT)
        pretrained_root=TRACK/'research/datasets/official-small-city-pretrained'
        manifest=baseline_json= evaluator.read_json(pretrained_root/'manifest.json')
        path=Path(manifest['hierarchy_header']['file']);require(digest(path)==manifest['hierarchy_header']['sha256'],'Author hierarchy hash differs')
        h=mapped_tree(path);leaf_count=int(h.nodes[:,3].sum(dtype=np.int64));sky_count=100000
        scaffold_path=Path(manifest['scaffold']['ply']);require(digest(scaffold_path)==manifest['scaffold']['sha256'],'Author scaffold changed')
        scaffold=PlyData.read(scaffold_path)['vertex'].data
        original_path=TRACK/'research/datasets/official-small-city/extracted/small_city/camera_calibration/aligned/sparse/0/points3D.ply'
        original=PlyData.read(original_path)['vertex'].data
        require(len(scaffold)-sky_count==len(original)==2390040,'Scaffold/source point count mismatch')
        for key in ('x','y','z'):require(np.array_equal(scaffold[key][sky_count:],original[key].astype(np.float32)),'No common scaffold/source global frame')
        save(OUTPUT/'source-frame-audit.json',dict(status='passed_exact_float32_rowwise_global_XYZ',source_points=len(original),
            scaffold_path=str(scaffold_path),scaffold_sha256=digest(scaffold_path),global_input_ply=str(original_path),global_input_ply_sha256=digest(original_path),
            rule='Scaffold removes exactly first pc_info100000 sky rows; three XYZ columns equal source by row after float32 conversion, no numeric-ID guessing',
            frame='Original author SfM frame retained; no local normalization, Sim3 or metre claim',skybox_append_count=sky_count,skybox_in_tree=0))
        archive=TRACK/'official-inputs/small-city-protocol-002/global-cameras.npz';z=np.load(archive);names=z['names'].astype(str)
        indices=[]
        for capture in ('pass1','pass2','pass3','pass4'):
            group=np.flatnonzero(np.char.startswith(names,capture+'_'));group=group[np.argsort(names[group],kind='stable')]
            indices.extend(group[np.rint(np.array([1/6,1/2,5/6])*(len(group)-1)).astype(int)].tolist())
        require(len(set(indices))==12,'Pilot selection duplicate')
        camera_scope=evaluator.official_camera_math_CPU();image_root=TRACK/'research/datasets/official-small-city/extracted/small_city/camera_calibration/rectified/images'
        derived=OUTPUT/'derived-inputs';derived.mkdir();rows=[];cameras=[];all_masks=[];all_RGB=[]
        lower_depth=h.nodes[:,0];parent=h.nodes[:,1];leaves=h.nodes[:,3];merged=h.nodes[:,4]
        all_setting_counts={name:[] for name,_ in evaluator.SETTINGS}
        for j,index in enumerate(indices):
            name=names[index];image_path=image_root/name;mask_path=image_root.parent/'masks'/(Path(name).stem+'.png')
            with Image.open(image_path) as im:require(im.mode=='RGB' and im.size==(1024,690),'Unexpected source RGB');im.load();rgb=np.asarray(im.resize((256,172))).copy()
            with Image.open(mask_path) as im:require(im.mode=='L' and im.size==(1024,690),'Unexpected source dynamic mask');im.load();mask8=np.asarray(im.resize((256,172))).copy()
            Image.fromarray(rgb).save(derived/(Path(name).stem+'-RGB.png'));Image.fromarray(mask8).save(derived/(Path(name).stem+'-mask.png'))
            mask=mask8.astype(np.float32)[:,:,None]/255.;all_masks.append(mask);all_RGB.append(rgb)
            K=z['K'][index].copy();K[0,:]*=256/1024;K[1,:]*=172/690
            cam=evaluator.make_official_camera_CPU(K,z['c2w'][index],256,172,name,camera_scope);cameras.append(cam)
            require(np.max(np.abs(cam.camera_center.numpy()-z['c2w'][index,:3,3]))<1e-4,'City camera center/global frame mismatch')
            sizes=official_size_proxy_cpu(h,cam.camera_center.numpy());scopes={}
            for setting,tau in evaluator.SETTINGS:
                scope=evaluator.threshold_scope(tau,float(K[0,0]));scopes[setting]=scope
                if tau is None:count=leaf_count;root_failure=False
                else:
                    active=sizes>=scope['threshold'];parent_active=np.zeros(len(active),np.bool_);parent_active[1:]=active[parent[1:]]
                    crossing=(~active)&parent_active
                    count=int(leaves[active].sum(dtype=np.int64)+leaves[crossing].sum(dtype=np.int64)+merged[crossing&(lower_depth!=0)].sum(dtype=np.int64))
                    root_failure=bool(sizes[0]<scope['threshold'])
                scope.update(CPU_predicate_selected_tree_G=count,root_fallback_failure=root_failure);all_setting_counts[setting].append(count)
            row=dict(name=name,capture=name.split('_')[0],global_camera_row=index,original_dimensions=[1024,690],derived_dimensions=[256,172],
                original_K=z['K'][index].tolist(),derived_K=K.tolist(),C2W=z['c2w'][index].tolist(),source_image=str(image_path),source_image_sha256=digest(image_path),
                source_mask=str(mask_path),source_mask_sha256=digest(mask_path),mask_mean=float(mask.mean()),mask_nonzero_pixels=int((mask>0).sum()),
                mask_full_one_pixels=int((mask==1).sum()),mask_squared_weight_sum=float(np.sum(mask.astype(np.float64)**2)),setting_scope=scopes)
            rows.append(row);save(OUTPUT/'camera-mask-memory-per-view.json',rows)
            print(json.dumps(dict(name=name,counts={s:d['CPU_predicate_selected_tree_G'] for s,d in scopes.items()})),flush=True)
            del sizes
        np.savez(OUTPUT/'pilot-inputs.npz',names=names[indices],source_camera_rows=np.array(indices),rgb=np.stack(all_RGB),mask=np.stack(all_masks),
            c2w=z['c2w'][indices],K=np.stack([np.array(row['derived_K']) for row in rows]))
        plans={setting:evaluator.memory_plan(len(h.positions)+sky_count,len(h.nodes),leaf_count,max(counts),sky_count,12)
               for setting,counts in all_setting_counts.items()}
        for plan in plans.values():plan['GPU_allocation_decision']='No CUDA queried/allocated; compare known upper with actual free GPU plus native workspace at authorized runtime'
        save(OUTPUT/'actual-memory-plans.json',plans)
        protocol=dict(status='passed_CPU_city_reconstruction_pilot_readiness',views=rows,selection='Each lexical capture-pass list positions nearest1/6,1/2,5/6, frozen here:12 views',
            global_camera_archive=str(archive),global_camera_archive_sha256=digest(archive),pretrained_manifest_sha256=digest(pretrained_root/'manifest.json'),
            model_sha256=manifest['hierarchy_header']['sha256'],training_split_unknown=True,scope='Author pretrained reconstruction-view inspection; not heldout metrics or local reproduction',
            resize='PIL RGB/L default resize256x172; K scaled independently x256/1024,y172/690; source originals preserved',
            mask_chain='Pinned Camera masks GT once, render_hierarchy eval masks GT again/pred once:GTm2/predm. Main full RGB denominator3WH; separately symmetric valid squared-weight MSE denominator3sum(m^2).',
            metrics='Fixed previous Truck TorchMetrics1.7.1 SSIM/Alex LPIPS; not author VGG/paper values; no exposure files, correction disabled',
            coverage_limit='Readiness CPU predicates only; actual selected-node coverage, weights/finite renders and hashes remain required at authorized GPU evaluation',cuda_called=False)
        save(OUTPUT/'protocol.json',protocol)
        (OUTPUT/'report.md').write_text('# SmallCity 官方预训练模型 CPU 就绪审查\n\n实际树为18,598,279个高斯、18,598,276节点，所有存储均归树；模型之外仅追加scaffold前100,000天空高斯一次。去天空后的2,390,040行XYZ与原global points3D.ply在float32按行完全一致。\n\n固定四个pass各3张，共12个重建视角，256×172；原训练划分未提供，不能声称留出评价或本地训练复现。真实CPU cut数与固定模型、缓存、screenspace、参数插值显存上界见actual-memory-plans.json；native tile重复与排序开销另计，未盲加载GPU。\n\n动态遮罩按固定官方Camera及render_hierarchy链路记录GT乘两次m、pred乘一次m，全图误差分母3WH。辅助对称单次mask有效权重PSNR使用3sum(m²)分母，不能与论文曝光/遮罩/VGG/未知test条件直接对标。CPU尚未渲染或计算模型图像成绩。\n',encoding='utf-8')
        outcome.update(status=protocol['status'],elapsed_seconds=time.perf_counter()-started,script_sha256=digest(__file__))
        save(OUTPUT/'outcome.json',outcome)
    except Exception as error:
        outcome.update(status='failed_preserved_CPU_readiness',error=str(error),traceback=traceback.format_exc());save(OUTPUT/'outcome.json',outcome);raise

if __name__=='__main__':main()
