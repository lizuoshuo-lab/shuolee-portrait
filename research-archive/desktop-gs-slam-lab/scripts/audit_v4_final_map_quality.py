"""Independent CPU arithmetic/input audit of frozen-map future-camera renders.

No evaluator functions, renderer, optimization, registration or CUDA execution.
GT is read only for this posthoc known-camera artifact audit. Sensor association
uses the frozen loader; RGB/depth decoding, rigid gauge and metric arithmetic
are explicitly checked here. Old experiment outputs are never rewritten.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tum-v4-final-map-future10-001'
PROTOCOL = ROOT / 'outputs/tum-v4-final-map-future10-cpu-check-003/protocol.json'
PROTOCOL_SHA = '264ba4481cac19a1a2336481634a59b96a3bb123fc597dcbaafede05cf1b48be'
EVALUATOR = ROOT / 'scripts/evaluate_tum_v4_final_maps.py'
EVALUATOR_SHA = '0e6d8d976c924bf9ce492d2a405735d7cb0f4288b7da68a6518bf6589f970896'
EVAL_IDS = list(range(300,310))
TRAIN_IDS = list(range(200,300))
DIFFERENCES = []


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def array_record(value):
    if isinstance(value,torch.Tensor):
        value=value.detach().cpu().numpy()
    a=np.ascontiguousarray(value)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,
            'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def compare(saved, actual, label):
    require(type(saved) is type(actual), label+' type')
    if isinstance(actual,dict):
        require(set(saved)==set(actual), label+' fields')
        for key in actual: compare(saved[key],actual[key],label+'/'+key)
    elif isinstance(actual,list):
        require(len(saved)==len(actual),label+' retained items')
        for i,(a,b) in enumerate(zip(saved,actual)):compare(a,b,label+'/'+str(i))
    elif isinstance(actual,float):
        require(math.isfinite(saved) and math.isfinite(actual),label+' finite')
        require(math.isclose(saved,actual,abs_tol=2e-10,rel_tol=2e-13),label+' value')
        DIFFERENCES.append({'path':label,'absolute_difference':abs(saved-actual)})
    else:
        require(saved==actual,label+' exact value')


def inverse_rigid(pose):
    out=np.eye(4,dtype=np.float64)
    out[:3,:3]=pose[:3,:3].T
    out[:3,3]=-pose[:3,:3].T@pose[:3,3]
    return out


def independent_tum_pose(row):
    numbers=np.array(row[1:],dtype=np.float64);v=numbers[3:6];w=numbers[6]
    norm=math.sqrt(float(v@v+w*w));v=v/norm;w=w/norm
    x,y,z=v
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    out=np.eye(4);out[:3,:3]=(w*w-float(v@v))*np.eye(3)+2*np.outer(v,v)+2*w*skew
    out[:3,3]=numbers[:3]
    return out


def depth_statistics(prediction, observed, selected):
    errors=np.asarray(prediction,dtype=np.float64)[selected]-np.asarray(observed,dtype=np.float64)[selected]
    count=int(np.count_nonzero(selected))
    absolute=float(np.sum(np.abs(errors),dtype=np.float64))
    squared=float(np.sum(errors*errors,dtype=np.float64))
    return {'pixels':count,'absolute_error_sum_m':absolute,'squared_error_sum_m2':squared,
            'mae_m':absolute/count if count else None,'rmse_m':math.sqrt(squared/count) if count else None}


def psnr(mse):
    return {'mse':mse,'psnr_db':-10*math.log10(mse) if mse else None,'psnr_positive_infinity':mse==0.}


def measure(rgb,depth,alpha,reference_rgb,reference_depth,valid):
    require(rgb.shape==reference_rgb.shape==(120,160,3),'RGB array shape')
    require(depth.shape==alpha.shape==reference_depth.shape==valid.shape==(120,160),'depth/mask array shape')
    require(valid.dtype==np.bool_ and np.count_nonzero(valid)>0,'sensor valid count')
    require(all(np.isfinite(x).all() for x in (rgb,depth,alpha,reference_rgb,reference_depth)),'finite arrays')
    require(alpha.min()>=-1e-6 and alpha.max()<=1+1e-6,'alpha range')
    covered=valid & (alpha>0.5) & (depth>0) & np.isfinite(depth)
    delta=rgb.astype(np.float64)-reference_rgb.astype(np.float64)
    squared=float(np.sum(delta*delta,dtype=np.float64));channels=int(delta.size)
    rgb_values={'squared_error_sum':squared,'channel_values':channels,**psnr(squared/channels)}
    # Holes are explicitly zero depth for every sensor-valid pixel outside cover.
    with_holes=np.zeros_like(depth);with_holes[covered]=depth[covered]
    n=int(np.count_nonzero(valid));c=int(np.count_nonzero(covered))
    return {'rgb':rgb_values,'valid_depth_pixels':n,'covered_valid_pixels':c,
            'sensor_valid_coverage':c/n,'covered_depth':depth_statistics(depth,reference_depth,covered),
            'all_valid_holes_as_zero_depth':depth_statistics(with_holes,reference_depth,valid)},covered


def mean(values):
    return math.fsum(values)/len(values)


def defined_mean(values):
    valid=[v for v in values if v is not None]
    return {'value':mean(valid) if valid else None,'defined_view_count':len(valid),
            'undefined_view_count':len(values)-len(valid)}


def pooled_depth(rows):
    count=sum(r['pixels'] for r in rows)
    absolute=math.fsum(r['absolute_error_sum_m'] for r in rows)
    squared=math.fsum(r['squared_error_sum_m2'] for r in rows)
    return {'pixels':count,'absolute_error_sum_m':absolute,'squared_error_sum_m2':squared,
            'mae_m':absolute/count if count else None,'rmse_m':math.sqrt(squared/count) if count else None}


def aggregate(rows,completed):
    n=len(completed);result={'requested_views':10,'completed_views':n,
        'failed_views':sum(v['status']=='failed' for v in rows),'not_completed_views':10-n,
        'summary_scope':'successful views only; missing/failed views are not filled with zero'}
    if not n:return result
    sse=math.fsum(v['rgb']['squared_error_sum'] for v in completed)
    channel_count=sum(v['rgb']['channel_values'] for v in completed)
    sensors=sum(v['valid_depth_pixels'] for v in completed);covered=sum(v['covered_valid_pixels'] for v in completed)
    result['pooled']={'rgb':{'squared_error_sum':sse,'channel_values':channel_count,**psnr(sse/channel_count)},
        'sensor_valid_pixels':sensors,'covered_valid_pixels':covered,'sensor_valid_coverage':covered/sensors,
        'covered_depth':pooled_depth([v['covered_depth'] for v in completed]),
        'all_valid_holes_as_zero_depth':pooled_depth([v['all_valid_holes_as_zero_depth'] for v in completed])}
    infinite=any(v['rgb']['psnr_positive_infinity'] for v in completed)
    result['mean_per_view']={'rgb_mse':mean([v['rgb']['mse'] for v in completed]),
        'rgb_psnr_db':None if infinite else mean([v['rgb']['psnr_db'] for v in completed]),
        'rgb_psnr_positive_infinity':infinite,'sensor_valid_coverage':mean([v['sensor_valid_coverage'] for v in completed]),
        'covered_depth_mae_m':defined_mean([v['covered_depth']['mae_m'] for v in completed]),
        'covered_depth_rmse_m':defined_mean([v['covered_depth']['rmse_m'] for v in completed]),
        'all_valid_holes_as_zero_rmse_m':mean([v['all_valid_holes_as_zero_depth']['rmse_m'] for v in completed])}
    return result


def cpu_controls():
    target=np.array([[1.,2.,3.,4.]]);prediction=np.array([[1.1,9.,2.8,4.]])
    alpha=np.array([[1.,.5,.8,.2]]);valid=np.ones((1,4),dtype=bool)
    covered=valid&(alpha>.5)&(prediction>0)
    a=depth_statistics(prediction,target,covered)
    zero=np.zeros_like(prediction);zero[covered]=prediction[covered]
    h=depth_statistics(zero,target,valid)
    require(a['pixels']==2 and math.isclose(a['squared_error_sum_m2'],.05,abs_tol=1e-14),'strict .5 depth handcheck')
    require(math.isclose(h['squared_error_sum_m2'],20.05,abs_tol=1e-14),'holes-as-zero handcheck')
    require(depth_statistics(prediction,target,np.zeros_like(valid))['rmse_m'] is None,'empty coverage null')
    require(psnr(0.)['psnr_positive_infinity'] and psnr(0.)['psnr_db'] is None,'perfect RGB infinity encoding')
    pooled=pooled_depth([{'pixels':1,'absolute_error_sum_m':1.,'squared_error_sum_m2':1.},
                        {'pixels':8,'absolute_error_sum_m':0.,'squared_error_sum_m2':0.}])
    require(pooled['rmse_m']==1/3 and pooled['rmse_m']!=mean([1.,0.]),'pooled versus mean distinguishing control')
    return {'strict_alpha_threshold_hand_sse':a['squared_error_sum_m2'],
            'holes_as_zero_hand_sse':h['squared_error_sum_m2'],'empty_is_null':True,'perfect_rgb_encoded_infinity':True}


def load_inputs(scene,record):
    from tum_rgbd import associate_frames
    dataset=ROOT/record['dataset'];frames,association=associate_frames(dataset,.02,.02,32.)
    require(record['association']==association,'association '+scene)
    train=frames[200:300];evaluation=frames[300:310]
    require(len(train)==100 and len(evaluation)==10,'data frame counts')
    for group,name in ((train,'training_sensor_metadata'),(evaluation,'evaluation_sensor_metadata')):
        rows=[{k:v for k,v in f.metadata().items() if k not in('c2w','w2c')} for f in group]
        require(rows==record[name],'exact sensor metadata '+scene+'/'+name)
    for field in ('frame_id','rgb_index','depth_index','rgb_path','depth_path'):
        left={getattr(f,field)for f in train};right={getattr(f,field)for f in evaluation}
        require(len(left)==100 and len(right)==10 and not left&right,'training/future disjoint '+field)
    for filename,digest in record['index_file_sha256'].items(): require(sha(dataset/filename)==digest,'index SHA')
    for row in record['raw_sensor_files']:
        for field in ('rgb_path','depth_path'):
            evidence=row[field];file=dataset/evidence['path']
            require(file.resolve().is_relative_to(dataset.resolve()) and sha(file)==evidence['sha256'] and
                    file.stat().st_size==evidence['bytes'],'raw PNG hash')
    images=[];depths=[];masks=[]
    for frame in evaluation:
        with Image.open(dataset/frame.rgb_path)as source:
            require(source.size==(640,480),'raw RGB size')
            images.append(np.asarray(source.convert('RGB').resize((160,120),Image.Resampling.BILINEAR),dtype=np.float32)/np.float32(255.))
        with Image.open(dataset/frame.depth_path)as source:
            require(source.size==(640,480),'raw depth size')
            depths.append(np.asarray(source.resize((160,120),Image.Resampling.NEAREST),dtype=np.float32)/np.float32(5000.))
        masks.append(np.isfinite(depths[-1])&(depths[-1]>=.2)&(depths[-1]<=5.))
    rgb=np.stack(images);depth=np.stack(depths);valid=np.stack(masks)
    rawK=np.array([[517.3,0.,318.6],[0.,516.5,255.3],[0.,0.,1.]])
    resize=np.array([[.25,0.,-.375],[0.,.25,-.375],[0.,0.,1.]])
    K=resize@rawK;anchor=train[0].c2w;absolute=np.stack([f.c2w for f in evaluation])
    relative=np.linalg.inv(anchor)[None]@absolute;w2c=np.linalg.inv(relative)
    tensors={'rgb':rgb,'depth':depth,'valid_depth_mask':valid,'K_float64':K,'K_render_float32':K.astype(np.float32)[None],
        'anchor_gt_c2w_float64':anchor,'evaluation_gt_c2w_absolute_float64':absolute,
        'evaluation_c2w_relative_float64':relative,'evaluation_w2c_float64':w2c,
        'evaluation_w2c_render_float32':w2c.astype(np.float32)}
    require({key:array_record(value)for key,value in tensors.items()}==record['tensors'],'decoded inputs/camera bytes '+scene)
    gt_rows=[line.split()for line in (dataset/'groundtruth.txt').read_text().splitlines()if line.strip()and not line.lstrip().startswith('#')]
    independent=np.stack([independent_tum_pose(gt_rows[f.gt_index])for f in evaluation]);independent_anchor=independent_tum_pose(gt_rows[train[0].gt_index])
    independent_rel=inverse_rigid(independent_anchor)[None]@independent
    independent_w2c=np.stack([inverse_rigid(p)for p in independent_rel])
    quaternion_error=float(np.max(np.abs(independent-absolute)))
    gauge_error=float(np.max(np.abs(independent_w2c-w2c)))
    require(max(quaternion_error,gauge_error)<1e-12,'independent xyzw/R.T gauge')
    wrong=np.stack([p@np.linalg.inv(anchor)for p in absolute])
    wrong_error=float(np.max(np.abs(wrong-relative)))
    require(wrong_error>1e-3,'actual gauge order negative control distinguishes wrong order')
    return tensors,{'scene':scene,'raw_png_count_checked':len(record['raw_sensor_files'])*2,
        'all_decoded_input_and_protocol_camera_array_sha_exact':True,'independent_normalized_xyzw_rotation_max_difference':quaternion_error,
        'independent_analytic_rigid_inverse_w2c_max_difference':gauge_error,'wrong_gauge_order_max_difference':wrong_error,
        'valid_depth_pixels':valid.sum(axis=(1,2)).tolist(),'rgb_input_min_max':[float(rgb.min()),float(rgb.max())],
        'depth_units':'uint16/5000 -> meters; inclusive sensor valid [0.2,5.0]m',
        'camera_formula':'c2w_relative=inv(GT_c2w_200)@GT_c2w_eval; w2c=inv(c2w_relative); no fitted alignment/map transform'}


def source_review(protocol):
    require(sha(EVALUATOR)==EVALUATOR_SHA,'frozen evaluator SHA')
    require(protocol['sources']['evaluation_script_sha256']==EVALUATOR_SHA,'protocol evaluator SHA')
    for name,digest in protocol['sources']['source_sha256'].items():require(sha(ROOT/'scripts'/name)==digest,'frozen source '+name)
    v4=ast.parse((ROOT/'scripts/train_tum_online_gs_v4.py').read_text())
    main=next(n for n in v4.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    render=next(n for n in main.body if isinstance(n,ast.FunctionDef)and n.name=='render')
    require(hashlib.sha256(ast.dump(render,include_attributes=False).encode()).hexdigest()==protocol['sources']['render_ast_sha256'],'original renderer AST')
    tree=ast.parse(EVALUATOR.read_text());execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='execute_protocol')
    text=ast.unparse(execute)
    for snippet in ('with torch.no_grad()', 'before == frozen[\'parameter_tensors\']', 'before == after',
                    "file_sha(checkpoint_path) == frozen['checkpoint_sha256']", 'value.detach().clone().to(\'cuda\')'):
        require(snippet in text,'reviewed renderer no_grad/rawhash guard '+snippet)
    require(not any(isinstance(n,ast.Attribute)and n.attr in('backward','step')for n in ast.walk(execute)), 'no optimizer calls')
    return {'evaluator_sha256':EVALUATOR_SHA,'original_v4_render_ast_sha256':protocol['sources']['render_ast_sha256'],
        'no_grad_and_detached_cloned_upload':True,'upload_parameter_bytes_equals_protocol_guard':True,
        'render_parameter_before_after_bytes_guard':True,'original_checkpoint_before_after_file_hash_guard':True,
        'guards_scope':'Read-only source review plus current actual checkpoint/protocol-byte equality; GPU before/after tensors are not rerendered by this audit.'}


def main():
    require(sha(PROTOCOL)==PROTOCOL_SHA and (OUT/'protocol_used.json').read_bytes()==PROTOCOL.read_bytes(),'reviewed exact protocol bytes')
    protocol=read(PROTOCOL);metrics=read(OUT/'metrics.json')
    require(metrics==read(OUT/'progress.json'),'final progress and metrics agree')
    require(metrics['protocol_sha256']==PROTOCOL_SHA and metrics['sources']==protocol['sources'] and metrics['settings']==protocol['settings'],'protocol result provenance')
    require(metrics['status']=='completed' and metrics['completed_map_count']==12 and metrics['completed_render_count']==120 and
            metrics['optimization_steps']==0 and metrics['known_cameras_use_gt'] is True and metrics['gt_used_for_fitting_or_map_updates'] is False,'complete eval scope')
    require(len(metrics['runs'])==len(protocol['runs'])==12 and len(metrics['pairs'])==6,'all maps/pairs retained')
    source=source_review(protocol);checks=cpu_controls();inputs={};dataset_checks=[]
    for scene in ('desk','xyz'): inputs[scene],record=load_inputs(scene,protocol['datasets'][scene]);dataset_checks.append(record)
    run_audits=[];predictions={};view_count=0
    for frozen,row in zip(protocol['runs'],metrics['runs']):
        name=row['run'];scene=row['scene'];checkpoint_path=ROOT/frozen['checkpoint']
        require({k:row[k]for k in('run','scene','seed','prior_weight','checkpoint_sha256')}==
                {k:frozen[k]for k in('run','scene','seed','prior_weight','checkpoint_sha256')},'map protocol identity')
        require(sha(checkpoint_path)==frozen['checkpoint_sha256'] and checkpoint_path.stat().st_size==frozen['checkpoint_bytes'],'actual checkpoint file bytes')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        require(checkpoint['ground_truth_included'] is False and not any('evaluation_only_gt' in k for k in checkpoint),'checkpoint GT free')
        require(checkpoint['configuration']==frozen['configuration_expected'] and checkpoint['frame_ids_with_stored_pose']==TRAIN_IDS and
                checkpoint['current_local_index']==99 and [v['frame_id']for v in checkpoint['completed_frame_metrics']]==TRAIN_IDS,'all training frames retained')
        require({key:array_record(value)for key,value in checkpoint['parameters'].items()}==frozen['parameter_tensors'],'actual parameter raw bytes')
        require(array_record(checkpoint['K'])==frozen['K_tensor']==array_record(inputs[scene]['K_render_float32']),'map/eval camera intrinsics')
        source_ids=np.asarray(checkpoint['source_frame_ids_per_gaussian'],dtype=np.int64)
        require(array_record(source_ids)==frozen['source_frame_ids_tensor'] and set(source_ids).issubset(TRAIN_IDS)and not set(source_ids)&set(EVAL_IDS),'no future Gaussian source')
        c2w=checkpoint['estimated_c2w'].numpy();w2c=checkpoint['estimated_w2c'].numpy()
        require(np.array_equal(c2w[0],np.eye(4))and np.max(np.abs(c2w@w2c-np.eye(4)))<2e-6,'map first gauge/inverse')
        training=read(checkpoint_path.parent/'metrics.json')
        require(sha(checkpoint_path.parent/'metrics.json')==frozen['metrics_sha256'],'training metrics SHA')
        require(sha(checkpoint_path.parent/'configuration.json')==frozen['configuration_sha256'],'training config SHA')
        training_flags=sum(bool(v['failure_flags'])for v in checkpoint['completed_frame_metrics'])
        require(training_flags==training['failure_frame_count']==frozen['training_failure_frame_count'],'training failures retained')
        require(row['status']=='completed' and row['parameter_bytes_unchanged'] and row['original_checkpoint_bytes_unchanged'],'eval parameter guards passed')
        require([v['frame_id']for v in row['views']]==EVAL_IDS,'all future requested frames')
        measured=[];views=[];predictions[name]=[]
        for i,view in enumerate(row['views']):
            require(view['status']=='completed','completed view status')
            filename=OUT/view['prediction'];require(filename.resolve().is_relative_to(OUT.resolve()),'safe prediction path')
            require(sha(filename)==view['prediction_sha256'],'NPZ file SHA')
            with np.load(filename,allow_pickle=False)as z:
                require(set(z.files)=={'rgb','depth','alpha'},'prediction fields')
                arrays={key:z[key].copy()for key in z.files}
            require(all(a.dtype==np.float32 for a in arrays.values()),'prediction float32 channels')
            require({key:array_record(value)for key,value in arrays.items()}==view['arrays'],'prediction array raw SHA')
            require(array_record(inputs[scene]['evaluation_w2c_render_float32'][i])==view['render_w2c'],'actual render camera SHA')
            value,covered=measure(arrays['rgb'],arrays['depth'],arrays['alpha'],inputs[scene]['rgb'][i],inputs[scene]['depth'][i],inputs[scene]['valid_depth_mask'][i])
            compare(view['metrics'],value,name+'/'+str(view['frame_id']))
            measured.append(value);predictions[name].append((arrays,covered));view_count+=1
            views.append({'frame_id':view['frame_id'],'metrics_recomputed':value,'NPZ_sha256':view['prediction_sha256'],
                          'rgb_prediction_mean':float(arrays['rgb'].astype(np.float64).mean()),
                          'rgb_target_mean':float(inputs[scene]['rgb'][i].astype(np.float64).mean())})
        combined=aggregate(row['views'],measured);compare(row['aggregate'],combined,name+'/aggregate')
        run_audits.append({'run':name,'scene':scene,'seed':row['seed'],'weight':row['prior_weight'],
            'checkpoint_sha256':sha(checkpoint_path),'current_checkpoint_parameter_payloads_equal_frozen_protocol':True,
            'source_frame_histogram':dict(Counter(str(x)for x in source_ids)),'training_flag_count':training_flags,
            'all_training_frame_denominator':100,'tracked_nonanchor_denominator':99,
            'requested_eval_views':10,'completed_eval_views':10,'failed_eval_views':0,'views':views,'aggregate_recomputed':combined})
    pair_audits=[];pair_count=0
    for row in metrics['pairs']:
        scene=row['scene'];a=row['prior0_run'];b=row['prior001_run']
        require(row['requested_views']==row['completed_views']==10 and [v['frame_id']for v in row['views']]==EVAL_IDS,'pair view denominator')
        require(next(r for r in metrics['runs']if r['run']==a)['prior_weight']==0. and
                next(r for r in metrics['runs']if r['run']==b)['prior_weight']==.01 and
                all(next(r for r in metrics['runs']if r['run']==n)['seed']==row['seed'] for n in(a,b)),'same-seed weight pair')
        recomputed=[]
        for i,v in enumerate(row['views']):
            arrays_a,mask_a=predictions[a][i];arrays_b,mask_b=predictions[b][i];common=mask_a&mask_b
            observed=inputs[scene]['depth'][i];valid=inputs[scene]['valid_depth_mask'][i]
            stats_a=depth_statistics(arrays_a['depth'],observed,common);stats_b=depth_statistics(arrays_b['depth'],observed,common)
            n=int(valid.sum());c=int(common.sum())
            rec={'frame_id':EVAL_IDS[i],'status':'completed','sensor_valid_pixels':n,'intersection_pixels':c,
                 'intersection_sensor_valid_coverage':c/n,'prior0_depth':stats_a,'prior001_depth':stats_b,
                 'mae_difference_prior001_minus_prior0_m':stats_b['mae_m']-stats_a['mae_m']if c else None,
                 'rmse_difference_prior001_minus_prior0_m':stats_b['rmse_m']-stats_a['rmse_m']if c else None}
            compare(v,rec,a+'/'+b+'/'+str(EVAL_IDS[i])+'/common');recomputed.append(rec);pair_count+=1
        common_count=sum(v['intersection_pixels']for v in recomputed);sensor_count=sum(v['sensor_valid_pixels']for v in recomputed)
        pool={'intersection_pixels':common_count,'sensor_valid_pixels':sensor_count,
              'prior0_depth':pooled_depth([v['prior0_depth']for v in recomputed]),
              'prior001_depth':pooled_depth([v['prior001_depth']for v in recomputed]),
              'intersection_sensor_valid_coverage':common_count/sensor_count}
        means={'intersection_sensor_valid_coverage':mean([v['intersection_sensor_valid_coverage']for v in recomputed]),
               **{f'{name}_{key}':defined_mean([v[name][key]for v in recomputed])for name in('prior0_depth','prior001_depth')for key in('mae_m','rmse_m')}}
        compare(row['pooled'],pool,a+'/'+b+'/pooled');compare(row['mean_per_view'],means,a+'/'+b+'/mean')
        pair_audits.append({'scene':scene,'seed':row['seed'],'prior0_run':a,'prior001_run':b,
            'all_10_common_pixel_masks_recomputed':True,'views':recomputed,'pooled_recomputed':pool,'mean_per_view_recomputed':means})
    require(view_count==120 and pair_count==60 and not torch.cuda.is_initialized(),'complete CPU-only arithmetic')
    result={'status':'passed','audit_source_sha256':sha(Path(__file__)),'protocol_sha256':PROTOCOL_SHA,
        'metrics_sha256':sha(OUT/'metrics.json'),'progress_sha256':sha(OUT/'progress.json'),
        'source_review':source,'cpu_controls':checks,'datasets':dataset_checks,'map_count':12,'view_count':view_count,
        'same_seed_pairs':6,'common_coverage_views':pair_count,'scalar_checks':len(DIFFERENCES),
        'maximum_absolute_scalar_difference':max(v['absolute_difference']for v in DIFFERENCES),
        'largest_10_numeric_differences':sorted(DIFFERENCES,key=lambda v:v['absolute_difference'],reverse=True)[:10],
        'all_120_NPZ_files_and_360_array_payload_hashes_equal':True,'runs':run_audits,'pairs':pair_audits,
        'CUDA_initialized':False,'limits':['Known-GT-camera map quality, not localization accuracy.',
            'Frames300..309 are not trained into maps200..299, but are same-sequence diagnostics rather than an untouched final test.',
            'Association reuses the frozen loader. Pixel metric formulas, RGB/depth decoding and quaternion/rigid gauge controls are recomputed here without evaluator metric functions.',
            'Before/after GPU tensor guards are source-reviewed and current checkpoint bytes match the reviewed protocol; this CPU audit does not rerender or reconstruct GPU execution.',
            'Covered depth MAE/RMSE are conditional on strict alpha>0.5, finite positive ED and sensor validity. Holes-as-zero and common coverage expose different denominators.',
            'Nearest GT samples, sensor time offsets, registered RGB-D/Freiburg1 intrinsics and no lens-distortion/motion compensation remain assumptions.',
            'Low color PSNR and remaining depth errors are retained across all twelve maps; no best-seed/map/threshold selection.']}
    with (OUT/'independent_map_quality_audit.json').open('x',encoding='utf-8')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'maps':12,'views':120,'pairs':6,'common_views':60,
        'max_absolute_scalar_difference':result['maximum_absolute_scalar_difference'],
        'pooled':[{k:r[k]for k in('run','scene','seed','weight','aggregate_recomputed')}for r in run_audits],
        'CUDA_initialized':False},indent=2))


if __name__=='__main__':main()
