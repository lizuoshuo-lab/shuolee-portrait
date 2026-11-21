"""Hash-frozen CPU protocol and real-validation entrypoint, no implicit execute.

Preparation loads/checks sensors and constructs geometry at identity only;
it never runs register_pair on real data. Execution is a separate explicit CLI.
GT fields/times are read by the frozen association loader; pose matrices enter
only final trajectory evaluation after every estimated pose is already saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

import rgbd_photogeometric_odometry as odom

ROOT=Path(__file__).resolve().parents[1]
DEADLINE=dt.datetime.fromisoformat('2026-10-09T08:00:00+08:00')
TRAIN_IDS=list(range(200,300))
SOURCE_SHA={
    'rgbd_photogeometric_odometry.py':'80160ea921a40363b9143d6c2a11c1954063aa5fe610397e4f8dcc336261b125',
    'check_rgbd_photogeometric_odometry.py':'e2f71002e0b522f9419fd024f06c165101f04ba2c35da9ee6e23faca3cc22037',
    'audit_photogeometric_math.py':'7b28a4f6c76120c5f62052ebda3a1dc5213ffd55e197c0da463662429499ee6f',
    'track_rgbd_icp.py':'21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a',
    'tum_rgbd.py':'471a0aebb80c7268b5388bc85a556bdf99c8fda0355dd3307ca7bd3639b71b11',
    'audit_tum_trajectory.py':'e7acfaea60297dbc31eccc62b6ea9f50f577fb317209ac50e3342630424a42d7',
}
BASELINES={'desk':'tum-icp-transfer100-001','xyz':'tum-xyz-icp-transfer100-001'}
ROUNDS=[{'order':i+1,'round_id':f'{scene}-photo{weight}', 'scene':scene,'lambda_photo':float(weight),
         'run':f'tum-{scene}-photogeometric-photo{weight}-transfer100-001'}
        for i,(scene,weight)in enumerate((('desk',0),('desk',1),('xyz',0),('xyz',1)))]


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    with path.open('x',encoding='utf-8')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def update(path,value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def array_record(value):
    a=np.ascontiguousarray(value)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}


def source_checks():
    for name,digest in SOURCE_SHA.items():require(sha(ROOT/'scripts'/name)==digest,'frozen source changed '+name)
    return {**SOURCE_SHA,'track_rgbd_photogeometric.py':sha(Path(__file__)),
            'run_rgbd_photogeometric_validation.py':sha(ROOT/'scripts/run_rgbd_photogeometric_validation.py')}


def before_deadline(now=None):
    stamp=now or dt.datetime.now(dt.timezone.utc)
    require(stamp.tzinfo is not None and stamp<DEADLINE,'No new CPU validation round starts at/after 08:00 Beijing')
    return stamp.isoformat()


def config_for(weight):
    return replace(odom.OdometryConfig(),lambda_photo=float(weight))


def metadata(frame):
    return {'frame_id':frame.frame_id,'rgb_index':frame.rgb_index,'depth_index':frame.depth_index,'gt_index':frame.gt_index,
        'rgb_timestamp':frame.rgb_timestamp,'depth_timestamp':frame.depth_timestamp,'gt_timestamp':frame.gt_timestamp,
        'rgb_depth_abs_dt_s':abs(frame.rgb_timestamp-frame.depth_timestamp),'rgb_gt_abs_dt_s':abs(frame.rgb_timestamp-frame.gt_timestamp),
        'rgb_path':frame.rgb_path,'depth_path':frame.depth_path,
        'gt_quaternion_norm_before_normalization':frame.gt_quaternion_norm_before_normalization}


def dataset_preflight(scene):
    from tum_rgbd import associate_frames,load_frames,scaled_intrinsics,safe_data_path
    dataset=ROOT/f'data/tum/rgbd_dataset_freiburg1_{scene}'
    frames,association=associate_frames(dataset,.02,.02,32.);selected=frames[200:300]
    require([f.frame_id for f in selected]==TRAIN_IDS,'100 associated frames200..299')
    baseline_dir=ROOT/'outputs'/BASELINES[scene]
    baseline=read(baseline_dir/'metrics.json');validation=read(baseline_dir/'validation.json')
    require(association==baseline['association'],'association differs from frozen depth ICP baseline')
    rows=[metadata(f)for f in selected]
    require(rows==validation['selected_sensor_metadata'],'loader metadata differs from original ICP100')
    pyramid=[];array_hashes={};sensor_files=[]
    for level,(width,height)in enumerate(odom.icp.LEVELS):
        rgb,depth,valid=load_frames(dataset,selected,width,height,.2,5.)
        gray=np.stack([odom.grayscale(image)for image in rgb]);K=scaled_intrinsics(width,height)
        pyramid.append({'rgb':rgb,'depth':depth,'valid':valid,'gray':gray,'K':K})
        array_hashes[str(level)]={key:array_record(value)for key,value in pyramid[-1].items()}
    first_start={}
    for level,d in enumerate(pyramid):
        starts=[]
        for weight in(0.,1.):
            config=config_for(weight)
            source=odom.make_frame(d['depth'][1],d['valid'][1],d['K'],d['rgb'][1],config)
            target=odom.make_frame(d['depth'][0],d['valid'][0],d['K'],d['rgb'][0],config)
            pairs=odom.pair_correspondences(np.eye(4),source,target,config.geometry.distance_gates_m[level],config)
            starts.append({'counts':pairs['counts'],'ratio':pairs['ratio'],
                'arrays':{key:array_record(pairs[key])for key in('source','transformed','target','normals','source_intensity')}})
        require(starts[0]==starts[1],'actual photo0/photo1 identity support differs')
        # No normal-equation solve, optimizer step or real-data register_pair.
        first_start[str(level)]={**starts[0],'same_photo0_photo1_at_identity_actual_both_configs_checked':True}
    for frame in selected:
        sensor_files.append({'frame_id':frame.frame_id,**{key:{'path':getattr(frame,key),
            'sha256':sha(safe_data_path(dataset,getattr(frame,key)))}for key in('rgb_path','depth_path')}})
    record={'scene':scene,'dataset':dataset.as_posix(),'association':association,'selected_sensor_metadata':rows,
        'index_sha256':{name:sha(dataset/name)for name in('rgb.txt','depth.txt','groundtruth.txt')},
        'raw_sensor_files':sensor_files,'input_array_hashes':array_hashes,
        'baseline_run':BASELINES[scene],'baseline_metrics_sha256':sha(baseline_dir/'metrics.json'),
        'baseline_validation_sha256':sha(baseline_dir/'validation.json'),'first_pair_identity_start':first_start,
        'real_data_pair_registration_runs':0,'ground_truth_scope':'Frozen loader reads GT matrices/indices/times for association and validation. No GT matrix is passed into geometry, grayscale, identity correspondences or optimizer.'}
    return record,pyramid,selected


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True);modes.add_argument('--prepare',action='store_true');modes.add_argument('--execute',action='store_true')
    p.add_argument('--protocol',type=Path);p.add_argument('--protocol-sha');p.add_argument('--round-id',choices=[r['round_id']for r in ROUNDS]);p.add_argument('--run',required=True)
    return p


def arguments(argv=None):
    p=parser();a=p.parse_args(argv)
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,119}',a.run):p.error('simple unique run name required')
    if a.execute!=(a.protocol is not None)or a.execute!=(a.protocol_sha is not None)or a.execute!=(a.round_id is not None):p.error('execute requires protocol/hash/round-id, prepare accepts none')
    if a.protocol_sha and not re.fullmatch(r'[a-f0-9]{64}',a.protocol_sha):p.error('exact64 lowercase protocol SHA required')
    return a


def prepare(output):
    sources=source_checks();datasets={};bundles={}
    for scene in ('desk','xyz'):
        datasets[scene],data,_=dataset_preflight(scene)
        filename=output/f'{scene}-sensor-pyramid.npz'
        np.savez_compressed(filename,**{f'level{level}_{key}':value for level,row in enumerate(data)for key,value in row.items()})
        bundles[scene]={'path':filename.relative_to(ROOT).as_posix(),'sha256':sha(filename),'contains_GT_poses':False}
    rounds=[]
    for spec in ROUNDS:
        require(not(ROOT/'outputs'/spec['run']).exists(),'predeclared run already exists')
        rounds.append({**spec,'configuration':{'frames':100,'start_frame':200,'frame_ids':TRAIN_IDS,
            'dataset':datasets[spec['scene']]['dataset'],'levels':[list(v)for v in odom.icp.LEVELS],
            'pair_initialization':'identity current-to-previous for each pair','first_c2w':'identity',
            'odometry':asdict(config_for(spec['lambda_photo']))}})
    now=DEADLINE-dt.timedelta(microseconds=1);before_deadline(now)
    rejected=[]
    for now in(DEADLINE,DEADLINE+dt.timedelta(microseconds=1)):
        try:before_deadline(now)
        except ValueError:rejected.append(now.isoformat())
        else:raise AssertionError('deadline equality/after accepted')
    protocol={'status':'ready_for_explicit_root_review_then_CPU_execution','source_sha256':sources,
        'rounds':rounds,'datasets':datasets,'sensor_bundles':bundles,'deadline_beijing':DEADLINE.isoformat(),
        'round_order':['desk-photo0','desk-photo1','xyz-photo0','xyz-photo1'],
        'prepared_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'real_data_registration_runs':0,'GPU_runs':0,
        'synthetic_success_reference':'outputs/rgbd-photogeometric-cpu-controls-003/metrics.json',
        'synthetic_success_sha256':sha(ROOT/'outputs/rgbd-photogeometric-cpu-controls-003/metrics.json'),
        'synthetic_supplement_sha256':sha(ROOT/'outputs/rgbd-photogeometric-cpu-controls-003/supplementary_math_controls.json'),
        'deadline_controls':{'one_us_before_passed':True,'equality_and_after_rejected':rejected},
        'limits':['No real optimizer or pair registration ran during preparation; identity geometry/correspondence checking is read-only.',
            'New photo0 still uses one-cell bilinear-support filtering and geometric normalization; it is not byte-equivalent old ICP.',
            'New photo0/photo1 share sensor arrays and identity support start, but later correspondences/trial paths can differ.',
            'Gray uses .299R+.587G+.114B; no affine exposure, gamma calibration or motion compensation.',
            'Engineering .01m/.1intensity scales, normalized Huber1, photo0/1, support1cell and final intensity gate.1 are fixed without GT weight/frame/scale selection.',
            'GT pose matrices are only for final evaluation after accepted/rejected estimated trajectories are fixed; GT fields/times are honestly read by association loader.',
            'Every rejected/exception frame is retained with previous pose propagation and candidate diagnostics.',
            'CPU odometry may overlap separate GS GPU work; observed durations are not isolated speed/FPS or cross-method performance comparisons.']}
    write(output/'protocol.json',protocol);digest=sha(output/'protocol.json');cli=[]
    for row in rounds:
        argv=['--execute','--protocol',str(output/'protocol.json'),'--protocol-sha',digest,'--round-id',row['round_id'],'--run',row['run']]
        parsed=arguments(argv)
        require(parsed.round_id==row['round_id']and parsed.run==row['run']and
            asdict(config_for(row['lambda_photo']))==row['configuration']['odometry'],'actual four CLI parses and configs')
        cli.append({'round_id':row['round_id'],'parsed_fields':{'execute':parsed.execute,'protocol':str(parsed.protocol),'protocol_sha':parsed.protocol_sha,
                    'round_id':parsed.round_id,'run':parsed.run},'argv':argv,'fixed_configuration':row['configuration']})
    write(output/'cli_checks.json',{'status':'passed','protocol_sha256':digest,'all_four_actual_argument_parses_exact':True,
        'trainer_or_register_pair_invoked':False,'rounds':cli,'GPU_runs':0})
    write(output/'source_manifest.json',{'source_sha256':sources,'protocol_sha256':digest})
    for name in('track_rgbd_photogeometric.py','run_rgbd_photogeometric_validation.py'):
        shutil.copyfile(ROOT/'scripts'/name,output/name)
    return {'status':protocol['status'],'protocol':str(output/'protocol.json'),'protocol_sha256':digest,
            'source_sha256':sources,'real_data_registration_runs':0,'GPU_runs':0}


def pose_text(path,frames,poses):
    with path.open('w',encoding='utf-8')as stream:
        for frame,pose in zip(frames,poses):
            q=Rotation.from_matrix(pose[:3,:3]).as_quat()
            stream.write(f'{frame.rgb_timestamp:.9f} '+ ' '.join(f'{x:.12f}'for x in np.r_[pose[:3,3],q])+'\n')


def execute(args,output):
    require(sha(args.protocol)==args.protocol_sha,'exact reviewed protocol SHA')
    protocol=read(args.protocol);require(protocol['source_sha256']==source_checks(),'protocol fixed sources')
    require(protocol['status']=='ready_for_explicit_root_review_then_CPU_execution','ready protocol')
    rows=[r for r in protocol['rounds']if r['round_id']==args.round_id];require(len(rows)==1,'fixed round uniquely declared')
    row=rows[0];require(row['run']==args.run,'predeclared runname only')
    data_record,data,frames=dataset_preflight(row['scene'])
    require(data_record==protocol['datasets'][row['scene']],'sensor/reference bytes changed after preflight')
    bundle=protocol['sensor_bundles'][row['scene']];require(sha(ROOT/bundle['path'])==bundle['sha256'],'prepared sensor bundle changed')
    config=config_for(row['lambda_photo']);require(asdict(config)==row['configuration']['odometry'],'fixed scales/config')
    start=before_deadline();output.mkdir(exist_ok=False)
    (output/'protocol_used.json').write_bytes(args.protocol.read_bytes())
    write(output/'configuration.json',{**row['configuration'],'round_id':args.round_id,'source_sha256':protocol['source_sha256'],
        'protocol_sha256':args.protocol_sha,'CPU_only':True})
    state={'status':'running','round_id':args.round_id,'configuration':row['configuration'],'source_sha256':protocol['source_sha256'],
        'protocol_sha256':args.protocol_sha,'started_utc':start,'frame_metrics':[],'frames_processed':0,'GT_pose_used_in_optimizer':False,
        'ground_truth_scope':data_record['ground_truth_scope'],'GPU_runs':0}
    poses=[np.eye(4)];targets=[];records=state['frame_metrics'];started=time.perf_counter()
    def geometry(index):
        return [odom.make_frame(d['depth'][index],d['valid'][index],d['K'],d['rgb'][index],config)for d in data]
    try:
        targets=geometry(0)
        records.append({'local_index':0,'frame_id':200,'accepted':True,'failure_flags':[],
            'status':'fixed_first_camera_identity','estimated_c2w':poses[0].tolist(),'pair_initialized_from_GT':False})
        iterations_path=output/'iterations.jsonl'
        with iterations_path.open('x',encoding='utf-8')as iteration_stream:
            for index in range(1,100):
                sources=geometry(index)
                try:result=odom.register_pair(sources,targets,config)
                except (np.linalg.LinAlgError,ValueError,FloatingPointError)as error:
                    result={'accepted':False,'failure_flags':['pair_numeric_exception'],'candidate_previous_from_current':None,
                        'final_counts':{},'final_statistics':{},'iteration_history':[],'levels':[],
                        'exception_type':type(error).__name__,'exception_message':str(error),'traceback':traceback.format_exc(),
                        'candidate_unknown_due_to_exception':True}
                pose=odom.integrate_relative(poses[-1],result);poses.append(pose)
                record={**result,'local_index':index,'frame_id':200+index,'estimated_c2w':pose.tolist(),
                    'pair_initialized_from_GT':False,'source_sensor_metadata':data_record['selected_sensor_metadata'][index]}
                records.append(record)
                for item in result['iteration_history']:
                    iteration_stream.write(json.dumps({'local_index':index,'frame_id':200+index,**item},allow_nan=False)+'\n')
                iteration_stream.flush();targets=sources
                state['frames_processed']=len(records)
                update(output/'progress.json',state)
                # Causal partial estimate, containing no GT values or future data.
                pose_text(output/'trajectory_estimated_partial.txt',frames[:len(poses)],poses)
                print(json.dumps({'round':args.round_id,'frame_id':200+index,'accepted':result['accepted'],'flags':result['failure_flags']}),flush=True)
        pose_text(output/'trajectory_estimated.txt',frames,poses)
        # Only now access pose matrices for evaluation. No candidate is changed.
        absolute_GT=np.stack([frame.c2w for frame in frames])
        anchored_GT=np.linalg.inv(absolute_GT[0])[None]@absolute_GT
        from audit_tum_trajectory import audit_pose_arrays
        trajectory=audit_pose_arrays(np.asarray(poses),anchored_GT)
        pose_text(output/'trajectory_groundtruth_anchored.txt',frames,anchored_GT)
        np.savez_compressed(output/'trajectory_arrays.npz',estimated_c2w=np.asarray(poses),evaluation_only_anchored_GT_c2w=anchored_GT,
            rgb_timestamps=np.asarray([f.rgb_timestamp for f in frames]))
        failures=sum(bool(r['failure_flags'])for r in records)
        state.update(status='completed'if not failures else'completed_with_pair_rejections',frames_processed=100,
            failure_frame_count=failures,tracked_nonanchor_frame_denominator=99,all_frame_denominator=100,
            failure_flag_counts=dict(Counter(f for r in records for f in r['failure_flags'])),
            trajectory=trajectory,selected_sensor_metadata=data_record['selected_sensor_metadata'],
            input_array_hashes=data_record['input_array_hashes'],observed_wall_seconds=time.perf_counter()-started,
            timing_scope='Observed CPU host wall time includes geometry, iteration/partial trajectory/progress I/O and final evaluation/export; not isolated speed/FPS.',
            GT_pose_evaluation_after_all_100_estimates_saved=True,limits=protocol['limits'])
        require(source_checks()==protocol['source_sha256'],'source changed while computing')
        require(sha(ROOT/'outputs'/data_record['baseline_run']/'metrics.json')==data_record['baseline_metrics_sha256'],'old reference changed')
        write(output/'metrics.json',state);update(output/'progress.json',state)
        return {'status':state['status'],'frames_processed':100,'failure_frame_count':failures,'metrics_sha256':sha(output/'metrics.json')}
    except BaseException as error:
        state.update(status='failed_incomplete_original_frames_retained',exception_type=type(error).__name__,exception_message=str(error),
            traceback=traceback.format_exc(),frames_processed=len(records),GT_final_eval_completed=False)
        write(output/'failure.json',state);update(output/'progress.json',state)
        raise


def main():
    args=arguments();output=ROOT/'outputs'/args.run
    require(not output.exists(),'unique output; never overwrite previous run')
    if args.prepare:
        output.mkdir();result=prepare(output)
    else:result=execute(args,output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
