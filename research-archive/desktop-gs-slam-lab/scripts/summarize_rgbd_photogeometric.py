"""Read-only compact public summaries and posthoc trajectory/rejection figures.

Requires all four frozen CPU processes to have terminated successfully before
any aggregate/comparison is written. No odometry, renderer, optimizer or GT
selection is executed. Original large metrics and iteration ledgers stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BATCH=ROOT/'outputs/rgbd-photogeometric-real-validation-001'
PROTOCOL_SHA='81325bceecd336b678e8cadda1af4bf926fd3c7e5788acbb0db128524ccc265b'
KEYS=('raw_anchored_translation_rmse_m','se3_aligned_ate_translation_rmse_m',
      'adjacent_rpe_translation_rmse_m','adjacent_rpe_rotation_rmse_deg')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
    require(len(data)<1024*1024,'compact artifact must stay below 1MiB '+str(path))
    with path.open('xb')as stream:stream.write(data)
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':sha(path)}


def rejection_segments(frames):
    groups=[]
    for frame in frames:
        flags=frame['failure_flags']
        if not flags:continue
        frame_id=frame['frame_id']
        if not groups or frame_id!=groups[-1]['last_frame_id']+1:
            groups.append({'first_frame_id':frame_id,'last_frame_id':frame_id,'frames':[],'flags':[]})
        group=groups[-1];group['last_frame_id']=frame_id
        group['frames'].append({'frame_id':frame_id,'failure_flags':flags})
        group['flags']=sorted(set(group['flags'])|set(flags))
    return groups


def collect(batch):
    path=batch/'metrics.json'
    require(path.exists(),'All four processes must finish before summarization')
    ledger=read(path);require(ledger['status']=='all_four_predeclared_CPU_rounds_completed'and
        ledger['runs_started']==4 and ledger['current_run']is None,'terminal four-round ledger')
    require(ledger['protocol_sha256']==PROTOCOL_SHA,'fixed reviewed protocol')
    protocol_path=ROOT/'outputs/rgbd-photogeometric-real-cpu-preparation-003/protocol.json'
    require(sha(protocol_path)==PROTOCOL_SHA,'actual reviewed protocol bytes')
    protocol=read(protocol_path);methods=[]
    require(len(ledger['run_records'])==4,'all four child records')
    for frozen,record in zip(protocol['rounds'],ledger['run_records']):
        require(record['round_id']==frozen['round_id']and record['run']==frozen['run']and
            record['order']==frozen['order']and record['configuration']==frozen['configuration']and record['returncode']==0,'fixed order/config/process exits')
        folder=ROOT/'outputs'/frozen['run'];metric_path=folder/'metrics.json';original=read(metric_path)
        require(sha(metric_path)==record['metrics_sha256']and sha(batch/(frozen['round_id']+'.log'))==record['log_sha256'],'original metric/log bytes')
        require((folder/'protocol_used.json').read_bytes()==protocol_path.read_bytes(),'run exact protocol copy')
        require(original['source_sha256']==protocol['source_sha256']and original['configuration']==frozen['configuration'],'source/config unchanged')
        require(original['frames_processed']==100 and [r['frame_id']for r in original['frame_metrics']]==list(range(200,300)),'all actual frames retained')
        require(original['selected_sensor_metadata']==protocol['datasets'][frozen['scene']]['selected_sensor_metadata']and
            original['input_array_hashes']==protocol['datasets'][frozen['scene']]['input_array_hashes'],'all metadata/input hashes')
        flags=sum(bool(row['failure_flags'])for row in original['frame_metrics'])
        require(flags==original['failure_frame_count']==record['failure_frame_count'],'failure counts unchanged')
        # All per-frame fields except large iteration history and repeated fixed
        # configuration remain exact. Every final count/flag/level/metadata,
        # candidate (including rejected) and accepted prefix remains present.
        frames=[]
        for row in original['frame_metrics']:
            frame={key:value for key,value in row.items()if key not in('iteration_history','configuration')}
            if 'iteration_history'in row:
                frame['iteration_history_summary']={'records':len(row['iteration_history']),
                    'accepted_updates':sum(bool(i['accepted'])for i in row['iteration_history']),
                    'stop_reasons':dict(Counter(i.get('stop_reason','continued')for i in row['iteration_history']))}
            frames.append(frame)
        summary={key:value for key,value in original.items()if key not in('frame_metrics','observed_wall_seconds','timing_scope')}
        summary['frame_metrics']=frames
        summary['public_summary_provenance']={'original_metrics_relative_path':metric_path.relative_to(ROOT).as_posix(),
            'original_metrics_sha256':sha(metric_path),'original_metrics_bytes':metric_path.stat().st_size,
            'summarizer_sha256':sha(Path(__file__)),'full_100_flags_final_counts_levels_metadata_candidates_prefix_retained':True,
            'local_iterations_jsonl_sha256':sha(folder/'iterations.jsonl'),
            'iteration_scope':'Full per-iteration correspondences, line-search costs and matrices remain in original metrics/iterations.jsonl; compact summary keeps complete per-frame diagnostics and per-level stops.',
            'timing_scope':'CPU/GPU work can overlap. Durations are not used for cross-method speed comparisons.'}
        methods.append({'name':frozen['round_id'],'run':frozen['run'],'scene':frozen['scene'],
            'kind':'new_photo0_with_support_filter'if frozen['lambda_photo']==0 else'new_photo1_plus_geometry',
            'summary':summary,'folder':folder,'poses':np.asarray([r['estimated_c2w']for r in original['frame_metrics']]),
            'failure_frame_count':flags,'rejection_segments':rejection_segments(original['frame_metrics']),
            'trajectory':original['trajectory'],'frames':original['frame_metrics']})
    for scene in('desk','xyz'):
        witness=protocol['datasets'][scene];folder=ROOT/'outputs'/witness['baseline_run'];path=folder/'metrics.json';baseline=read(path)
        require(sha(path)==witness['baseline_metrics_sha256']and sha(folder/'validation.json')==witness['baseline_validation_sha256'],'immutable old ICP reference')
        require(read(folder/'validation.json')['selected_sensor_metadata']==witness['selected_sensor_metadata'],'same reference observations')
        require(baseline['frames_processed']==100 and [r['frame_id']for r in baseline['frame_metrics']]==list(range(200,300)),'baseline frames retained')
        methods.append({'name':scene+'-old-ICP','run':witness['baseline_run'],'scene':scene,'kind':'old_geometric_ICP_reference',
            'summary':{'run':witness['baseline_run'],'status':baseline['status'],'configuration':baseline['configuration'],
                'frames_processed':100,'failure_frame_count':baseline['failure_frame_count'],'frame_metrics':baseline['frame_metrics'],
                'selected_sensor_metadata':witness['selected_sensor_metadata'],'trajectory':baseline['trajectory'],
                'public_summary_provenance':{'original_metrics_sha256':sha(path),'original_metrics_bytes':path.stat().st_size,
                    'original_metrics_relative_path':path.relative_to(ROOT).as_posix(),'summarizer_sha256':sha(Path(__file__)),
                    'scope':'Read-only original ICP100 reference; new photo0 is not declared byte-equivalent to this method.'}},
            'folder':folder,'poses':np.asarray([r['estimated_c2w']for r in baseline['frame_metrics']]),
            'failure_frame_count':baseline['failure_frame_count'],'rejection_segments':rejection_segments(baseline['frame_metrics']),
            'trajectory':baseline['trajectory'],'frames':baseline['frame_metrics']})
    return protocol,ledger,methods


def plot_all(methods,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    fig=plt.figure(figsize=(15,9),constrained_layout=True)
    for row,scene in enumerate(('desk','xyz')):
        selected=[next(m for m in methods if m['name']==name)for name in(scene+'-old-ICP',scene+'-photo0',scene+'-photo1')]
        truth=np.loadtxt(selected[0]['folder']/'trajectory_groundtruth_anchored.txt')[:,1:4]
        # Same fixed first-camera gauge. No SE3/Sim3 alignment is fitted to plot.
        points=np.concatenate([truth]+[m['poses'][:,:3,3]for m in selected])
        low=points.min(axis=0);high=points.max(axis=0);margin=np.maximum((high-low)*.05,.002)
        for col,method in enumerate(selected):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');estimated=method['poses'][:,:3,3]
            ax.plot(*truth.T,color='#444444',label='GT, first-camera anchored')
            ax.plot(*estimated.T,color='#248e96',label='estimated prefix')
            rejected=np.array([bool(frame['failure_flags'])for frame in method['frames']])
            if rejected.any():ax.scatter(*estimated[rejected].T,color='#d04438',s=25,label=f'rejected ({rejected.sum()}/99)')
            ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=method['name']+f' | {method["failure_frame_count"]}/99 rejected')
            ax.set_xlim(low[0]-margin[0],high[0]+margin[0]);ax.set_ylim(low[1]-margin[1],high[1]+margin[1]);ax.set_zlim(low[2]-margin[2],high[2]+margin[2]);ax.view_init(elev=22,azim=-60)
            ax.legend(fontsize=7)
    fig.suptitle('All fixed CPU methods: unaligned first-camera trajectories\nRejected poses remain in the prefix; fixed axis ranges within each sequence')
    fig.savefig(output/'trajectories-and-rejections.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    labels=('old-ICP','photo0','photo1');colors=('#888888','#c97836','#248e96')
    for row,scene in enumerate(('desk','xyz')):
        for band,(token,color)in enumerate(zip(labels,colors)):
            m=next(m for m in methods if m['name']==scene+'-'+token)
            errors=np.asarray(m['trajectory']['per_frame_raw_translation_error_m'])*1000
            axes[row,0].plot(np.arange(200,300),errors,color=color,label=token)
            axes[row,1].broken_barh([(200.5,99)],(band-.25,.5),facecolors='#ededed')
            for segment in m['rejection_segments']:
                axes[row,1].broken_barh([(segment['first_frame_id']-.5,segment['last_frame_id']-segment['first_frame_id']+1)],(band-.25,.5),facecolors='#d04438')
        axes[row,0].set(title=scene+': raw anchored 3D translation error',xlabel='associated frame ID',ylabel='mm');axes[row,0].grid(alpha=.2);axes[row,0].legend()
        axes[row,1].set(title=scene+': all tracked frames and rejected segments',xlabel='associated frame ID',yticks=[0,1,2],yticklabels=labels,xlim=(200.5,299.5))
    fig.suptitle('Posthoc position errors and complete failure ledger\nGray = requested frames 201..299; red = rejected, retained with previous-pose propagation')
    fig.savefig(output/'position-errors-and-rejected-segments.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--run',default='rgbd-photogeometric-comparison-001');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    require(args.run and all(c in'abcdefghijklmnopqrstuvwxyz0123456789-'for c in args.run),'simple unique output name')
    protocol,ledger,methods=collect(args.batch)
    if args.check_only:
        print(json.dumps({'status':'all_four_terminal_input_checks_passed','statistics_not_written':True,'methods':len(methods)}));return
    output=ROOT/'outputs'/args.run;output.mkdir(exist_ok=False);artifacts=[]
    # Finish building/checking every input before writing comparison artifacts.
    for method in methods:
        # Old-reference summaries live in this new folder; old files stay intact.
        target=output/(method['name']+'-summary.json')
        artifacts.append(save(target,method['summary']))
    table=[]
    for method in methods:
        table.append({'name':method['name'],'run':method['run'],'scene':method['scene'],'kind':method['kind'],
            'frames':100,'tracked_nonanchor_denominator':99,'failure_frame_count':method['failure_frame_count'],
            'failure_fraction':method['failure_frame_count']/99.,'rejected_segments':method['rejection_segments'],
            'trajectory_metrics':{key:method['trajectory'][key]for key in KEYS},
            'original_metrics_sha256':sha(method['folder']/'metrics.json'),
            'original_trajectory_text_sha256':sha(method['folder']/'trajectory_estimated.txt')})
    comparison={'status':'all_four_predeclared_CPU_rounds_terminal','protocol_sha256':PROTOCOL_SHA,
        'batch_metrics_sha256':sha(args.batch/'metrics.json'),'summarizer_sha256':sha(Path(__file__)),
        'new_rounds':4,'original_ICP_references':2,'all_frames_per_method':100,'all_failure_frames_retained':True,
        'methods':table,'public_compact_summaries':artifacts,'process_returncodes':[r['returncode']for r in ledger['run_records']],
        'limits':protocol['limits']+['Trajectory metric formula reuses audit_pose_arrays also used by old ICP; not a second independent evaluation formula.',
            'Plots and metrics are posthoc: no estimate, scale, weight, gate or frame was selected using GT.',
            'No means are computed over unfinished rounds. All four CPU methods and both immutable ICP references are shown.']}
    save(output/'metrics.json',comparison);plot_all(methods,output)
    print(json.dumps({'status':comparison['status'],'output':str(output),'summary_sizes':[a['bytes']for a in artifacts],'methods':table},indent=2))


if __name__=='__main__':main()
