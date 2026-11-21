"""Read-only independent metric reconstruction of completed full ICP controls.

Rebuild GT rotations from raw TUM xyzw values and the stored association rows;
do not import the solver, loader or its trajectory metric functions. All frames
and failures are retained. No training or GPU work is performed.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/full-icp-two-sequence-summary-001'
RUNS = [('desk', 'tum-icp-full549-001', 549), ('xyz', 'tum-xyz-icp-full748-001', 748)]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def inverse(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result

def raw_pose(row):
    x,y,z,w = row[4:8] / np.linalg.norm(row[4:8])
    p = np.eye(4)
    p[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    p[:3, 3] = row[1:4]
    return p

def rms(values):
    return math.sqrt(math.fsum(float(v)*float(v) for v in values)/len(values))

def measure(estimated, truth):
    raw = np.linalg.norm(estimated[:, :3, 3]-truth[:, :3, 3], axis=1)
    a,b = estimated[:, :3, 3],truth[:, :3, 3]
    am,bm = a.mean(axis=0),b.mean(axis=0)
    u,s,vt = np.linalg.svd((a-am).T@(b-bm))
    correction = np.diag([1.,1.,np.linalg.det(vt.T@u.T)])
    rotation = vt.T@correction@u.T
    aligned = (a-am)@rotation.T+bm
    ate = np.linalg.norm(aligned-b,axis=1)
    te,re = [],[]
    for i in range(len(estimated)-1):
        dg = inverse(truth[i])@truth[i+1]
        de = inverse(estimated[i])@estimated[i+1]
        residual = inverse(dg)@de
        te.append(np.linalg.norm(residual[:3,3]))
        re.append(np.degrees(np.arccos(np.clip((np.trace(residual[:3,:3])-1)/2,-1,1))))
    values = dict(raw_anchored_translation_rmse_m=rms(raw),
                  se3_aligned_ate_translation_rmse_m=rms(ate),
                  adjacent_rpe_translation_rmse_m=rms(te),
                  adjacent_rpe_rotation_rmse_deg=rms(re))
    return values,raw,np.asarray(te),np.asarray(re)

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    results,plot_data = [],[]
    max_diffs = {'translation_m':0.,'rotation_degrees':0.,'prefix_matrix':0.,'GT_text_matrix':0.}
    for scene,run,count in RUNS:
        folder = ROOT/'outputs'/run
        metric = read(folder/'metrics.json')
        summary = read(folder/'summary.json')
        validation = read(folder/'validation.json')
        metadata = read(folder/'selected_sensor_metadata.json')['selected_frames']
        assert metric['status']=='completed' and summary['status']=='completed' and validation['status']=='passed'
        for name,expected in validation['evidence_sha256'].items():
            assert digest(folder/name)==expected, name
        rows = metric['frame_metrics']
        assert len(rows)==len(metadata)==count and [r['frame_id'] for r in rows]==list(range(count))
        estimated = np.asarray([r['estimated_c2w'] for r in rows],dtype=np.float64)
        rawgt = np.loadtxt(ROOT/f'data/tum/rgbd_dataset_freiburg1_{scene}/groundtruth.txt')
        chosen = rawgt[[r['gt_index'] for r in metadata]]
        assert np.array_equal(chosen[:,0],np.asarray([r['gt_timestamp'] for r in metadata]))
        absolute = np.asarray([raw_pose(row) for row in chosen])
        truth = np.asarray([inverse(absolute[0])@p for p in absolute])
        assert np.array_equal(estimated[0],np.eye(4)) and np.isfinite(estimated).all()
        for poses in (estimated,truth):
            assert np.max(np.abs(poses[:,3,:]-np.array([0.,0.,0.,1.])))<1e-12
            assert max(np.max(np.abs(p[:3,:3].T@p[:3,:3]-np.eye(3))) for p in poses)<1e-10
        for i,row in enumerate(rows[1:],1):
            expected = estimated[i-1]@np.asarray(row['candidate_previous_from_current']) if row['accepted'] else estimated[i-1]
            delta = float(np.max(np.abs(expected-estimated[i])))
            assert delta<1e-10
            max_diffs['prefix_matrix'] = max(max_diffs['prefix_matrix'],delta)
        textgt = np.loadtxt(folder/'trajectory_groundtruth_anchored.txt')
        textposes = np.asarray([raw_pose(row) for row in textgt])
        textdelta = float(np.max(np.abs(textposes-truth)))
        assert textdelta<2e-9
        max_diffs['GT_text_matrix']=max(max_diffs['GT_text_matrix'],textdelta)
        values,raw,rt,rr = measure(estimated,truth)
        comparisons={}
        for key,value in values.items():
            delta=abs(value-metric['trajectory'][key])
            unit='rotation_degrees' if key.endswith('_deg') else 'translation_m'
            assert delta < (2e-6 if unit=='rotation_degrees' else 2e-8),(key,delta)
            max_diffs[unit]=max(max_diffs[unit],delta)
            comparisons[key]={'independently_reconstructed':value,'original':metric['trajectory'][key],'absolute_difference':delta}
        failure_count=sum(bool(r['failure_flags']) for r in rows)
        assert failure_count==metric['failure_frame_count']==summary['failure_frame_count']
        result={'scene':scene,'run':run,'frames':count,'pairs':count-1,'frame_ids_inclusive':[0,count-1],
                'sensor_span_seconds':metadata[-1]['rgb_timestamp']-metadata[0]['rgb_timestamp'],
                'metrics':values,'metric_comparisons':comparisons,'failure_frames':failure_count,
                'iteration_budget_levels':summary['level_stop_reason_counts']['iteration_budget'],
                'total_pyramid_levels':3*(count-1),'raw_final_translation_error_m':float(raw[-1]),
                'raw_max_translation_error_m':float(raw.max()),
                'input_hashes':{name:digest(folder/name) for name in ('metrics.json','summary.json','validation.json','selected_sensor_metadata.json')},
                'raw_groundtruth_sha256':digest(ROOT/f'data/tum/rgbd_dataset_freiburg1_{scene}/groundtruth.txt')}
        results.append(result)
        plot_data.append((scene,np.asarray([r['rgb_timestamp'] for r in metadata])-metadata[0]['rgb_timestamp'],estimated,truth,raw,rt,rr))
    assert 'torch' not in sys.modules and 'gsplat' not in sys.modules
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for axrow,data in zip(axes,plot_data):
        scene,t,e,g,raw,rt,rr=data
        axrow[0].plot(g[:,0,3],g[:,2,3],color='black',label='Sensor-time associated GT')
        axrow[0].plot(e[:,0,3],e[:,2,3],color='#2563eb',label='Depth ICP estimate')
        axrow[0].set_aspect('equal',adjustable='datalim');axrow[0].set_xlabel('X (m)');axrow[0].set_ylabel('Z (m)')
        axrow[0].set_title(f'{scene}: anchored X-Z projection');axrow[0].legend(fontsize=8)
        axrow[1].plot(t,raw*1000,color='#2563eb');axrow[1].set_title(f'{scene}: raw center error');axrow[1].set_ylabel('mm')
        axrow[2].plot(t[1:],rt*1000,color='#d97706');axrow[2].set_title(f'{scene}: adjacent translation RPE');axrow[2].set_ylabel('mm')
        for ax in axrow[1:]:ax.set_xlabel('Elapsed sensor time (s)')
        for ax in axrow:ax.grid(alpha=.2)
    fig.suptitle('Full associated depth-only ICP sequences; separate scenes, no map or loop closure\nAll frames retained; zero heuristic rejection does not imply zero drift. Independent axes per sequence.')
    fig.savefig(OUTPUT/'comparison.png',dpi=160);plt.close(fig)
    record={'status':'passed_read_only_independent_reconstruction','source_script_sha256':digest(Path(__file__)),
            'solver_loader_metric_functions_imported':False,'torch_imported':False,'gsplat_imported':False,
            'max_absolute_differences_by_unit':max_diffs,'all_original_evidence_sha256_verified':True,
            'raw_GT_xyzw_normalized_then_anchored':True,'GT_only_read_for_completed_result_evaluation':True,
            'alignment':'One independent Kabsch SE3 camera-center alignment per sequence; scale=1',
            'all_frames_and_failure_flags_retained':True,'results':results,
            'limitations':['Only two same-sensor-domain sequences; cannot compare scene difficulty as method superiority',
                           'Independent artifact/metric reconstruction does not prove optimizer correctness',
                           'Zero flags is not full convergence; budget-exhausted levels explicitly counted',
                           'Plot uses X-Z projection and separate axes; errors use full 3D poses'],
            'plot_sha256':digest(OUTPUT/'comparison.png')}
    (OUTPUT/'summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'max_differences':max_diffs,'results':[{k:r[k] for k in ('scene','frames','metrics','failure_frames','iteration_budget_levels')} for r in results]},indent=2))

if __name__=='__main__':
    main()
