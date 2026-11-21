from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import cv2
A,B,experiment=map(Path,sys.argv[1:])
out=B/'quality-comparison'
read=lambda path: json.loads(path.read_bytes())
a,b=(read(run/'joint/metrics.json') for run in (A,B))
sa,sb=(read(run/'run.json') for run in (A,B))
assert sa['status']==sb['status']=='complete' and a['resumed_step']==b['resumed_step']==0
tag=b['source_signature']['mesh_offset'].pop('late_prune')
assert tag==[5000,8000,.005,1.] and a['source_signature']==b['source_signature']
assert a['holdout_frames']==b['holdout_frames'] and a['training_frames']==b['training_frames']
assert (A/'joint/cameras.json').read_bytes()==(B/'joint/cameras.json').read_bytes()
assert len(b['head_events'])==1
event=b['head_events'][0]
with np.load(B/'joint/opacity-5000.npz') as f, np.load(B/'joint/opacity-8000.npz') as g:
 assert np.array_equal(f['binding'],g['binding']) and np.array_equal(g['binding'],np.arange(event['before']))
 assert (((f['alpha']<.005)&(g['alpha']<.005)).sum()==event['candidates'])
assert event['after']==event['before']-event['candidates']*int(event['accepted'])
assert event['after']==b['scene_and_head_point_counts'][1] and set(event['per_training_view_max_rgb_delta_255'])==set(b['training_frames'])
metrics=['head_rgb_mae_255','face_mae_255','scalp_mae_255','seam_mae_255','body_mae_255','room_mae_255','face_uncovered_fraction']
def vals(m,n):
 return {k:m['frames'][n]['trained'][k] for k in metrics}
per={n:{'control':vals(a,n),'trial':vals(b,n)} for n in b['holdout_frames']}
avg={label:{k:round(float(np.mean([r[label][k] for r in per.values() if r[label][k] is not None])),5)
                for k in metrics if any(r[label][k] is not None for r in per.values())} for label in ('control','trial')}
def timing(s,m):
 h={p['step']:p for p in m['loss_history']}
 return dict(train_loop_s=h[10000]['elapsed'],steps_8000_to_10000_s=h[10000]['elapsed']-h[8000]['elapsed'],gaussian_stage_s=s['stages']['gaussian_training']['elapsed_s'],
             scene_and_head=m['scene_and_head_point_counts'],mesh_max_world=s['measured_max_vertex_shift_world'],
             scene_at_5000=h[5000]['scene_points'],scene_at_8000=h[8000]['scene_points'],loss_at_5000=h[5000]['loss'])
timings={l:timing(s,m) for l,s,m in [('control',sa,a),('trial',sb,b)]}
with (experiment/'gpu-samples.csv').open() as f: samples=list(csv.DictReader(f))
for k,l in [('A','control'),('B','trial')]:
 vals_mb=[int(row['gpu_memory_mib']) for row in samples if row['run']==k and 8100<=int(row['step_floor'])<=9900]
 if vals_mb: timings[l]['sampled_last_phase_peak_mib']=max(vals_mb)
def crop(run,n):
 im=cv2.imread(str(run/'joint'/(Path(n).stem+'-heldout.jpg')))
 w=im.shape[1]//3
 fw,fh=b['image_size']
 full=30+round(fh*w/fw)
 rows=(im.shape[0]-full)//2
 return im[full+30:full+rows,:w],im[full+30:full+rows,2*w:]
tiles=[]
for n in b['holdout_frames']:
 source,before=crop(A,n)
 check,after=crop(B,n)
 assert source.shape==check.shape and np.abs(source.astype(float)-check.astype(float)).mean()<.4
 row=np.hstack([source,before,after])
 row=cv2.copyMakeBorder(row,32,0,0,0,cv2.BORDER_CONSTANT)
 for idx,title in enumerate(['Source '+Path(n).stem,'Full head','Late-prune trial']):
  cv2.putText(row,title,(6+source.shape[1]*idx,23),cv2.FONT_HERSHEY_SIMPLEX,.60,(242,242,242),1,cv2.LINE_AA)
 assert cv2.imwrite(str(out/('own-'+Path(n).stem+'.png')),row)
 tiles.append(row)
assert cv2.imwrite(str(out/'own-pose-holdouts.png'),np.vstack(tiles))
def sameprobe(path):
 d=read(path)
 return d['holdout_frame_mean']
report=dict(experiment=str(experiment),control=str(A),trial=str(B),reference=b['reference'],seed=b['seed'],training=len(b['training_frames']),holdout=b['holdout_frames'],
            source_signature=a['source_signature'],threshold_rule=tag,prune=event,timing=timings,own_pose_holdouts=per,own_pose_mean=avg,
            fixed_pose_holdout_mean=sameprobe(out/'fixed-detail.json'),camera_sha256=hashlib.sha256((B/'joint/cameras.json').read_bytes()).hexdigest())
(out/'trial-result.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('VERIFIED',event['candidates'],event['after'],avg,timings)

