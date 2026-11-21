"""Compare every retained observation in the same image domain; keep detector targets explicit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
run=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
out=base/'alignment-control-summary';out.mkdir(exist_ok=True)
weights=np.load(run/'fit/observation-weights.npz')['face'];valid=weights>0
rtmw=np.load(run/'data/observations.npz')['keypoints'][:,40:91]
fan=np.load(base/'face-local-detector/observations.npz')['decoded_face68'][:,17:68]
models={'static':np.load(run/'fit/fitted.npz')['projected_face51'],
        'dynamic_rtmw':np.load(base/'dynamic-face-control/fitted.npz')['projected_face51'],
        'dynamic_fan':np.load(base/'dynamic-face-fan-control/fitted.npz')['projected_face51']}
stat=lambda a:dict(count=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))
records={};per_frame=[]
for name,points in models.items():
    records[name]={}
    for detector,targets in [('rtmw',rtmw),('fan',fan)]:
        error=np.linalg.norm(points-targets,axis=-1)
        records[name][detector]=dict(overall=stat(error[valid]),per_frame=[stat(e[v]) for e,v in zip(error,valid)])
for i in range(len(rtmw)):
    per_frame.append(dict(frame=f'frame_{i+1:06d}',retained_points=int(valid[i].sum()),
       baseline_to_rtmw=records['static']['rtmw']['per_frame'][i]['median'],
       baseline_to_fan=records['static']['fan']['per_frame'][i]['median'],
       dynamic_rtmw_to_rtmw=records['dynamic_rtmw']['rtmw']['per_frame'][i]['median'],
       dynamic_fan_to_fan=records['dynamic_fan']['fan']['per_frame'][i]['median']))
read=lambda p:json.loads(Path(p).read_text())
skin={name:read(base/f'dynamic-face{suffix}-control/skin-validation.json') for name,suffix in [('dynamic_rtmw',''),('dynamic_fan','-fan')]}
baseline=skin['dynamic_rtmw']['before']
assert skin['dynamic_fan']['before']==baseline
report=dict(frames=len(rtmw),retained_observations=int(valid.sum()),image_domain_wh=[899,1600],
    face_regions=dict(brows=int(valid[:,:10].sum()),nose=int(valid[:,10:19].sum()),eyes=int(valid[:,19:31].sum()),mouth=int(valid[:,31:].sum())),
    same_face_weights=True,cameras_changed=False,frame_selection_changed=False,
    cross_detector_keypoint_errors=records,per_frame=per_frame,
    independent_skin_track=dict(baseline=baseline,**{k:v['held'] for k,v in skin.items()},native_wh=[1520,2704],
        anchors=6,held=35,frames=[40,80],used_for_head_or_expression_fit=False,
        limitation='One local skin trajectory only; original images are used by subsequent Gaussian training.'),
    quality_accepted=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(2,1,figsize=(12,8))
x=np.arange(1,len(rtmw)+1)
for detector,ax in zip(['rtmw','fan'],axes):
    for model,label in [('static','Original static'),('dynamic_rtmw','Dynamic face / body detector'),('dynamic_fan','Dynamic face / face-crop detector')]:
        ax.plot(x,[r['median'] for r in records[model][detector]['per_frame']],label=label,lw=1.6)
    ax.set(title=f'All 95 frames against the same {detector.upper()} targets',ylabel='Median error (decoded pixels)',xlabel='Frame')
    ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('Fixed background cameras | unchanged face validity mask | no frame removal',fontsize=13)
fig.tight_layout();fig.savefig(out/'all-frame-alignment.png',dpi=160);plt.close(fig)
print(json.dumps(dict(frames=report['frames'],observations=report['retained_observations'],overall={k:{d:x['overall'] for d,x in v.items()} for k,v in records.items()},skin_held=report['independent_skin_track'])))
