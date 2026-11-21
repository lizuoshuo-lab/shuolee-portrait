"""Compare fine graph with its existing matched coarse counterpart, without selecting new targets."""
from pathlib import Path
import json,hashlib
import numpy as np
L=Path(__file__).resolve().parents[2];A=L/'outputs/dynamic-human-enerf-hand-joint-084';B=L/'outputs/dynamic-human-enerf-hand-fine-085'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
ar=read(A/'comparison/records.json')['records'];br=read(B/'comparison/records.json')['records'];assert len(ar)==len(br)==120
old={(r['camera'],r['frame']):r for r in ar};new={(r['camera'],r['frame']):r for r in br};assert old.keys()==new.keys()
ad={(r['camera'],r['frame'],r['name']):r for r in read(A/'comparison/detail-metrics.json')['records']};bd={(r['camera'],r['frame'],r['name']):r for r in read(B/'comparison/detail-metrics.json')['records']};assert ad.keys()==bd.keys()
result={}
for coarse,fine in [('joint_photo','fine_photo'),('joint_flow','fine_flow')]:
    x,y=[dict(np.load(root/branch/'schedule.npz')) for root,branch in [(A,coarse),(B,fine)]]
    assert x.keys()==y.keys() and all(np.array_equal(x[k],y[k]) for k in x)
    for k in old:assert old[k]['metrics']['parent']==new[k]['metrics']['parent'] and old[k]['split']==new[k]['split']
    rows=[dict(camera=k[0],frame=k[1],split=old[k]['split'],foreground_psnr_change_db=new[k]['metrics'][fine]['foreground']['psnr_db']-old[k]['metrics'][coarse]['foreground']['psnr_db']) for k in old]
    details=[dict(camera=k[0],frame=k[1],name=k[2],psnr_change_db=bd[k]['metrics'][fine]['psnr_db']-ad[k]['metrics'][coarse]['psnr_db']) for k in ad]
    splits={}
    for s in ['train','validation_time','validation_camera']:
        r=[x for x in rows if x['split']==s];vals=np.array([x['foreground_psnr_change_db'] for x in r]);splits[s]=dict(count=len(r),mean_change_db=float(vals.mean()),improved=int((vals>1e-6).sum()),regressed=int((vals< -1e-6).sum()),worst=sorted(r,key=lambda x:x['foreground_psnr_change_db'])[:3])
    result[fine]=dict(coarse=coarse,same_sampling_and_parent=True,splits=splits,details={name:dict(count=sum(r['name']==name for r in details),mean_change_db=float(np.mean([r['psnr_change_db'] for r in details if r['name']==name]))) for name in sorted({r['name'] for r in details})},records=rows,detail_records=details)
save(B/'coarse-versus-fine.json',dict(status='same120contexts30detailwindows_andidenticaltrainingdraws_verified',branches=result,qualification='Adds32advectedlocalcontrols and correspondingregularization to084control;notpureparametercountisolation. Same scene development.',goal_complete=False))
print(json.dumps({b:{k:v for k,v in r.items() if k not in ['records','detail_records']} for b,r in result.items()}))

