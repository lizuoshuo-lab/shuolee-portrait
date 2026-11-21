"""Frozen native-pixel photo / original / registered-projection comparison."""
from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/dynamic-human-4k4d-projection-control-040'
A=ROOT/'original_projection';B=ROOT/'calibrated_projection';OUT=ROOT/'paired-comparison';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(ROOT/'independent-state-audit.json')['matched_recorded_sampling']
for p in [A,B]:assert read(p/'native-evaluation/independent-image-audit.json')['full_pngs']==24
rows=[]
for a,b in zip(read(A/'native-evaluation/metrics.json')['records'],read(B/'native-evaluation/metrics.json')['records']):
    for k in ['camera','split','source_cameras','source_rgb_sha256','source_matte_sha256','bbox']:assert a[k]==b[k],k
    c=a['camera'];prefix=f'camera{c}-frame000750'
    arr=lambda p,s:np.array(Image.open(p/'native-evaluation'/f'{prefix}-{s}.png'))
    photo=arr(A,'source');assert np.array_equal(photo,arr(B,'source'));assert np.array_equal(arr(A,'reference'),arr(B,'reference'))
    old=arr(A,'trained');new=arr(B,'trained');x0,y0,x1,y1=a['bbox'];w=x1-x0;h=y1-y0
    panel=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(panel)
    for i,(title,im) in enumerate([('Original photograph',photo),('Original projection 2000',old),('Corrected projection 2000',new)]):
        panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*w,32));d.text((i*w+5,10),title,fill='black')
    p=OUT/f'{prefix}-full-person.png';panel.save(p);actual=np.array(Image.open(p))
    for i,im in enumerate([photo,old,new]):assert np.array_equal(actual[32:,i*w:(i+1)*w],im[y0:y1,x0:x1])
    rows.append(dict(camera=c,split=a['split'],bbox=a['bbox'],source_cameras=a['source_cameras'],before=a['metrics'],after=b['metrics'],
        psnr_delta_db={key:b['metrics'][key]['psnr_db']-a['metrics'][key]['psnr_db'] for key in a['metrics']},panel=p.name,panel_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
report=dict(status='six_matched_native_panels_verified_visual_review_pending',records=rows,scope='Fixed native photograph and imperfect released-matte reference; only empirical projection treatment differs. Same039parent,1000newupdates each.',goal_complete=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],foreground_deltas={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in rows})))
