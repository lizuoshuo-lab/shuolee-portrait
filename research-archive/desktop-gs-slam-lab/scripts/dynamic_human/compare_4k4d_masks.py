"""Same-budget true renders, with original photograph visible beside them."""
from pathlib import Path
import json,numpy as np,hashlib
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2]
BASE=LAB/'outputs/dynamic-human-4k4d-official-035'
NEW=LAB/'outputs/dynamic-human-4k4d-mask-control-039';OUT=NEW/'paired-comparison';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
for p in [BASE,NEW]:assert read(p/'native-evaluation/independent-image-audit.json')['full_pngs']==24
a=read(BASE/'native-evaluation/metrics.json')['records'];b=read(NEW/'native-evaluation/metrics.json')['records'];rows=[]
for x,y in zip(a,b):
    c=x['camera'];assert c==y['camera'] and x['bbox']==y['bbox'] and x['source_cameras']==y['source_cameras']
    assert x['source_rgb_sha256']==y['source_rgb_sha256'] and x['source_matte_sha256']==y['source_matte_sha256']
    prefix=f'camera{c}-frame000750'
    photo=np.array(Image.open(BASE/'native-evaluation'/f'{prefix}-source.png'))
    before=np.array(Image.open(BASE/'native-evaluation'/f'{prefix}-trained.png'))
    after=np.array(Image.open(NEW/'native-evaluation'/f'{prefix}-trained.png'))
    assert np.array_equal(photo,np.array(Image.open(NEW/'native-evaluation'/f'{prefix}-source.png')))
    assert np.array_equal(np.array(Image.open(BASE/'native-evaluation'/f'{prefix}-reference.png')),np.array(Image.open(NEW/'native-evaluation'/f'{prefix}-reference.png')))
    boxes=[('full-person',x['bbox'])]
    if c=='01':boxes.append(('forearm',[736,430,824,610]))
    panels=[]
    for label,(x0,y0,x1,y1) in boxes:
        w=x1-x0;h=y1-y0;im=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(im)
        for j,(title,array) in enumerate([('Original photo',photo),('Released-mask1000',before),('Candidate-mask1000',after)]):
            im.paste(Image.fromarray(array[y0:y1,x0:x1]),(j*w,32));d.text((j*w+4,10),title,fill='black')
        p=OUT/f'{prefix}-{label}.png';im.save(p);v=np.array(Image.open(p))
        for j,array in enumerate([photo,before,after]):assert np.array_equal(v[32:,j*w:(j+1)*w],array[y0:y1,x0:x1])
        panels.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    rows.append(dict(camera=c,split=x['split'],same_texture_source_cameras=x['source_cameras'],before=x['metrics'],after=y['metrics'],
        psnr_delta_db={key:y['metrics'][key]['psnr_db']-x['metrics'][key]['psnr_db'] for key in x['metrics']},panels=panels))
report=dict(status='same1000budget6views7panels_verified_visual_review_pending',records=rows,
    scope='Input-mask pipeline including train-only hull/crop/source premultiplication changed. Both1000updates, sameK/R/T, imagecontent and sourcecamera membership. Not loss-only isolatedablation; released targetmatte remains imperfect.',goal_complete=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],foreground_deltas={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in rows})))
