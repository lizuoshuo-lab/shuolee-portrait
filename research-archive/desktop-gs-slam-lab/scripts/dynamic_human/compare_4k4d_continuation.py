"""Generate unenhanced, native-pixel paired images after both renders are audited."""
from pathlib import Path
import os,json,hashlib,numpy as np
from PIL import Image,ImageDraw
LAB=Path(__file__).resolve().parents[2]
PARENT=LAB/'outputs/dynamic-human-4k4d-official-035'
CHILD=PARENT/'continuation-10000'
OUT=CHILD/'paired-comparison';OUT.mkdir(exist_ok=True)
for root in [PARENT,CHILD]:
    check=json.loads((root/'native-evaluation/independent-image-audit.json').read_text())
    assert check['full_pngs']==24 and check['native_comparison_panels']==12
rows={root:json.loads((root/'native-evaluation/metrics.json').read_text())['records'] for root in [PARENT,CHILD]}
results=[]
for a,b in zip(rows[PARENT],rows[CHILD]):
    assert a['camera']==b['camera'] and a['bbox']==b['bbox'] and a['source_cameras']==b['source_cameras']
    assert a['source_rgb_sha256']==b['source_rgb_sha256'] and a['source_matte_sha256']==b['source_matte_sha256']
    name=a['camera'];prefix=f'camera{name}-frame000750';parents=PARENT/'native-evaluation';children=CHILD/'native-evaluation'
    reference=np.array(Image.open(parents/f'{prefix}-reference.png'))
    assert np.array_equal(reference,np.array(Image.open(children/f'{prefix}-reference.png')))
    before=np.array(Image.open(parents/f'{prefix}-trained.png'));after=np.array(Image.open(children/f'{prefix}-trained.png'))
    x0,y0,x1,y1=a['bbox'];ym=y0+round(.6*(y1-y0));images=[]
    for label,(xx0,yy0,xx1,yy1) in [('full-person',a['bbox']),('upper',[x0,y0,x1,ym])]:
        w,h=xx1-xx0,yy1-yy0;canvas=Image.new('RGB',(3*w,h+38),'white');draw=ImageDraw.Draw(canvas)
        for j,(title,array) in enumerate([('Reference (released matte)',reference),('Official4K4D 1000 updates',before),('Official4K4D 10000 updates',after)]):
            canvas.paste(Image.fromarray(array[yy0:yy1,xx0:xx1]),(j*w,38));draw.text((j*w+8,10),title,fill='black')
        path=OUT/f'{prefix}-{label}.png';canvas.save(path)
        saved=np.array(Image.open(path))
        for j,array in enumerate([reference,before,after]):assert np.array_equal(saved[38:,j*w:(j+1)*w],array[yy0:yy1,xx0:xx1])
        images.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_pixel_content_exact=True))
    delta={key:b['metrics'][key]['psnr_db']-a['metrics'][key]['psnr_db'] for key in a['metrics']}
    results.append(dict(camera=name,split=a['split'],before=a['metrics'],after=b['metrics'],psnr_delta_db=delta,images=images))
report=dict(status='sixviews_matched_references_and_native_panels_verified_visual_review_pending',records=results,
    scope='Same frame, resolution, camera calibration, matte, rendering path, source cameras, and region definitions. Development views already inspected at1000updates; not a fresh final test.',
    enhancement='none',goal_complete=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status=report['status'],foreground_deltas=[dict(camera=r['camera'],delta=r['psnr_delta_db']['foreground_matte_gt0_5']) for r in results])))
