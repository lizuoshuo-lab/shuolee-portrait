from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-computer00-001')
run=base/'remote-review-001/hover-msv2-fast-001'
views=run/'test/ours_18000'
out=base/'visual-review'
out.mkdir(exist_ok=True)
font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',18)
summary=json.loads((run/'validated-summary.json').read_text())
per=json.loads((run/'per_view.json').read_text())['ours_18000']
split=json.loads((base/'hover-sfm-001/state.json').read_text())['appearance_split']['test']
checks=[]
for i,name in enumerate(sorted(per['PSNR'])):
    gt=np.asarray(Image.open(views/'gt'/name).convert('RGB'))
    pred=np.asarray(Image.open(views/'renders'/name).convert('RGB'))
    source=np.asarray(Image.open(base/'hover-sfm-001/dataset/images'/split[i]).convert('RGB'))
    assert gt.shape==pred.shape==source.shape
    assert np.array_equal(source,gt), (name,int(np.abs(source.astype(int)-gt.astype(int)).max()))
    psnr=-10*math.log10(np.square((gt.astype(np.float64)-pred.astype(np.float64))/255).mean())
    assert abs(psnr-per['PSNR'][name])<0.001
    checks.append({'output':name,'source':split[i],'psnr':psnr,'source_gt_identical':True})
    pair=Image.new('RGB',(2*gt.shape[1],gt.shape[0]+34),'white')
    pair.paste(Image.fromarray(gt),(0,34));pair.paste(Image.fromarray(pred),(gt.shape[1],34))
    dr=ImageDraw.Draw(pair);dr.text((12,7),'Source '+split[i],font=font,fill='#172c3b');dr.text((gt.shape[1]+12,7),f'Mini-Splatting2 | PSNR {psnr:.2f} dB',font=font,fill='#172c3b')
    pair.save(out/f'view-{i:02d}-full.png')
for page in range(2):
    canvas=Image.new('RGB',(1280,1212),'#e8ecf0')
    for j in range(4):
        i=page*4+j
        pair=Image.open(out/f'view-{i:02d}-full.png')
        pair.thumbnail((620,574),Image.Resampling.LANCZOS)
        x=(j%2)*640+10;y=(j//2)*606+28
        canvas.paste(pair,(x,y))
        ImageDraw.Draw(canvas).text((x,y-24),f'Test {i:02d} / source {split[i]}',font=font,fill='#172c3b')
    canvas.save(out/f'overview-{page+1}.png')
(out/'independent-checks.json').write_text(json.dumps({'checks':checks,'psnr_mean':float(np.mean([x['psnr'] for x in checks]))},indent=2))
print(json.dumps({'view_count':len(checks),'psnr_mean':float(np.mean([x['psnr'] for x in checks])),'review':str(out)}))
