"""CPU contact crops from the already frozen finite source shortlist only."""
from pathlib import Path
import json
import hashlib
import math
from PIL import Image, ImageDraw
import numpy as np

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/officewoman-circle24-input127')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan=json.loads((ROOT/'plan-before-source-review.json').read_text())
    ids=sorted(plan['source_review_template']);assert len(ids)==64
    out=ROOT/'source-review';out.mkdir(exist_ok=True)
    assert not (out/'source-crops.json').exists(),'Do not regenerate an existing review.'
    rows={r['frame']:r for r in plan['source_rows']};records=[]
    for frame in ids:
        r=rows[frame];assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
        im=Image.open(r['image']).convert('RGB');m=Image.open(r['mask']).convert('L');a=np.asarray(m)
        ys,xs=np.nonzero(a>0);assert len(xs)
        x0=max(0,int(xs.min())-16);x1=min(im.width,int(xs.max())+17)
        y0=max(0,int(ys.min())-16);y1=min(im.height,int(ys.max())+17);height=y1-y0
        boxes={'body':[x0,y0,x1,y1],'mask':[x0,y0,x1,y1],
               'head':[x0,y0,x1,min(y1,y0+math.ceil(.34*height))],
               'arms':[x0,y0+math.floor(.25*height),x1,min(y1,y0+math.ceil(.80*height))]}
        paths={}
        for name,b in boxes.items():
            path=out/(frame+'-'+name+'.png');(m if name=='mask' else im).crop(b).save(path)
            paths[name]=dict(path=str(path),sha256=sha(path),crop_original_xyxy=b,native_integer_crop=True,no_resize=True)
        records.append(dict(frame=frame,source_sha256=r['source_sha256'],source_mask_sha256=r['mask_sha256'],paths=paths))
    pages=[]
    for start in range(0,len(records),4):
        page=Image.new('RGB',(1800,1420),'#ececec');draw=ImageDraw.Draw(page)
        for index,r in enumerate(records[start:start+4]):
            x=(index%2)*900;y=(index//2)*710;draw.text((x+8,y+5),r['frame']+' source / mask / head / arms',fill='#000000')
            for name,ox,oy,w,h in [('body',0,25,240,675),('mask',245,25,240,675),('head',495,25,390,310),('arms',495,340,390,355)]:
                tile=Image.open(r['paths'][name]['path']).convert('RGB');tile.thumbnail((w,h),Image.Resampling.LANCZOS)
                page.paste(tile,(x+ox+(w-tile.width)//2,y+oy+(h-tile.height)//2))
        path=out/('contact-%02d.jpg'%(start//4));page.save(path,quality=94)
        pages.append(dict(path=str(path),sha256=sha(path),frames=[r['frame'] for r in records[start:start+4]]))
    result=dict(source_plan_sha256=sha(ROOT/'plan-before-source-review.json'),source_rows=records,pages=pages,
                actual_visual_review=False,source_images_modified=False,source_masks_modified=False,CPU_only=True)
    (out/'source-crops.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('SOURCE64_CONTACTS_READY',out)


if __name__=='__main__':main()
