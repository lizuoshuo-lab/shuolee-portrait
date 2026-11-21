"""Text-free, same-pixel visual evidence for the all59 limb screen."""
import os
import numpy as np
from PIL import Image
from audit_body_limb_screen123 import ROOT, OUT, read, save, sha, stamp


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    result=read(OUT/'result.json'); assert result['frames']==59 and result['step']==60000
    rows={r['frame']:r for r in read(ROOT/'data/training-inputs.json')['records']}
    snap=ROOT/'gaussian-run/step-060000'
    record=__import__('pathlib').Path('/mnt/d/00/记录/demo')
    selected={'frame_000016','frame_000018','frame_000023','frame_000033','frame_000040','frame_000051'}
    thumbnails=[]; pages=[]; examples=[]
    for item in result['records']:
        frame=item['frame'];r=rows[frame]
        raw=Image.open(r['image']).convert('RGB'); rgb=Image.open(snap/(frame+'_gaussian.png')).convert('RGB')
        mask=Image.open(r['mask']).convert('L')
        a=np.asarray(Image.open(snap/'render-coverage'/(frame+'_alpha.png')),dtype=float)/65535
        alpha=Image.fromarray(np.uint8(np.round(a*255))).convert('RGB')
        yy,xx=np.where(np.asarray(mask)>127)
        box=(max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(raw.width,int(xx.max())+21),min(raw.height,int(yy.max())+21))
        panels=[im.crop(box).convert('RGB') for im in (raw,rgb,mask,alpha)]
        w,h=panels[0].size; full=Image.new('RGB',(w*4,h))
        for j,im in enumerate(panels):full.paste(im,(j*w,0))
        if frame in selected:
            path=record/f'分析_全身60000步肢体筛选_{frame}_原图_高斯_mask_透明度.png'
            assert not path.exists();full.save(path)
            assert np.array_equal(np.asarray(Image.open(path))[:,:w],np.asarray(raw.crop(box)))
            examples.append(dict(frame=frame,path=str(path),sha256=sha(path),crop_integer_xyxy=list(box),resampling=False,text_added=False))
        scale=min(1,380/h,600/(w*4));thumb=full.resize((round(w*4*scale),round(h*scale)),Image.Resampling.LANCZOS)
        thumbnails.append((frame,thumb))
    for start in range(0,len(thumbnails),6):
        batch=thumbnails[start:start+6];canvas=Image.new('RGB',(1220,1180),(238,238,238))
        for j,(_,im) in enumerate(batch):canvas.paste(im,((j%2)*610,(j//2)*390))
        path=record/f'分析_全身60000步肢体筛选_总览{start//6+1:02d}_原图_高斯_mask_透明度.png'
        assert not path.exists();canvas.save(path)
        pages.append(dict(frames=[f for f,_ in batch],path=str(path),sha256=sha(path),text_added=False,uniform_thumbnail_scaling=True))
    save(OUT/'image-manifest.json',dict(updated=stamp(),result_sha256=sha(OUT/'result.json'),
        panels=['native source','Gaussian RGB','target mask','true alpha'],frames=59,pages=pages,examples=examples,
        selected_examples_scope='Prior already-reviewed failure/control identities; not training-frame selection',
        quality_accepted=False,actual_visual_review=False))
    print('LIMB_SCREEN_IMAGES',len(pages),len(examples),flush=True)


if __name__=='__main__':main()
