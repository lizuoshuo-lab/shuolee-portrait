"""Read-only CPU coverage audit of the existing head-scope parser."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',required=True)
    parser.add_argument('--images',required=True)
    parser.add_argument('--face-run',required=True)
    parser.add_argument('--parser-source',required=True)
    parser.add_argument('--parser-weight',required=True)
    parser.add_argument('--detector-weight',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    sys.path.insert(0,args.package)
    from msv2.head import HeadPixelExtractor,HEAD_LABELS
    from msv2.io import read_json,save_json,sha256
    def deny(*a,**kw):raise RuntimeError('Network disabled during local CPU audit')
    socket.create_connection=deny
    socket.socket.connect=deny
    socket.socket.connect_ex=deny
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    source=Path(args.images)
    face_run=Path(args.face_run)
    srfile=face_run/'swinir4x/superresolution.json'
    sr=read_json(srfile)
    face_rows={r['name']:r for r in sr['records']}
    paths=sorted(source.glob('frame_*.png'))
    assert len(paths)==65
    before={str(p):sha256(p) for p in paths}
    before[str(srfile)]=sha256(srfile)
    code=Path(args.package)/'msv2/head.py'
    before[str(code)]=sha256(code)
    extractor=HeadPixelExtractor(args.parser_source,args.parser_weight,args.detector_weight,mask_scope='head')
    assert list(HEAD_LABELS)==[*range(1,14),17,18]
    assert all(p.device.type=='cpu' for p in extractor.model.parameters())
    captured=[]
    parse_original=extractor._parse
    def capture(rgb):
        labels=parse_original(rgb)
        captured.append(labels.copy())
        return labels
    extractor._parse=capture
    assets=extractor.assets
    for section in (assets['parser'],assets['detector']):
        for key,val in section.items():
            if key.endswith('_sha256'):pass
    fontpath=Path('/mnt/c/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(fontpath),18)
    small=ImageFont.truetype(str(fontpath),14)
    manifest=dict(schema='msv2-whole-head-existing-parser-audit-v1',state='running',
        started=time.time(),mask_scope='head',labels=list(HEAD_LABELS),assets=assets,
        method='Existing HeadPixelExtractor.regions without any parameter/ROI/model modification',
        model_device='cpu',CUDA_VISIBLE_DEVICES='-1',network_connections_denied=True,
        includes_all_long_hair_claimed=False,whole_head_coverage_accepted=False,
        records=[],protected_before=before)
    save_json(output/'audit.json',manifest)
    for number,path in enumerate(paths,1):
        started=time.perf_counter()
        folder=output/'frames'/path.stem;folder.mkdir(parents=True)
        with Image.open(path) as image:rgb=np.array(image.convert('RGB'),copy=True)
        rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
        assert sha256(path)==before[str(path)]
        prepared=face_run/'prepared/dataset/images'/path.name
        assert sha256(prepared)==face_rows[path.name]['native_sha256']
        before[str(prepared)]=sha256(prepared)
        assert rgbsha==face_rows[path.name]['face_extraction']['source_rgb_sha256']
        assert [rgb.shape[1],rgb.shape[0]]==face_rows[path.name]['face_extraction']['source_wh']
        with Image.open(prepared) as prepared_image:
            assert hashlib.sha256(np.asarray(prepared_image.convert('RGB')).tobytes()).hexdigest()==rgbsha
        captured.clear()
        regions,meta=extractor.regions(rgb)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==rgbsha
        union=np.zeros(rgb.shape[:2],bool)
        for region in regions:union|=region['mask_native']
        oldface=np.zeros(rgb.shape[:2],bool)
        for face in face_rows[path.name]['faces']:
            maskpath=face_run/'swinir4x/face-masks'/face['mask_name']
            assert sha256(maskpath)==face['mask_sha256']
            before[str(maskpath)]=sha256(maskpath)
            with Image.open(maskpath) as im:
                assert im.size==(rgb.shape[1],rgb.shape[0])
                oldface|=np.asarray(im)>0
        roi_records=[]
        for r,labels in zip(meta['records'],captured):
            x0,y0,x1,y1=r['segmentation_roi_native']
            native_labels=np.asarray(Image.fromarray(labels).resize((x1-x0,y1-y0),Image.Resampling.NEAREST))
            local=np.isin(native_labels,HEAD_LABELS)
            hair=native_labels==17
            hats=native_labels==18
            def contacts(mask):
                return dict(left=int(mask[:,0].sum()),right=int(mask[:,-1].sum()),
                            top=int(mask[0,:].sum()),bottom=int(mask[-1,:].sum()))
            roi_records.append(dict(segmentation_roi_native=[x0,y0,x1,y1],bbox_native=r['bbox_native'],
                head_border_pixels=contacts(local),hair_border_pixels=contacts(hair),hat_border_pixels=contacts(hats),
                label_pixels_native={str(i):int((native_labels==i).sum()) for i in range(19)},
                head_bbox_touches_locator={side:contacts(local)[side]>0 for side in ('left','right','top','bottom')},
                includes_all_long_hair=r['includes_all_long_hair']))
        if meta['records']:
            x0,y0,x1,y1=meta['records'][0]['segmentation_roi_native']
            rw,rh=x1-x0,y1-y0
            context=[max(0,x0-rw),max(0,y0-rh),min(rgb.shape[1],x1+rw),min(rgb.shape[0],y1+rh)]
        else:
            context=[0,0,rgb.shape[1],rgb.shape[0]]
        a,b,c,d=context
        source_crop=rgb[b:d,a:c]
        head_crop=union[b:d,a:c]
        face_crop=oldface[b:d,a:c]
        previews=[]
        for mask,color in [(None,None),(head_crop,[20,255,40]),(face_crop,[0,220,255])]:
            picture=source_crop.copy()
            if mask is not None:
                picture[mask]=np.rint(picture[mask]*.55+np.asarray(color)*.45).astype(np.uint8)
            im=Image.fromarray(picture)
            draw=ImageDraw.Draw(im)
            for r in roi_records:
                q=r['segmentation_roi_native']
                draw.rectangle((q[0]-a,q[1]-b,q[2]-a-1,q[3]-b-1),outline='#ff00ff',width=2)
            previews.append(im)
        width,height=source_crop.shape[1],source_crop.shape[0]
        canvas=Image.new('RGB',(3*width,height+66),'white')
        draw=ImageDraw.Draw(canvas)
        for index,(im,label) in enumerate(zip(previews,('原像素；紫框=定位ROI','现有head mask绿色（含发/帽）','前版face mask青色'))):
            canvas.paste(im,(index*width,66))
            draw.text((index*width+8,7),f'{path.stem} | {label}',font=font,fill='black')
            draw.text((index*width+8,34),'预览区域扩大只用于看覆盖；不改变parser输入。',font=small,fill='#333333')
        canvas.save(folder/'head_vs_face.png')
        Image.fromarray(union.astype(np.uint8)*255).save(folder/'head_mask_native.png')
        ys,xs=np.nonzero(union)
        bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else None
        if bbox:
            l,t,r,bt=bbox
            Image.fromarray(np.dstack((rgb[t:bt,l:r],union[t:bt,l:r].astype(np.uint8)*255))).save(folder/'head_pixels_native.png')
        # Keep a one-panel, contextual head overlay for the65-frame contact sheet.
        previews[1].save(folder/'head_overlay_context.png')
        entry=dict(frame=path.stem,source_png_sha256=before[str(path)],source_rgb_sha256=rgbsha,
            prepared_png_sha256=before[str(prepared)],source_prepared_RGB_identical=True,
            source_wh=[rgb.shape[1],rgb.shape[0]],head_count=len(regions),head_bbox=bbox,
            head_pixels=int(union.sum()),face_pixels=int(oldface.sum()),
            face_pixels_not_in_head=int((oldface & ~union).sum()),added_head_pixels=int((union & ~oldface).sum()),
            head_mask_sha256=sha256(folder/'head_mask_native.png'),
            roi_records=roi_records,metadata=meta,preview_context=context,
            seconds=time.perf_counter()-started)
        manifest['records'].append(entry)
        save_json(output/'audit.json',manifest)
        if number%8==0 or number==65:
            print(json.dumps(dict(done=number,head_pixels=entry['head_pixels'],roi_contacts=roi_records[0]['head_border_pixels'] if roi_records else {})),flush=True)
    final={path:sha256(path) for path in before}
    assert final==before
    assert not extractor.torch.cuda.is_initialized()
    manifest.update(state='completed',finished=time.time(),source_images=65,
        head_images=sum(r['head_count']>0 for r in manifest['records']),
        roi_border_touch_images=sum(any(any(v>0 for v in rr['head_border_pixels'].values()) for rr in r['roi_records']) for r in manifest['records']),
        hair_border_touch_images=sum(any(any(v>0 for v in rr['hair_border_pixels'].values()) for rr in r['roi_records']) for r in manifest['records']),
        face_superset_images=sum(r['face_pixels_not_in_head']==0 for r in manifest['records']),
        protected_after=final,protected_unchanged=True,GPU_used=False,CUDA_initialized=False)
    # Contact sheets preserve all originals; only equal-size aspect thumbnailing.
    for start in range(0,65,8):
        batch=manifest['records'][start:start+8]
        sheet=Image.new('RGB',(1200,720),'white');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'现有head scope覆盖审计 {start+1}–{start+len(batch)} /65 | 绿色=head mask，紫框=parser定位ROI',font=font,fill='black')
        for index,item in enumerate(batch):
            with Image.open(output/'frames'/item['frame']/'head_overlay_context.png') as im:
                im=im.convert('RGB');im.thumbnail((290,290),Image.Resampling.LANCZOS)
                x=(index%4)*300;y=43+(index//4)*330
                sheet.paste(im,(x+(300-im.width)//2,y+31+(290-im.height)//2))
                draw.text((x+8,y+4),f"{item['frame']} | 触边={any(any(v for v in rr['head_border_pixels'].values()) for rr in item['roi_records'])}",font=small,fill='black')
        sheet.save(output/f'overview_{start//8+1:02d}.png')
    save_json(output/'audit.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('state','source_images','head_images','roi_border_touch_images','hair_border_touch_images','face_superset_images','GPU_used','protected_unchanged')},ensure_ascii=False))


if __name__=='__main__':main()
