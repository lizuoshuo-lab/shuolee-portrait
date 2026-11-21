"""All held-out views in the predeclared second temporal interval, with real renders."""
import json,time,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from adapter import LAB,DATA,load_official,dataset
from evaluate_benchmark import metric,load_params,png,sha

OUT=LAB/'outputs/dynamic-human-temporal-validation-006'


def main():
    protocol=json.loads((OUT/'evaluation-protocol.json').read_text());deadline=time.monotonic()+6*3600
    while not all((OUT/mode/'completion.json').exists() for mode in ['aliased','frozen']):
        if time.monotonic()>deadline:raise TimeoutError('Validation training did not finish')
        time.sleep(10)
    assert not (OUT/'validation-metrics.json').exists()
    for item in protocol['implementation_files']:assert sha(LAB/item['path'])==item['sha256']
    torch.set_num_threads(8);tr,h=load_official();md=json.loads((DATA/'basketball/test_meta.json').read_text())
    assets=OUT/'assets';assets.mkdir(exist_ok=True);rows=[];panels={c:[] for c in [0,10,15,30]};previous={};quantization=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for t in protocol['frames']:
        params={m:load_params(OUT/m/f'frame-{t:04d}.pt') for m in ['aliased','frozen']}
        for view in dataset(t,md):
            camera=int(view['filename'].split('/')[0]);gt=view['im'].permute(1,2,0).cpu().numpy()
            x0,y0,x1,y1=protocol['fixed_human_rois_xyxy'][str(camera)];rect=(slice(y0,y1),slice(x0,x1));images={}
            for mode,p in params.items():
                rgb,_,_=tr.Renderer(view['cam'])(**h.params2rendervar(p));pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy();images[mode]=pred
                row=dict(mode=mode,frame=t,camera=camera,full_image=metric(pred,gt),person_region=metric(pred[rect],gt[rect]));key=(camera,mode)
                if key in previous:
                    a,b=previous[key];row['temporal_image_change_mae_person_region']=float(np.abs((pred[rect]-a[rect])-(gt[rect]-b[rect])).mean())
                previous[key]=(pred,gt);rows.append(row)
                dest=OUT/f'renders/{mode}/{camera}';dest.mkdir(parents=True,exist_ok=True);im=png(pred);im.save(dest/f'{t:04d}.png')
                q=np.asarray(im).astype(np.float32)/255;quantization.append(abs(metric(q[rect],gt[rect])['psnr']-row['person_region']['psnr']))
            w=x1-x0;hh=y1-y0;panel=Image.new('RGB',(3*w,hh+57),'white');d=ImageDraw.Draw(panel)
            for j,(label,im) in enumerate(zip(['原始观测','未修正参照','冻结颜色参照'],[gt,images['aliased'],images['frozen']])):
                panel.paste(png(im[rect]),(j*w,30));d.text((j*w+w/2,3),label,font=font,anchor='mt',fill='#183747')
            d.text((8,hh+34),f'相机 {camera} · 帧 {t:03d} · 后续区间留出视角验证',font=small,fill='#183747');panels[camera].append(panel)
            if t in [31,45,60]:panel.save(assets/f'camera{camera}-frame{t:03d}.png')
        print('VALIDATION EVALUATED',t,flush=True);del params
    assert max(quantization)<.08
    summary={}
    for mode in ['aliased','frozen']:
        rs=[r for r in rows if r['mode']==mode]
        summary[mode]={region:{name:float(np.mean([r[region][name] for r in rs])) for name in ['psnr','ssim','reference_gradient_mae']} for region in ['full_image','person_region']}
        summary[mode]['temporal_image_change_mae_person_region']=float(np.mean([r['temporal_image_change_mae_person_region'] for r in rs if 'temporal_image_change_mae_person_region' in r]))
    for camera,images in panels.items():
        images[0].save(assets/f'camera{camera}-all30.gif',save_all=True,append_images=images[1:],duration=80,loop=0,optimize=False)
        w,hh=images[0].size;sheet=Image.new('RGB',(w,hh*3),'white')
        for j,t in enumerate([31,45,60]):sheet.paste(images[t-31],(0,j*hh))
        sheet.save(assets/f'camera{camera}-frames031-045-060.png')
    report=dict(status='validation_evaluated_visual_review_pending',evaluation_role=protocol['evaluation_role'],
        protocol_sha256=sha(OUT/'evaluation-protocol.json'),paired_views=120,summary=summary,per_view=rows,
        png_float_psnr_max_difference_db=max(quantization))
    (OUT/'validation-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
