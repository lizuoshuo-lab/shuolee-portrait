"""Evaluate matched color-reference forks without replacing the original baseline."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from adapter import LAB,DATA,load_official,dataset
from evaluate_benchmark import metric,load_params,png,sha

OUT=LAB/'outputs/dynamic-human-color-continuity-004'
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    protocol=json.loads((OUT/'evaluation-protocol.json').read_text());deadline=time.monotonic()+6*3600
    while not all((OUT/mode/'completion.json').exists() for mode in ['aliased','frozen']):
        if time.monotonic()>deadline:raise TimeoutError('Color-reference training completion timeout')
        time.sleep(10)
    assert not (OUT/'diagnostic-metrics.json').exists(),'Completed evaluation cannot be replaced'
    for mode in ['aliased','frozen']:
        completion=json.loads((OUT/mode/'completion.json').read_text());assert completion['frames']==31 and completion['variant_updates']==60000
    for item in protocol['implementation_files']:assert sha(LAB/item['path'])==item['sha256'],item['path']
    torch.set_num_threads(8);tr,h=load_official();md=json.loads((DATA/'basketball/test_meta.json').read_text());assets=OUT/'assets';assets.mkdir(exist_ok=True);rows=[];panels={c:[] for c in [0,10,15,30]};previous={};float_png_error=[];first_frame_equal=[]
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',20);small=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',15)
    for t in protocol['frames']:
        params={m:load_params(OUT/m/f'frame-{t:04d}.pt') for m in ['aliased','frozen']}
        for view in dataset(t,md):
            camera=int(view['filename'].split('/')[0]);gt=view['im'].permute(1,2,0).cpu().numpy();roi=protocol['fixed_human_rois_xyxy'][str(camera)];x0,y0,x1,y1=roi;rect=(slice(y0,y1),slice(x0,x1));images={}
            for mode,p in params.items():
                rgb,_,_=tr.Renderer(view['cam'])(**h.params2rendervar(p));pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy();images[mode]=pred
                row=dict(mode=mode,frame=t,camera=camera,full_image=metric(pred,gt),person_region=metric(pred[rect],gt[rect]));key=(camera,mode)
                if key in previous:
                    a,b=previous[key];row['temporal_image_change_mae_person_region']=float(np.abs((pred[rect]-a[rect])-(gt[rect]-b[rect])).mean())
                previous[key]=(pred,gt);rows.append(row);dest=OUT/f'renders/{mode}/{camera}';dest.mkdir(parents=True,exist_ok=True);pimage=png(pred);pimage.save(dest/f'{t:04d}.png');quantized=np.asarray(pimage).astype(np.float32)/255
                float_png_error.append(abs(metric(quantized[rect],gt[rect])['psnr']-row['person_region']['psnr']))
            if t==0:first_frame_equal.append(bool(np.array_equal(images['aliased'],images['frozen'])))
            w=x1-x0;hh=y1-y0;panel=Image.new('RGB',(w*3,hh+57),'white');d=ImageDraw.Draw(panel)
            for j,(label,im) in enumerate(zip(['原始观测','未修正参照','冻结颜色参照'],[gt,images['aliased'],images['frozen']])):
                panel.paste(png(im[rect]),(j*w,30));d.text((j*w+w/2,3),label,font=font,anchor='mt',fill='#183747')
            d.text((8,hh+34),f'相机 {camera} · 帧 {t:03d} · 相同首帧、优化器和更新预算',font=small,fill='#183747');panels[camera].append(panel)
            if (camera,t) in [(10,15),(10,30),(15,21),(15,30)]:panel.save(assets/f'color-camera{camera}-frame{t:03d}.png')
        print('COLOR EVALUATED',t,flush=True)
        del params
    assert all(first_frame_equal),'Matched forks must render identically at the shared first frame'
    assert max(float_png_error)<.08
    summary={}
    for mode in ['aliased','frozen']:
        rs=[r for r in rows if r['mode']==mode];summary[mode]={region:{name:float(np.mean([r[region][name] for r in rs])) for name in ['psnr','ssim','reference_gradient_mae']} for region in ['full_image','person_region']};summary[mode]['temporal_image_change_mae_person_region']=float(np.mean([r['temporal_image_change_mae_person_region'] for r in rs if 'temporal_image_change_mae_person_region' in r]))
    for camera,images in panels.items():
        images[0].save(assets/f'color-camera{camera}-all31.gif',save_all=True,append_images=images[1:],duration=80,loop=0,optimize=False)
        w,hh=images[0].size;sheet=Image.new('RGB',(w,hh*3),'white')
        for j,t in enumerate([0,15,30]):sheet.paste(images[t],(0,j*hh))
        sheet.save(assets/f'color-camera{camera}-frames000-015-030.png')
    save(OUT/'diagnostic-metrics.json',dict(status='diagnostic_evaluated_visual_review_pending',evaluation_role=protocol['evaluation_role'],protocol_sha256=sha(OUT/'evaluation-protocol.json'),first_frame_render_equal_all_cameras=all(first_frame_equal),png_float_psnr_max_difference_db=max(float_png_error),paired_views=124,summary=summary,per_view=rows))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
