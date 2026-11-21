"""Frozen held-out cameras, fixed person ROIs, and actual three-way renderings."""
import json,hashlib,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity
from adapter import OUT,DATA,load_official,dataset

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(pred,gt):
    mse=float(np.mean((pred-gt)**2));gray_p=pred@np.array([.299,.587,.114]);gray_g=gt@np.array([.299,.587,.114])
    return dict(psnr=float(-10*math.log10(max(mse,1e-12))),ssim=float(structural_similarity(pred,gt,channel_axis=2,data_range=1.,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),reference_gradient_mae=float(np.mean([np.abs(np.diff(gray_p,axis=k)-np.diff(gray_g,axis=k)).mean() for k in [0,1]])))
def load_params(path):return {k:v.cuda() for k,v in torch.load(path,map_location='cpu',weights_only=True).items()}
def png(array):return Image.fromarray((array.clip(0,1)*255+.5).astype(np.uint8))
def main():
    assert not (OUT/'test-metrics.json').exists(),'Evaluation already frozen; do not overwrite after inspection'
    for mode in ['static','dynamic']:assert (OUT/mode/'completion.json').exists(),f'{mode} not finished'
    protocol=json.loads((OUT/'evaluation-protocol.json').read_text());tr,h=load_official();md=json.loads((DATA/'basketball/test_meta.json').read_text());renderdir=OUT/'renders';renderdir.mkdir(exist_ok=True);assets=OUT/'assets';rows=[];previous={};panels=[]
    frozen={mode:sha(OUT/mode/'completion.json') for mode in ['static','dynamic']}
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',20);small=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',16)
    static=load_params(OUT/'static/frame-0000.pt');weights=[]
    for t in protocol['frames']:
        dynamic=load_params(OUT/f'dynamic/frame-{t:04d}.pt');weights.append(dict(frame=t,sha256=sha(OUT/f'dynamic/frame-{t:04d}.pt')))
        views=dataset(t,md)
        for c,view in enumerate(views):
            camera=int(view['filename'].split('/')[0]);gt=view['im'].permute(1,2,0).cpu().numpy();x0,y0,x1,y1=protocol['fixed_human_rois_xyxy'][str(camera)];rect=(slice(y0,y1),slice(x0,x1));images={}
            for mode,params in [('static',static),('dynamic',dynamic)]:
                rgb,_,_=tr.Renderer(view['cam'])(**h.params2rendervar(params));pred=rgb.clamp(0,1).permute(1,2,0).cpu().numpy();images[mode]=pred
                row=dict(mode=mode,frame=t,camera=camera,full_image=metric(pred,gt),person_region=metric(pred[rect],gt[rect]))
                key=(camera,mode)
                if key in previous:
                    oldpred,oldgt=previous[key];row['temporal_image_change_mae_person_region']=float(np.abs((pred[rect]-oldpred[rect])-(gt[rect]-oldgt[rect])).mean())
                previous[key]=(pred,gt);rows.append(row)
                target=renderdir/mode/str(camera);target.mkdir(parents=True,exist_ok=True);png(pred).save(target/f'{t:04d}.png')
            if camera==protocol['display_camera']:
                # Every frame uses the same source-defined ROI and displayed size.
                crops=[gt[rect],images['static'][rect],images['dynamic'][rect]];height,width=crops[0].shape[:2];canvas=Image.new('RGB',(width*3,height+55),'white');draw=ImageDraw.Draw(canvas)
                for j,(name,array) in enumerate(zip(['原始观测','静态表示','动态表示'],crops)):
                    canvas.paste(png(array),(j*width,30));draw.text((j*width+width/2,3),name,font=font,fill='#173747',anchor='mt')
                draw.text((10,height+33),f'留出相机 {camera} · 帧 {t:03d} · 相同人物区域',font=small,fill='#173747');panels.append(canvas)
                if t in protocol['display_frames']:
                    full=Image.new('RGB',(1920,394),'white');d=ImageDraw.Draw(full)
                    for j,(name,array) in enumerate(zip(['原始观测','静态表示','动态表示'],[gt,images['static'],images['dynamic']])):
                        full.paste(png(array),(j*640,34));d.text((j*640+320,5),name,font=font,fill='#173747',anchor='mt')
                    full.save(assets/f'full-scene-frame{t:03d}.png');canvas.save(assets/f'person-detail-frame{t:03d}.png')
        del dynamic,views
        print('EVALUATED frame',t,flush=True)
    panels[0].save(assets/'person-detail-comparison.gif',save_all=True,append_images=panels[1:],duration=80,loop=0,optimize=False)
    summary={}
    for mode in ['static','dynamic']:
        selected=[r for r in rows if r['mode']==mode];summary[mode]={region:{name:float(np.mean([r[region][name] for r in selected])) for name in ['psnr','ssim','reference_gradient_mae']} for region in ['full_image','person_region']};summary[mode]['temporal_image_change_mae_person_region']=float(np.mean([r['temporal_image_change_mae_person_region'] for r in selected if 'temporal_image_change_mae_person_region' in r]))
    result=dict(status='evaluated_visual_review_pending',protocol_sha256=sha(OUT/'evaluation-protocol.json'),training_completion_hashes=frozen,dynamic_checkpoint_hashes=weights,static_checkpoint_sha256=sha(OUT/'static/frame-0000.pt'),views_per_method=len(rows)//2,aggregate='unweighted mean across each frame-camera result, same fixed ROIs for both methods',summary=summary,per_view=rows,limitations=['human ROIs include some background and are not segmentation ground truth','temporal image-change residual is not optical-flow-compensated flicker or 3D tracking error','authors initialization point-cloud source cameras are undocumented','native 640x360 indoor benchmark, not street-scale completion','local gsplat port and author code temporal color regularization differ from paper rendering setup'])
    save(OUT/'test-metrics.json',result);print(json.dumps(summary,indent=2))
if __name__=='__main__':
    with torch.no_grad():main()
