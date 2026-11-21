"""Compare paired static reconstructions against unchanged video observations."""
from pathlib import Path
import argparse
import copy
import html
import importlib.util
import json
import shutil
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/opt/gs-slam-lab/static-person-upsampling-20261102')
DELIVERY = Path('/mnt/d/00/记录/2026-11-02_人体上采样静止重建')


def main(subject, native_input=False):
    root = (Path('/opt/gs-slam-lab/static-person-native-upsampling-20261102') if native_input else BASE)/subject
    spec = importlib.util.spec_from_file_location('paired_static', root/'experiment.py')
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
    ex.select_subject(subject)
    rt, torch = ex.runtime()
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.training import Views, camera, image, foreground_loss
    import cv2
    out = DELIVERY/(subject+'_native' if native_input else subject); out.mkdir(parents=True, exist_ok=True)
    plan = ex.read(root/'plan.json')
    video_path = plan['source_video']
    if (root/'video-path-resolution.json').exists():
        video_path = ex.read(root/'video-path-resolution.json')['resolved_video']
    assert ex.sha(video_path) == plan['source_video_sha256'], 'Original video identity changed.'
    rows = ex.read(root/'original/data/training-inputs.json')['records']
    chosen = set(np.linspace(0, len(rows)-1, 6).astype(int).tolist())
    views = Views(rows, rows[0]['wh'][1], 1)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device='cuda')
    arms = ['original', 'upsampled2x']
    video = cv2.VideoCapture(video_path)
    native_wh = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    src_rows = ex.read(Path(plan['source_run'])/'data/training-inputs.json')['records']
    src_wh = plan.get('crop_coordinate_wh', src_rows[0]['wh'])
    crop = np.asarray(plan['crop_xyxy'], dtype=float)
    native_crop = crop*np.tile(np.asarray(native_wh)/src_wh, 2)
    target_wh = np.rint((crop[2:]-crop[:2])*np.asarray(native_wh)/src_wh).astype(int).tolist()
    native_rows = []
    for row in rows:
        rr = copy.deepcopy(row)
        rr['K'] = (np.diag([target_wh[0]/row['wh'][0],target_wh[1]/row['wh'][1],1])@np.asarray(row['K'])).tolist()
        rr['wh'] = target_wh
        native_rows.append(rr)
    fit = dict(np.load(root/'original/fit/fitted.npz', allow_pickle=False))
    pts = np.load(root/'original/clothing-init/points.npz', allow_pickle=False)['xyz']
    center = np.median(pts, axis=0).astype(float)
    up = np.asarray(fit['root_rotation'][0][:,1],dtype=float); up /= np.linalg.norm(up)
    body_height = np.quantile(pts@up,.99)-np.quantile(pts@up,.01)
    E0 = np.asarray(rows[len(rows)//2]['w2c']); eye0 = -E0[:3,:3].T@E0[:3,3]
    a = eye0-center; a -= up*np.dot(a,up); a /= np.linalg.norm(a)
    b = np.cross(up,a); radius = max(1.6*body_height,.1)
    orbit_rows = []
    for deg in range(0,360,30):
        angle = np.deg2rad(deg)
        eye = center+radius*(np.cos(angle)*a+np.sin(angle)*b)+.08*radius*up
        forward = center-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward,up); right /= np.linalg.norm(right)
        down = np.cross(forward,right)
        E=np.eye(4); E[:3,:3]=np.stack([right,down,forward]); E[:3,3]=-E[:3,:3]@eye
        orbit_rows.append(dict(frame=f'orbit_{deg:03d}',wh=[512,768],K=[[768,0,256],[0,768,384],[0,0,1]],w2c=E.tolist(),angle_degrees=deg))
    metrics, models = {}, {}
    for arm in arms:
        result=ex.read(root/arm/'gaussian-run/result.json')
        assert result['steps']==18000 and result['schedule_steps']==18000
        source=root/arm/'gaussian-run/step-018000/world.ply'
        parity=ex.read(source.parent/'native-ply-parity.json')
        assert parity['passed'] and parity['sha256']==ex.sha(source)
        model=GaussianModel(3); model.load_ply(str(source)); model.active_sh_degree=3
        target=out/(arm+'.ply'); shutil.copy2(source,target)
        assert ex.sha(target)==ex.sha(source)
        models[arm]=dict(path=target.name,sha256=ex.sha(target),points=len(model.get_xyz),steps=18000,
            coordinate_space='Original scene world; accompanying OpenCV w2c cameras',native_ply_parity=parity)
        records=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                cam,gt,mask=views.get(i)
                pred=render(cam,model,pipe,bg)['render']
                aa=render(cam,model,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'][:1]
                _,l1,ssim=foreground_loss(pred,gt,mask)
                fg=mask[0]>.99
                mse=(pred-gt).square().mean(0)[fg].mean()
                box=np.asarray(row['face_box_xyxy']).reshape(2,2)
                x0,y0=np.maximum(0,np.floor(box[0]).astype(int)); x1,y1=np.ceil(box[1]).astype(int)
                x1=min(x1,gt.shape[2]);y1=min(y1,gt.shape[1])
                face_fg=fg[y0:y1,x0:x1]
                face_error=(pred-gt).abs().mean(0)[y0:y1,x0:x1][face_fg]
                assert face_error.numel()>0
                rec=dict(frame=row['frame'],foreground_l1=float(l1),foreground_ssim=float(ssim),
                    foreground_psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),
                    face_foreground_l1=float(face_error.mean()),face_foreground_pixels=int(face_error.numel()),
                    alpha_mae=float((aa-mask).abs().mean()),
                    mask_iou=float(((aa[0]>.5)&(mask[0]>.5)).sum()/((aa[0]>.5)|(mask[0]>.5)).sum()))
                records.append(rec)
                if i in chosen:
                    f=out/row['frame'];f.mkdir(exist_ok=True)
                    image(pred).save(f/(arm+'_body.png'))
                    nc,_=camera(native_rows[i],i)
                    native=render(nc,model,pipe,bg)['render']
                    im=image(native);im.save(f/(arm+'_native.png'))
                    if arm=='original':
                        video.set(cv2.CAP_PROP_POS_FRAMES,row['source_frame_index']);ok,bgr=video.read();assert ok
                        raw=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                        assert list(raw.size)==native_wh
                        raw.save(f/'input_source_frame.png')
                        raw=raw.transform(tuple(target_wh),Image.Transform.EXTENT,tuple(native_crop),Image.Resampling.BICUBIC)
                        raw.save(f/'input_native.png')
                    margin=np.maximum((box[1]-box[0])*.3,12)
                    face_box=np.r_[box[0]-margin,box[1]+margin]
                    ratio=np.tile(np.asarray(target_wh)/row['wh'],2)
                    face_box=np.rint((face_box+.5)*ratio-.5).astype(int)
                    face_box[:2]=np.maximum(face_box[:2],0);face_box[2:]=np.minimum(face_box[2:],target_wh)
                    im.crop(tuple(face_box)).save(f/(arm+'_face.png'))
                    if arm=='original':raw.crop(tuple(face_box)).save(f/'input_face.png')
                if i%20==0:print(subject,arm,'evaluated',i+1,'/',len(rows),flush=True)
            for j,row in enumerate(orbit_rows):
                cam,_=camera(row,j)
                folder=out/'orbit';folder.mkdir(exist_ok=True)
                image(render(cam,model,pipe,bg)['render']).save(folder/(row['frame']+'_'+arm+'.png'))
        metrics[arm]=records
        del model;torch.cuda.empty_cache()
    video.release()
    names=['foreground_l1','foreground_ssim','foreground_psnr','face_foreground_l1','alpha_mae','mask_iou']
    summary={a:{k:float(np.mean([r[k] for r in metrics[a]])) for k in names} for a in arms}
    paired={k:dict(mean_change=summary['upsampled2x'][k]-summary['original'][k],
        lower_frames=sum(b[k]<a[k] for a,b in zip(metrics['original'],metrics['upsampled2x'])),
        higher_frames=sum(b[k]>a[k] for a,b in zip(metrics['original'],metrics['upsampled2x']))) for k in names}
    ex.save(out/'comparison.json',dict(updated=ex.now(),subject=subject,frames=len(rows),plan=plan,models=models,
        metrics=metrics,mean_metrics=summary,paired_changes=paired,original_video_wh=native_wh,resolved_source_video=video_path,
        common_evaluation_wh=rows[0]['wh'],native_comparison_wh=target_wh,
        evaluation='All training frames evaluated against unchanged original observations at identical cameras/resolution; face metric excludes background.',
        novel_views=12,novel_view_ground_truth=False,quality_accepted=False,review_script_sha256=ex.sha(__file__)))
    ex.save(out/'cameras.json',dict(convention='OpenCV world-to-camera; K uses pixel corners',models_share_same_world=True,
        records=[{k:r[k] for k in ['frame','source_frame_index','time_seconds','wh','K','w2c']} for r in rows],
        novel_review_cameras=orbit_rows))
    render_report(out,subject,summary,sorted(chosen),rows)
    print(json.dumps(dict(subject=subject,mean_metrics=summary,paired_changes=paired)),flush=True)


def render_report(out,subject,summary,selected,rows):
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    panels=[];faces=[]
    for i in selected:
        frame=rows[i]['frame'];folder=out/frame
        ims=[Image.open(folder/(a+'_face.png')).convert('RGB') for a in ['input','original','upsampled2x']]
        width=max(240,max(im.width for im in ims));height=max(im.height for im in ims)+66
        board=Image.new('RGB',(width*3,height),'#eeeeee');draw=ImageDraw.Draw(board)
        for j,(label,im) in enumerate(zip(['原视频','原尺寸训练','2倍上采样训练'],ims)):
            draw.text((j*width+10,5),label,font=font,fill='black');board.paste(im,(j*width+(width-im.width)//2,62))
        draw.text((10,32),frame,font=font,fill='black');board.save(folder/'face_comparison.png');faces.append(board)
        body=Image.new('RGB',(900,560),'#eeeeee');bd=ImageDraw.Draw(body)
        for j,(label,a) in enumerate(zip(['原视频','原尺寸训练','2倍上采样训练'],['input','original','upsampled2x'])):
            im=Image.open(folder/(a+'_native.png')).convert('RGB');im.thumbnail((300,520),Image.Resampling.LANCZOS)
            bd.text((j*300+10,5),label,font=font,fill='black');body.paste(im,(j*300+(300-im.width)//2,38))
        body.save(folder/'body_comparison.png')
        panels.append(f'<section id="{frame}"><h2>{frame}</h2><img class="comparison" src="{frame}/face_comparison.png"><details><summary>完整人体对照</summary><img class="comparison" src="{frame}/body_comparison.png"></details></section>')
    maxw=max(im.width for im in faces);totalh=sum(im.height for im in faces)
    contact=Image.new('RGB',(maxw,totalh),'#eeeeee');y=0
    for im in faces:contact.paste(im,(0,y));y+=im.height
    contact.save(out/'face_contact.png')
    for start in [0,6]:
        board=Image.new('RGB',(800,1020),'#eeeeee');draw=ImageDraw.Draw(board)
        for slot,angle in enumerate(range(start*30,(start+6)*30,30)):
            x=(slot%2)*400;y=(slot//2)*340
            draw.text((x+5,y+5),f'{angle}°   原尺寸 / 2倍',font=font,fill='black')
            for j,arm in enumerate(['original','upsampled2x']):
                im=Image.open(out/'orbit'/f'orbit_{angle:03d}_{arm}.png');im.thumbnail((200,300),Image.Resampling.LANCZOS)
                board.paste(im,(x+j*200,y+35))
        board.save(out/f'orbit_contact_{start//6}.png')
    title={'sofa':'沙发坐姿','demo':'demo'}[subject]
    labels={'face_foreground_l1':'脸部误差 ↓','foreground_l1':'人体误差 ↓','foreground_ssim':'人体SSIM ↑','mask_iou':'轮廓IoU ↑'}
    cells=''.join(f'<tr><td>{label}</td><td>{summary["original"][k]:.5f}</td><td>{summary["upsampled2x"][k]:.5f}</td></tr>' for k,label in labels.items())
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · 人体上采样对照</title><style>body{{font:17px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#f4f5f7;color:#17202a}}h1,h2{{line-height:1.3}}section{{background:white;border-radius:10px;padding:18px;margin:24px 0}}img.comparison{{max-width:100%;height:auto}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px 20px;border:1px solid #ccd1d8}}a{{color:#1763ad}}summary{{cursor:pointer}}.note{{color:#505b68}}</style><h1>{title}：原尺寸与2倍上采样</h1><p>两组均使用沙发版静止衣物表面流程，各训练18,000步。相机、初始表面和随机种子相同；2倍组使用Lanczos插值，并同步调整内参和训练分辨率。</p><p class="note">插值不增加真实观测细节。下表在相同原尺寸下与未经增强的图像比较，使用训练视角；不代表独立测试或最终重建质量通过。</p><table><tr><th>指标</th><th>原尺寸</th><th>2倍上采样</th></tr>{cells}</table><p><a href="original.ply">原尺寸模型 PLY</a> · <a href="upsampled2x.ply">2倍模型 PLY</a> · <a href="cameras.json">相机参数</a> · <a href="comparison.json">完整测量</a></p>{''.join(panels)}<h2>相同新视角对照</h2><p class="note">每对左侧为原尺寸、右侧为2倍；这些视角没有实拍真值。</p><img class="comparison" src="orbit_contact_0.png"><img class="comparison" src="orbit_contact_1.png"></html>'''
    (out/'查看对照.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--subject',required=True,choices=['demo','sofa'])
    parser.add_argument('--native-input',action='store_true')
    args=parser.parse_args()
    main(args.subject,args.native_input)
