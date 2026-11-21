"""Native-resolution comparison and portable PLY/camera delivery for MSV2."""
from pathlib import Path
from types import SimpleNamespace
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import argparse
import ast
import copy
import importlib.util
import json
import shutil
import zipfile

import numpy as np
from PIL import Image,ImageDraw,ImageFont

RUN=Path('/opt/gs-slam-lab/msv2-demo-upsample3x-20261104')
spec=importlib.util.spec_from_file_location('msv2_experiment',RUN/'experiment.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
OUT=ex.RECORD
OLD=Path('/mnt/d/00/记录/人体上采样静止重建/demo_3x')
PLY_STEM='demo_msv2_3x_18000'
BUNDLE='demo_MSV2_3倍_18000步_点云与相机.zip'


def face_box(row,pad=True):
    box=np.array(row['face_box_xyxy']).reshape(2,2)
    d=np.maximum((box[1]-box[0])*.3,24) if pad else np.zeros(2)
    a=np.maximum(np.floor(box[0]-d).astype(int),0)
    b=np.minimum(np.ceil(box[1]+d).astype(int),row['wh'])
    return [int(a[0]),int(a[1]),int(b[0]),int(b[1])]


def board(paths,labels,dest,size=(320,650),title=''):
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    images=[]
    for p in paths:
        with Image.open(p) as im:a=im.convert('RGB')
        a.thumbnail(size,Image.Resampling.LANCZOS);images.append(a)
    cw=max(size[0],240);ch=max(i.height for i in images)+60
    canvas=Image.new('RGB',(cw*len(images),ch),'#eeeeee');d=ImageDraw.Draw(canvas)
    for j,(im,label) in enumerate(zip(images,labels)):
        d.text((j*cw+6,4),label,font=font,fill='black')
        d.text((j*cw+6,28),title,font=font,fill='black')
        canvas.paste(im,(j*cw+(cw-im.width)//2,58))
    canvas.save(dest)


def evaluation():
    torch=ex.runtime()
    from gaussian_renderer import render
    from scene import GaussianModel
    assert ex.read(RUN/'execution.json')['state']=='training_completed'
    final=RUN/'model/point_cloud/iteration_18000/point_cloud.ply'
    assert ex.read(final.parent/'ply-parity.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    # Use exactly the earlier foreground metric, independently of the MSV2 loss.
    source=ex.BASE/'source/src/shuolee_portrait/training.py'
    code=source.read_text(encoding='utf-8');tree=ast.parse(code)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='foreground_loss')
    namespace={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    foreground_loss=namespace['foreground_loss']
    rows=ex.read(RUN/'training-inputs.json')['records']
    native=ex.read(ex.BASE/'original/data/training-inputs.json')['records']
    old=ex.read(OLD/'comparison.json')
    model=GaussianModel(3);model.load_ply(str(final));model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');metrics={}
    with torch.no_grad():
        for domain,collection in [('enlarged',rows),('native',native)]:
            result=[]
            for i,row in enumerate(collection):
                cam=ex.Camera(row,i);pred=render(cam,model,pipe,bg)['render']
                gt=torch.tensor(cam.rgb,device='cuda',dtype=torch.float32).permute(2,0,1)/255.
                mask=torch.tensor(cam.mask,device='cuda',dtype=torch.float32)[None]/255.
                _,l1,ssim=foreground_loss(pred,gt,mask)
                x0,y0,x1,y1=face_box(row,False)
                fg=mask[0]>.99
                face=(pred-gt).abs().mean(0)[y0:y1,x0:x1][fg[y0:y1,x0:x1]]
                result.append(dict(frame=row['frame'],foreground_l1=float(l1),foreground_ssim=float(ssim),face_foreground_l1=float(face.mean())))
                folder=OUT/row['frame'];folder.mkdir(exist_ok=True)
                if domain=='enlarged':
                    ex.save_image(folder/'msv2.png',pred)
                    shutil.copy2(row['image'],folder/'input.png')
                    shutil.copy2(OLD/row['frame']/'upsampled3x_3x.png',folder/'previous_static.png')
                    for name in ['input','previous_static','msv2']:
                        with Image.open(folder/(name+'.png')) as im:im.crop(tuple(face_box(row))).save(folder/(name+'_face.png'))
                    board([folder/(x+'.png') for x in ['input','previous_static','msv2']],['3倍输入','上一版静止3倍','MSV2 3倍'],folder/'body_compare.png',title=row['frame'])
                    board([folder/(x+'_face.png') for x in ['input','previous_static','msv2']],['3倍输入','上一版静止3倍','MSV2 3倍'],folder/'face_compare.png',size=(240,300),title=row['frame'])
                elif i in ex.SELECTED:
                    ex.save_image(folder/'msv2_native.png',pred)
                if i%15==0 or i==58:print('EVALUATION',domain,i+1,'/59',flush=True)
                del cam,pred,gt,mask
            metrics[domain]=result
        orbit=ex.read(OLD/'cameras.json')['novel_review_cameras']
        (OUT/'orbit').mkdir()
        for i,row in enumerate(orbit):
            # Camera images are irrelevant for novel views; use a dummy canvas matching the camera.
            r=copy.deepcopy(row);dummy=OUT/'orbit'/f'{row["frame"]}_canvas.png'
            Image.new('RGB',tuple(row['wh'])).save(dummy)
            r.update(image=str(dummy),mask=str(dummy));cam=ex.Camera(r,i)
            ex.save_image(OUT/'orbit'/(row['frame']+'_msv2.png'),render(cam,model,pipe,bg)['render'])
            dummy.unlink()
            shutil.copy2(OLD/'orbit'/(row['frame']+'_upsampled3x.png'),OUT/'orbit'/(row['frame']+'_previous.png'))
            board([OUT/'orbit'/(row['frame']+s+'.png') for s in ['_previous','_msv2']],['上一版静止3倍','MSV2 3倍'],OUT/'orbit'/(row['frame']+'_compare.png'),size=(256,384),title=row['frame'])
    means={domain:{k:float(np.mean([r[k] for r in rows_])) for k in ['foreground_l1','foreground_ssim','face_foreground_l1']} for domain,rows_ in metrics.items()}
    comparison=dict(created=ex.now(),model='MSV2 demo 3x',frames=59,training_wh=[1428,2880],native_wh=[476,960],
        metrics=metrics,means=means,previous_static_means={d:old[k]['upsampled3x'] for d,k in [('enlarged','enlarged_mean'),('native','native_mean')]},
        foreground_metric_source_sha256=ex.sha(source),training_views_not_heldout=True,novel_views=12,novel_views_have_no_ground_truth=True,
        interpolation_not_learned_superresolution=True,quality_accepted=False,models_promoted=False)
    ex.save(OUT/'comparison.json',comparison)
    for name in ['plan.json','preflight.json','execution.json','config.json']:
        shutil.copy2(RUN/name,OUT/name)
    shutil.copy2(final,OUT/'demo_msv2_3x_18000_world.ply')
    shutil.copy2(final.parent/'ply-parity.json',OUT/'native-ply-parity.json')
    ex.save(OUT/'cameras_world.json',camera_records(rows))
    ex.save(OUT/'camera-input-mapping.json',dict(records=rows,convention='K uses pixel corners; corresponding pixel-index principal point is cx-.5,cy-.5',world='Original background SfM coordinates',static_model=True))
    export_upright(rows,model,pipe,bg)
    print(json.dumps(comparison['means']),flush=True)


def camera_records(rows):
    cameras=[]
    for i,r in enumerate(rows):
        w=np.array(r['w2c']);c=np.linalg.inv(w);k=np.array(r['K'])
        cameras.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],position=c[:3,3].tolist(),
          rotation=c[:3,:3].tolist(),fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],K=r['K'],pixelConvention='pixel_corners'))
    return cameras


def export_upright(rows,world_model,pipe,bg):
    import torch
    from plyfile import PlyData
    from scene import GaussianModel
    from gaussian_renderer import render
    from utils.sh_utils import eval_sh
    ply=PlyData.read(OUT/(PLY_STEM+'_world.ply'));v=ply['vertex'].data
    degree=world_model.max_sh_degree
    rest_count=(degree+1)**2-1
    assert degree in (0,1,2,3)
    assert len(v.dtype.names)==17+3*rest_count and all(np.isfinite(v[k]).all() for k in v.dtype.names)
    # Proper 180-degree X rotation. This makes the first camera's up direction +Y.
    # It is a presentation convention, not a gravity calibration.
    rotation=np.diag([1.,-1.,-1.]);transform=np.eye(4);transform[:3,:3]=rotation
    v['y']*=-1;v['z']*=-1
    q=np.stack([v[f'rot_{i}'] for i in range(4)],axis=1).copy()
    for i,a in enumerate([-q[:,1],q[:,0],-q[:,3],q[:,2]]):v[f'rot_{i}']=a
    signs=np.array([1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1],dtype=np.float32)
    for c in range(3):
        for j in range(rest_count):v[f'f_rest_{c*rest_count+j}']*=signs[j+1]
    upright=OUT/(PLY_STEM+'_yup.ply');ply.write(upright)
    rrows=copy.deepcopy(rows)
    for r in rrows:r['w2c']=(np.array(r['w2c'])@transform).tolist()
    ex.save(OUT/'cameras_yup.json',camera_records(rrows))
    n=GaussianModel(degree);n.load_ply(str(upright));n.active_sh_degree=degree
    # Check the complete covariance after PLY readback, independently of rendering.
    def quat_matrix(raw):
        raw=raw.astype(np.float64);raw/=np.linalg.norm(raw,axis=1,keepdims=True)
        w,x,y,z=raw.T
        return np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                         2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                         2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)
    original_rotation=quat_matrix(q)
    loaded_rotation=quat_matrix(n._rotation.detach().cpu().numpy())
    scale2=np.exp(2*np.stack([v[f'scale_{i}'] for i in range(3)],axis=1).astype(np.float64))
    original_cov=(original_rotation*scale2[:,None,:])@original_rotation.transpose(0,2,1)
    loaded_cov=(loaded_rotation*scale2[:,None,:])@loaded_rotation.transpose(0,2,1)
    expected_cov=rotation[None]@original_cov@rotation[None]
    covariance_error=float(np.max(np.abs(loaded_cov-expected_cov)))
    covariance_relative=float(np.max(np.linalg.norm(loaded_cov-expected_cov,axis=(1,2))/np.maximum(np.linalg.norm(expected_cov,axis=(1,2)),1e-15)))
    inverse_position_error=float(np.max(np.abs(n.get_xyz.detach().cpu().numpy()@rotation-world_model.get_xyz.detach().cpu().numpy())))
    assert covariance_relative<1e-6 and inverse_position_error<1e-6
    torch.manual_seed(42);directions=torch.randn((len(v),3),device='cuda');directions/=directions.norm(dim=1,keepdim=True)
    world_sh=world_model.get_features.transpose(1,2);new_sh=n.get_features.transpose(1,2)
    color_error=float((eval_sh(degree,world_sh,directions)-eval_sh(degree,new_sh,directions*torch.tensor([1.,-1.,-1.],device='cuda'))).abs().max())
    assert color_error<1e-5,color_error
    checks=[]
    with torch.no_grad():
        for i in [0,34,58]:
            a=ex.Camera(rows[i],i);b=ex.Camera(rrows[i],i)
            x=render(a,world_model,pipe,bg)['render'];y=render(b,n,pipe,bg)['render'];diff=(x-y).abs()
            checks.append(dict(frame=rows[i]['frame'],max_abs=float(diff.max()),mean_abs=float(diff.mean())))
    passed=all(x['max_abs']<.005 for x in checks)
    ex.save(OUT/'upright-parity.json',dict(passed=passed,rotation=transform.tolist(),SH_degree=degree,SH_color_max_abs=color_error,render_checks=checks,
        covariance_max_abs=covariance_error,covariance_max_relative=covariance_relative,inverse_position_max_abs=inverse_position_error,
        convention='First-camera vertical becomes +Y; not measured gravity',world_sha256=ex.sha(OUT/(PLY_STEM+'_world.ply')),upright_sha256=ex.sha(upright)))
    assert passed,checks
    package_scene()


def package_scene():
    upright=OUT/(PLY_STEM+'_yup.ply')
    with zipfile.ZipFile(OUT/BUNDLE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.write(upright,'point_cloud.ply');z.write(OUT/'cameras_yup.json','cameras.json')
        for f in ['upright-parity.json','native-ply-parity.json','camera-input-mapping.json','plan.json']:z.write(OUT/f,f)


def cameras():
    rows=ex.read(RUN/'training-inputs.json')['records']
    ex.save(OUT/'cameras_world.json',camera_records(rows))
    rrows=copy.deepcopy(rows);transform=np.array(ex.read(OUT/'upright-parity.json')['rotation'])
    for r in rrows:r['w2c']=(np.array(r['w2c'])@transform).tolist()
    ex.save(OUT/'cameras_yup.json',camera_records(rrows))
    package_scene()
    print('CAMERA_PACKAGE_UPDATED',flush=True)


def background():
    """Zero-update native-renderer check of black floaters exposed by gray background."""
    torch=ex.runtime()
    from scene import GaussianModel
    from gaussian_renderer import render
    free,total=torch.cuda.mem_get_info();assert free>3*1024**3
    torch.cuda.set_per_process_memory_fraction(2*1024**3/total)
    p=OUT/'background-diagnostic';p.mkdir(exist_ok=False)
    final=OUT/(PLY_STEM+'_world.ply');before=ex.sha(final)
    model=GaussianModel(3);model.load_ply(str(final));model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    rows=ex.read(ex.BASE/'original/data/training-inputs.json')['records'];checks=[]
    with torch.no_grad():
        for i in sorted(set(ex.SELECTED+[27])):
            r=rows[i];cam=ex.Camera(r,i)
            gray=render(cam,model,pipe,torch.full((3,),.5,device='cuda'))['render']
            alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(model.get_xyz))['render'][0]
            outside=torch.tensor(cam.mask,device='cuda')<3
            checks.append(dict(frame=r['frame'],outside_mask_pixels=int(outside.sum()),outside_alpha_mean=float(alpha[outside].mean()),outside_alpha_gt_01_fraction=float((alpha[outside]>.1).float().mean())))
            ex.save_image(p/(r['frame']+'_gray.png'),gray)
            ex.save_image(p/(r['frame']+'_alpha.png'),alpha[None].expand(3,-1,-1))
            shutil.copy2(r['mask'],p/(r['frame']+'_mask.png'))
            board([r['image'],p/(r['frame']+'_gray.png'),p/(r['frame']+'_alpha.png')],['原始输入','MSV2 灰底原生','MSV2 累计不透明度'],p/(r['frame']+'_compare.png'),size=(300,600),title=r['frame'])
    assert ex.sha(final)==before
    ex.save(p/'diagnostic.json',dict(created=ex.now(),optimizer_updates=0,model_sha256=before,model_unchanged=True,renderer='MSV2 native CUDA, SH3',checks=checks,
       interpretation='Alpha is integrated opacity across all depths, not a surface-closure metric; gray background exposes dark floaters hidden on black.'))
    print('ZERO_UPDATE_BACKGROUND_DIAGNOSTIC_COMPLETED',flush=True)


def publish():
    assert ex.read(OUT/'upright-parity.json')['passed']
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    for kind,size in [('body',(300,204)),('face',(300,150))]:
        for begin in range(1,60,12):
            ids=list(range(begin,min(begin+12,60)));cw,ch=size[0]+10,size[1]+30
            canvas=Image.new('RGB',(3*cw,((len(ids)+2)//3)*ch),'#eeeeee');d=ImageDraw.Draw(canvas)
            for j,i in enumerate(ids):
                name=f'frame_{i:06d}';x=(j%3)*cw;y=(j//3)*ch
                with Image.open(OUT/name/(kind+'_compare.png')) as im:
                    im=im.convert('RGB');im.thumbnail(size,Image.Resampling.LANCZOS)
                d.text((x+4,y+2),name,font=font,fill='black');canvas.paste(im,(x,y+27))
            canvas.save(OUT/f'all_{kind}_{(begin-1)//12+1:02d}.png')
    c=ex.read(OUT/'comparison.json');e=ex.read(OUT/'execution.json')
    findings=ex.read(OUT/'review-findings.json') if (OUT/'review-findings.json').exists() else {}
    findings_html=''.join('<p>'+escape(t)+'</p>' for t in findings.get('summary_paragraphs',['本地训练和点云回读已完成，逐帧观察记录正在整理。']))
    if findings:
        findings_html+='<p><a href="review-findings.json">完整审阅记录</a> · <a href="viewer-import-verification.json">查看器导入结果</a></p><h3>灰背景复核：原图／原生渲染／累计不透明度</h3><img src="background-diagnostic/frame_000001_compare.png"><p>原生CUDA渲染也出现黑色放射伪影；不是仅在网页查看器出现。累计不透明度不代表表面闭合。</p>'
    table=[]
    for domain,label in [('enlarged','1428×2880 放大图'),('native','476×960 原图')]:
        for metric,title in [('foreground_l1','人物颜色误差 ↓'),('face_foreground_l1','脸部颜色误差 ↓'),('foreground_ssim','人物 SSIM ↑')]:
            old=c['previous_static_means'][domain][metric];new=c['means'][domain][metric]
            table.append(f'<tr><td>{label}</td><td>{title}</td><td>{old:.5f}</td><td>{new:.5f}</td></tr>')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>demo · 3倍输入 · MSV2</title>',
      '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;background:#f6f6f3;color:#222;font:16px/1.65 system-ui}section{background:white;padding:22px;margin:20px 0;border-radius:12px}h1{font-size:30px}h2{font-size:22px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ddd;text-align:left;padding:8px}a{color:#176684}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}</style>',
      '<h1>demo · 宽高3倍输入 · Mini-Splatting2</h1>',
      f'<p>59帧 · 476×960 → 1428×2880 · 官方 MSV2 fast 18,000步时钟 · 实际17,999次参数更新 · {e["points"]:,}个高斯</p>',
      '<section><h2>点云与相机</h2><p><a download href="demo_MSV2_3倍_18000步_点云与相机.zip">下载：通用PLY＋59台同坐标相机ZIP</a></p><p><a download href="demo_msv2_3x_18000_yup.ply">+Y向上PLY</a> · <a href="cameras_yup.json">配套相机JSON</a> · <a href="demo_msv2_3x_18000_world.ply">训练世界坐标PLY</a> · <a href="cameras_world.json">训练相机</a></p><p>相机内参对应1428×2880人物裁片，包含主点；这是共享静态点云。+Y来自参考相机方向，不表示重力标定通过。</p></section>',
      '<section id="findings"><h2>效果核对</h2>'+findings_html+'</section>',
      '<section><h2>相同相机、相同像素下比较</h2><p>左侧上一版是沙发静止训练器，右侧是MSV2。两者使用相同的59帧3倍插值输入与未训练的轮廓初始点；MSV2沿用官方合成图L1/SSIM、增密/深度重建/简化，训练目标不同。这是路线比较，不能把变化只归因于某一个模块。指标来自训练视角；环绕视角没有实拍真值。</p><table><tr><th>渲染与参考尺寸</th><th>指标</th><th>上一版静止3倍</th><th>MSV2 3倍</th></tr>'+''.join(table)+'</table></section>',
      '<section><h2>输入与策略</h2><p>沿用Lanczos宽高各3倍上采样，没有新增实拍细节。完整K同步缩放，背景估计外参不变。18k fast时序保持；关闭半分辨率预热，全程使用3倍图。MSV2深度值按完整K重建到原像素射线，并排除无命中的零点。</p><p><a href="plan.json">配置与来源</a> · <a href="preflight.json">本地输入检查</a> · <a href="native-ply-parity.json">原生PLY回读</a> · <a href="upright-parity.json">旋转、SH与相机回读</a> · <a href="comparison.json">逐帧指标</a></p></section>']
    for i in [1,12,24,35,47,59]:
        name=f'frame_{i:06d}'
        parts.append(f'<section><h2>{name} · 原图／上一版／MSV2</h2><a href="{name}/face_compare.png"><img src="{name}/face_compare.png" loading="lazy"></a><a href="{name}/body_compare.png"><img src="{name}/body_compare.png" loading="lazy"></a><p><a href="{name}/input.png">3倍输入原图</a> · <a href="{name}/previous_static.png">上一版完整渲染</a> · <a href="{name}/msv2.png">MSV2完整渲染</a></p></section>')
    parts.append('<section><h2>全部59帧</h2><p>每项均为输入、上一版、MSV2三列。点击查看完整对照。</p><div class="grid">')
    for i in range(1,60):
        name=f'frame_{i:06d}';parts.append(f'<div><a href="{name}/body_compare.png">{name} 全身</a> · <a href="{name}/face_compare.png">脸部</a><a href="{name}/body_compare.png"><img src="{name}/body_compare.png" loading="lazy"></a></div>')
    parts.append('</div></section><section><h2>12个相同环绕视角</h2><div class="grid">')
    for angle in range(0,360,30):
        name=f'orbit_{angle:03d}';parts.append(f'<a href="orbit/{name}_compare.png"><img src="orbit/{name}_compare.png" loading="lazy"></a>')
    parts.append('</div></section></html>')
    page=OUT/'查看对照.html';page.write_text(''.join(parts),encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src') and v)
    p=Links();p.feed(page.read_text(encoding='utf-8'))
    for link in p.links:
        u=urlsplit(link)
        if not u.scheme and u.path:assert (OUT/unquote(u.path)).exists(),link
    images=[]
    for path in OUT.rglob('*.png'):
        with Image.open(path) as im:im.verify()
        images.append(dict(path=path.relative_to(OUT).as_posix(),sha256=ex.sha(path)))
    ex.save(OUT/'archive-verification.json',dict(passed=True,images=images,html_links=len(p.links),points=e['points'],quality_accepted=False))
    print('PUBLISHED',page,len(images),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','publish','cameras','background']);a=p.parse_args()
    {'evaluate':evaluation,'publish':publish,'cameras':cameras,'background':background}[a.action]()
