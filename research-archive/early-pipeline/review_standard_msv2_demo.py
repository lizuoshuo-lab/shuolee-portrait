"""Compare the standard full-frame MSV2 runs and export portable Gaussian scenes."""
from pathlib import Path
from types import SimpleNamespace
from html import escape
import argparse,copy,importlib.util,json,math,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

SCRIPT=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/standard_msv2_demo.py')
spec=importlib.util.spec_from_file_location('standard_experiment',SCRIPT)
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
OUT=ex.RECORD
LABELS={1:'原始完整图',2:'Lanczos 插值两倍',4:'SwinIR 超分四倍'}

def camera_records(rows):
    result=[]
    for i,r in enumerate(rows):
        w=np.asarray(r['w2c']);c=np.linalg.inv(w);k=np.asarray(r['K'])
        result.append(dict(id=i,img_name=r['frame'],width=r['wh'][0],height=r['wh'][1],position=c[:3,3].tolist(),rotation=c[:3,:3].tolist(),
            fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],K=r['K'],pixelConvention='pixel_corners',split=r['split']))
    return result

def regions(rows):
    # Prior detections only locate display/evaluation rectangles. No mask is read.
    prior=ex.read('/opt/gs-slam-lab/static-person-upsampling-20261102/demo-static-input/data/training-inputs.json')['records']
    ids=np.asarray([r['source_frame_index'] for r in prior]);result={}
    for r in rows:
        frame=r['source_frame_matches'][0]['source_index'];boxes={}
        for name,key,padding in [('body','initial_box_xyxy',.08),('face','face_box_xyxy',.45)]:
            coords=np.asarray([p[key] for p in prior]);box=np.asarray([np.interp(frame,ids,coords[:,j]) for j in range(4)]).reshape(2,2)
            pad=np.maximum((box[1]-box[0])*padding,8);a=np.maximum(np.floor(box[0]-pad),0).astype(int);b=np.minimum(np.ceil(box[1]+pad),[2704,1520]).astype(int)
            boxes[name]=[int(a[0]),int(a[1]),int(b[0]),int(b[1])]
        result[r['frame']]=boxes
    return result

def evaluate(scale):
    torch=ex.runtime();run=ex.ROOT/ex.NAMES[scale];out=OUT/ex.NAMES[scale];out.mkdir(parents=True,exist_ok=True)
    execution=ex.read(run/'execution.json');assert execution['state']=='training_completed' and ex.read(run/'ply-parity.json')['passed']
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from fused_ssim import fused_ssim
    model=GaussianModel(3);model.load_ply(str(run/'model/point_cloud/iteration_18000/point_cloud.ply'));model.active_sh_degree=3
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    rows=ex.read(ex.ROOT/'original/training-inputs.json')['records'];boxes=regions(rows);metrics=[]
    with torch.no_grad():
        for i,r in enumerate(rows):
            cam=ex.LazyCamera(r,i);pred=render(cam,model,pipe,bg)['render'].clamp(0,1);gt=cam.original_image;diff=pred-gt
            entry=dict(frame=r['frame'],split=r['split'],l1=float(diff.abs().mean()),psnr=float(-10*torch.log10(diff.square().mean().clamp_min(1e-12))),ssim=float(fused_ssim(pred[None],gt[None])))
            folder=OUT/'frames'/r['frame'];folder.mkdir(parents=True,exist_ok=True)
            dest=folder/(ex.NAMES[scale]+'.png');ex.save_image(dest,pred)
            if not (folder/'input.png').exists():shutil.copy2(r['native_image'],folder/'input.png')
            for key,box in boxes[r['frame']].items():
                x0,y0,x1,y1=box;entry[key+'_l1']=float(diff[:,y0:y1,x0:x1].abs().mean())
                with Image.open(dest) as im:im.crop(box).save(folder/(ex.NAMES[scale]+'_'+key+'.png'))
                if not (folder/('input_'+key+'.png')).exists():
                    with Image.open(r['native_image']) as im:im.crop(box).save(folder/('input_'+key+'.png'))
            metrics.append(entry)
            if i%20==0:print('STANDARD_REVIEW',scale,i+1,len(rows),flush=True)
    means={}
    for split in ('train','test'):
        subset=[r for r in metrics if r['split']==split]
        means[split]={key:float(np.mean([r[key] for r in subset])) for key in ('l1','psnr','ssim','body_l1','face_l1')}
    ex.save(out/'metrics.json',dict(scale=scale,execution=execution,comparison_wh=[2704,1520],metrics=metrics,means=means,
        no_mask=True,boxes=boxes,region_source='Interpolated display/evaluation rectangles from prior RTMW detections, never used in training or camera recovery',quality_accepted=False))
    for name in ('preflight.json','execution.json','resolved-options.json','ply-parity.json','progress.jsonl'):shutil.copy2(run/name,out/name)
    print('STANDARD_EVALUATION_DONE',scale,json.dumps(means),flush=True)

def export(scale):
    torch=ex.runtime();run=ex.ROOT/ex.NAMES[scale];out=OUT/ex.NAMES[scale];out.mkdir(parents=True,exist_ok=True)
    assert ex.read(run/'execution.json')['state']=='training_completed' and ex.read(run/'ply-parity.json')['passed']
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.sh_utils import eval_sh
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation
    rows=ex.read(run/'training-inputs.json')['records'];stem='demo_'+ex.NAMES[scale]+'_msv2_18000'
    world=out/(stem+'_world.ply');shutil.copy2(run/'model/point_cloud/iteration_18000/point_cloud.ply',world)
    ex.save(out/'cameras_world.json',camera_records(rows))
    world_model=GaussianModel(3);world_model.load_ply(str(world));world_model.active_sh_degree=3
    ply=PlyData.read(world);v=ply['vertex'].data
    assert len(v.dtype.names)==62 and all(np.isfinite(v[k]).all() for k in v.dtype.names)
    # Exact axis sign change avoids roundoff in arbitrary-angle world rotations.
    # COLMAP's initialization camera uses +Y down; its up becomes viewer +Y.
    rotation=np.diag([1.,-1.,-1.])
    assert abs(np.linalg.det(rotation)-1)<1e-7
    transform=np.eye(4);transform[:3,:3]=rotation
    xyz=world_model.get_xyz.detach().cpu().numpy();rotated=xyz@rotation.T
    for j,key in enumerate(('x','y','z')):v[key]=rotated[:,j]
    raw_quat=np.stack([v[f'rot_{j}'] for j in range(4)],1).copy()
    q=raw_quat.astype(np.float64);q/=np.linalg.norm(q,axis=1,keepdims=True)
    new_quat=np.stack([-raw_quat[:,1],raw_quat[:,0],-raw_quat[:,3],raw_quat[:,2]],axis=1)
    for j in range(4):v[f'rot_{j}']=new_quat[:,j]
    # Analytic signs for the 180-degree X rotation, with an independent basis check.
    rng=np.random.default_rng(9327);directions=rng.normal(size=(96,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    def basis(d):
        tensor=torch.as_tensor(d,dtype=torch.float64,device='cuda');columns=[]
        for j in range(16):
            coefficients=torch.zeros((len(d),1,16),dtype=torch.float64,device='cuda');coefficients[:,:,j]=1
            columns.append(eval_sh(3,coefficients,tensor).cpu().numpy()[:,0])
        return np.stack(columns,1)
    signs=np.asarray([1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1],dtype=np.float32)
    a=basis(directions@rotation.T);b=basis(directions);change=np.diag(signs)
    assert np.max(np.abs(a@change-b))<1e-12
    features=world_model.get_features.detach().cpu().numpy().transpose(0,2,1)
    new_features=features*signs[None,None,:]
    for channel in range(3):
        v[f'f_dc_{channel}']=new_features[:,channel,0]
        for j in range(15):v[f'f_rest_{channel*15+j}']=new_features[:,channel,j+1]
    yup=out/(stem+'_yup.ply');ply.write(yup)
    model=GaussianModel(3);model.load_ply(str(yup));model.active_sh_degree=3
    rrows=copy.deepcopy(rows)
    for r in rrows:r['w2c']=(np.asarray(r['w2c'])@np.linalg.inv(transform)).tolist()
    ex.save(out/'cameras_yup.json',camera_records(rrows))
    # Independent full covariance, SH and render validation after PLY readback.
    old_matrix=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();new_matrix=Rotation.from_quat(model._rotation.detach().cpu().numpy()[:,[1,2,3,0]]).as_matrix()
    scales2=np.exp(2*world_model._scaling.detach().cpu().numpy().astype(np.float64))
    old_cov=(old_matrix*scales2[:,None,:])@old_matrix.transpose(0,2,1);new_cov=(new_matrix*scales2[:,None,:])@new_matrix.transpose(0,2,1)
    expected=rotation[None]@old_cov@rotation.T[None]
    covariance_relative=float(np.max(np.linalg.norm(new_cov-expected,axis=(1,2))/np.maximum(np.linalg.norm(expected,axis=(1,2)),1e-15)))
    position_error=float(np.max(np.abs(model.get_xyz.detach().cpu().numpy()@rotation-xyz)))
    # Bound inverse coordinate error by float32 PLY precision at actual world scale.
    position_tolerance=float(8*np.finfo(np.float32).eps*np.max(np.abs(xyz)))
    testdirs=rng.normal(size=(len(v),3));testdirs/=np.linalg.norm(testdirs,axis=1,keepdims=True)
    td=torch.as_tensor(testdirs,dtype=torch.float32,device='cuda');tr=torch.as_tensor(testdirs@rotation.T,dtype=torch.float32,device='cuda')
    sh_error=float((eval_sh(3,world_model.get_features.transpose(1,2),td)-eval_sh(3,model.get_features.transpose(1,2),tr)).abs().max())
    assert covariance_relative<2e-6 and position_error<position_tolerance and sh_error<1e-4
    native=ex.read(ex.ROOT/'original/training-inputs.json')['records'];pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda');checks=[]
    with torch.no_grad():
        for i in (0,len(native)//2,len(native)-1):
            original=native[i];rotated_row=copy.deepcopy(original);rotated_row['w2c']=(np.asarray(original['w2c'])@np.linalg.inv(transform)).tolist()
            x=render(ex.LazyCamera(original,i),world_model,pipe,bg)['render'];y=render(ex.LazyCamera(rotated_row,i),model,pipe,bg)['render'];d=(x-y).abs()
            checks.append(dict(frame=original['frame'],max_abs=float(d.max()),mean_abs=float(d.mean())));del x,y,d
    passed=all(c['max_abs']<.005 for c in checks);assert passed,checks
    ex.save(out/'upright-parity.json',dict(passed=passed,transform=transform.tolist(),convention='COLMAP initialization-camera up mapped to +Y; exact 180-degree X rotation; not measured gravity',
        covariance_relative=covariance_relative,inverse_position_error=position_error,position_tolerance=position_tolerance,SH_error=sh_error,render_checks=checks))
    ex.save(out/'camera-input-mapping.json',dict(records=rows,mask_used=False,full_frame=True,coordinate_convention='COLMAP .5-based pixel centers',training_resolution=rows[0]['wh']))
    shutil.copy2(run/'ply-parity.json',out/'native-ply-parity.json');shutil.copy2(ex.ROOT/'plan.json',out/'plan.json')
    bundle=out/(stem+'_点云与相机.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        z.write(yup,'point_cloud.ply');z.write(out/'cameras_yup.json','cameras.json')
        for name in ('upright-parity.json','native-ply-parity.json','camera-input-mapping.json','plan.json'):z.write(out/name,name)
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None and len(json.loads(z.read('cameras.json')))==len(rows)
    ex.save(out/'delivery.json',dict(bundle=bundle.name,points=len(v),cameras=len(rows),sha256=ex.sha(bundle),parity_passed=True,quality_accepted=False))
    print('STANDARD_EXPORT_DONE',scale,len(v),len(rows),bundle,flush=True)

def publish():
    rows=ex.read(ex.ROOT/'original/training-inputs.json')['records'];tables={s:ex.read(OUT/ex.NAMES[s]/'metrics.json') for s in (1,2,4)}
    deliveries={s:ex.read(OUT/ex.NAMES[s]/'delivery.json') for s in (1,2,4)}
    frame_count=len(rows)
    if ex.EXPECTED_FRAMES is not None:assert frame_count==ex.EXPECTED_FRAMES
    assert all(deliveries[s]['cameras']==frame_count for s in (1,2,4))
    assert all(tables[s]['execution']['state']=='training_completed' for s in (1,2,4))
    assert all(deliveries[s]['parity_passed'] for s in (1,2,4))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plot_font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
    for ax,s in zip(axes,(1,2,4)):
        curve=[json.loads(line) for line in (ex.ROOT/ex.NAMES[s]/'progress.jsonl').read_text().splitlines()]
        ax.plot([r['iteration'] for r in curve],[r['points']/10000 for r in curve],color={1:'#27648d',2:'#8e682d',4:'#34876a'}[s])
        for step in (2000,3000,8000):ax.axvline(step,color='#aaa',linewidth=.7,linestyle='--')
        ax.set_title(LABELS[s]+f' · {frame_count}帧 · 最终 {deliveries[s]["points"]:,} 点',fontproperties=plot_font)
        ax.set_xlabel('训练步数',fontproperties=plot_font);ax.set_ylabel('高斯数（万）',fontproperties=plot_font);ax.grid(alpha=.16)
    fig.savefig(OUT/'高斯点数变化.png',dpi=160);plt.close(fig)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',19)
    for r in rows:
        folder=OUT/'frames'/r['frame']
        for region,size in [('full',(640,360)),('body',(260,580)),('face',(260,320))]:
            canvas=Image.new('RGB',(size[0]*4,size[1]+60),'#ededed');draw=ImageDraw.Draw(canvas)
            names=['input.png']+[ex.NAMES[s]+'.png' for s in (1,2,4)] if region=='full' else ['input_'+region+'.png']+[ex.NAMES[s]+'_'+region+'.png' for s in (1,2,4)]
            for j,name in enumerate(names):
                with Image.open(folder/name) as im:
                    im=im.convert('RGB');im.thumbnail(size,Image.Resampling.LANCZOS)
                    canvas.paste(im,(j*size[0]+(size[0]-im.width)//2,60+(size[1]-im.height)//2))
                draw.text((j*size[0]+8,5),['视频原图',LABELS[1],LABELS[2],LABELS[4]][j],font=font,fill='black')
                draw.text((j*size[0]+8,31),r['frame']+' / '+('留出' if r['split']=='test' else '训练'),font=font,fill='black')
            canvas.save(folder/(region+'_compare.jpg'),quality=94)
    data=dict(created=ex.now(),frame_count=frame_count,frames=rows,means={ex.NAMES[s]:tables[s]['means'] for s in (1,2,4)},deliveries={ex.NAMES[s]:deliveries[s] for s in (1,2,4)},training_completed=True,quality_accepted=False,config=ex.read(ex.ROOT/'plan.json')['recipe'])
    ex.save(OUT/'comparison.json',data)
    options=''.join('<option value="'+r['frame']+'">'+r['frame']+' · '+('留出' if r['split']=='test' else '训练')+'</option>' for r in rows)
    body=''
    for s in (1,2,4):
        n=ex.NAMES[s];metric=tables[s]['means']['test'];delivery=deliveries[s]
        viewer=f'http://127.0.0.1:8870/?model={escape(ex.model_id(s))}&amp;sh=3&amp;view=0'
        body+=f'<tr><td>{LABELS[s]}</td><td>{frame_count}</td><td>{2704*s} × {1520*s}</td><td>{delivery["points"]:,}</td><td>{metric["psnr"]:.3f}</td><td>{metric["ssim"]:.4f}</td><td><a href="{n}/{escape(delivery["bundle"])}">点云与相机</a> · <a href="{viewer}" target="_blank">本机查看器</a></td></tr>'
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MSV2 完整图倍率对照 · {{FRAME_COUNT}}帧</title><style>
body{margin:30px auto;max-width:1400px;padding:0 22px;font:16px/1.65 system-ui;background:#f5f6f7;color:#20242a}h1{font-size:28px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.panel{background:white;padding:18px;margin:18px 0;border-radius:12px}img{width:100%;height:auto}select{padding:8px;font-size:16px}a{color:#2266aa}.hint{color:#555}</style>
<h1>MSV2 · {{FRAME_COUNT}}帧 · 原始完整图 / 插值两倍 / 超分四倍</h1><p>背景一起训练，不使用 mask。原视频完整画面直接作为倍率起点。三版统一保留用户指定的 outdoor，其余使用之前复现的 config/fast，18,000 步，种子 0。官方前 3,000 步半尺寸预热保留。</p>
<table><tr><th>版本</th><th>输入帧数</th><th>目标训练尺寸</th><th>最终高斯点数</th><th>留出 PSNR</th><th>留出 SSIM</th><th>通用文件</th></tr>'''+body+'''</table>
<p class="hint">指标统一在原始完整画面 2704×1520 下计算。每 8 帧留出 1 帧，不用于高斯颜色训练；COLMAP 使用全部配准图像。训练完成及文件一致性通过不等于新视角几何质量通过。</p><div class="panel"><img src="高斯点数变化.png" alt="三版训练的实际高斯数量"><p class="hint">虚线依次为第 2,000 步深度重初始化、第 3,000 步及第 8,000 步官方简化。没有额外固定点数上限或训练后删点。</p></div>
<div class="panel"><label>帧 <select id="frame">'''+options+'''</select></label> <label>区域 <select id="region"><option value="full">完整画面</option><option value="body">人物区域</option><option value="face">面部区域</option></select></label><p><a id="direct" target="_blank"><img id="compare"></a></p><div id="links"></div></div>
<p><a href="标准配置核查.json">源码与配置来源</a> · <a href="comparison.json">完整指标与结果</a></p>
<script>const f=document.querySelector('#frame'),r=document.querySelector('#region');function update(){const base='frames/'+f.value+'/';const p=base+r.value+'_compare.jpg';document.querySelector('#compare').src=p;document.querySelector('#direct').href=p;document.querySelector('#links').innerHTML=['original','interpolation2x','swinir4x'].map((n,i)=>'<a target="_blank" href="'+base+n+'.png">'+['原始版渲染','两倍版渲染','四倍版渲染'][i]+'</a>').join(' · ')}f.onchange=r.onchange=update;update();</script></html>'''
    html=html.replace('{{FRAME_COUNT}}',str(frame_count))
    (OUT/'查看完整图倍率对照.html').write_text(html,encoding='utf-8')
    plan=ex.read(ex.ROOT/'plan.json')
    plan.update(state='completed',training_completed=True,quality_accepted=False,finished=ex.now())
    ex.save(ex.ROOT/'plan.json',plan)
    ex.save(OUT/'标准配置核查.json',plan)
    print('STANDARD_COMPARISON_READY',OUT,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','export','publish']);p.add_argument('--scale',type=int,choices=[1,2,4],default=1);a=p.parse_args()
    if a.action=='evaluate':evaluate(a.scale)
    elif a.action=='export':export(a.scale)
    else:publish()
if __name__=='__main__':main()
