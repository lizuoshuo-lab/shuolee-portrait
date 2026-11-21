"""Official SwinIR-L real-world x4 inputs for one fresh, bounded MSV2 demo."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,copy,hashlib,importlib.util,json,shutil,sys,time
import numpy as np
from PIL import Image

RUN=Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')
PREVIOUS=Path('/opt/gs-slam-lab/msv2-demo-upsample3x-20261104')
BASE=Path('/opt/gs-slam-lab/static-person-upsampling-20261102/demo')
REPO=Path('/mnt/d/00/工作/SwinIR')
WEIGHT=Path('/opt/gs-slam-lab/model-cache/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
WEIGHT_SHA='99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a'
RECORD=Path('/mnt/d/00/记录/2026-11-04_SwinIR4倍_MSV2/demo')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)


def create():
    assert sha(WEIGHT)==WEIGHT_SHA
    RUN.mkdir(exist_ok=False)
    shutil.copy2(__file__,RUN/'swinir-inputs.py')
    shutil.copytree(REPO,RUN/'swinir-source',ignore=shutil.ignore_patterns('.git','model_zoo','testsets','figs','__pycache__','*.pyc'))
    shutil.copytree(PREVIOUS/'source',RUN/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(PREVIOUS/'source-manifest.json',RUN/'source-manifest.json')
    shutil.copy2(PREVIOUS/'initial-points.npz',RUN/'initial-points.npz')
    shutil.copy2(PREVIOUS/'config.json',RUN/'config.json')
    text=(PREVIOUS/'experiment.py').read_text(encoding='utf-8')
    a=text.index('def prepare():');b=text.index('def read_fast():')
    text=text[:a]+text[b:]
    text=text.replace(str(PREVIOUS),str(RUN)).replace('/mnt/d/00/记录/2026-11-04_MSV2_人体3倍上采样/demo',str(RECORD))
    text=text.replace("tuple(im.shape)==(3,2880,1428)","tuple(im.shape)==(3,c.height,c.width)")
    text=text.replace('render_wh=[1428,2880]','render_wh=[c.width,c.height]')
    text=text.replace("choices=['prepare','preflight','train']","choices=['preflight','train']")
    compile(text,'experiment.py','exec');(RUN/'experiment.py').write_text(text,encoding='utf-8')
    native=read(BASE/'original/data/training-inputs.json')['records'];assert len(native)==59
    for r in native:
        assert r['wh']==[476,960] and sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
    save(RUN/'native-inputs.json',dict(records=native))
    plan=read(PREVIOUS/'plan.json')
    plan.update(created=now(),method='SwinIR-L real-world x4 + Mini-Splatting2 / msv2',
       inputs='All 59 original 476x960 crops passed through official pretrained SwinIR-L real_sr x4; bilinear masks',
       training_wh=[1904,3840],pixel_count_scale=16,learned_superresolution=True,
       camera_policy='Original background w2c retained; full pixel-corner K first two rows multiplied by 4; integer-center principal point recomputed as corner minus 0.5',
       initialization='Same unoptimized 200000-point silhouette-hull xyz and native source RGB; all Gaussian parameters and optimizer initialized fresh',
       initial_points_sha256=sha(RUN/'initial-points.npz'),
       comparability='Same MSV2 fast schedule/loss/seed/initial points as 3x Lanczos run. SR method and input resolution both change; this is not a resolution-matched SR-only ablation.',
       known_baseline_limitations='3x MSV2 contains dark opaque floaters and incomplete back views. Keep gray-background and all-view review; SR inputs do not establish a geometry or opacity fix.',
       swinir=dict(upstream='https://github.com/JingyunLiang/SwinIR',commit='6545850fbf8df298df73d81f3e8cba638787c8bd',
          weight_file=WEIGHT.name,weight_sha256=WEIGHT_SHA,weight_source='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'+WEIGHT.name,
          task='real_sr',scale=4,large_model=True,parameter_key='params_ema',tile=256,tile_overlap=32,precision='FP32, TF32 disabled',
          learned_details_are_predictions=True,weights_trained_for_this_subject=False),
       quality_accepted=False,production_deployment_changed=False)
    plan['adapter_changes']=[s.replace('3x','4x') for s in plan['adapter_changes']]
    plan.pop('depth_input_diagnostic',None)
    save(RUN/'plan.json',plan)
    save(RUN/'swinir-source-manifest.json',{p.relative_to(RUN/'swinir-source').as_posix():sha(p) for p in (RUN/'swinir-source').rglob('*') if p.is_file()})
    print('SWINIR_MSV2_PREPARED',RUN,flush=True)


def superresolve(first_only=False):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free,total=torch.cuda.mem_get_info();assert free>7*1024**3
    torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
    assert sha(WEIGHT)==WEIGHT_SHA
    for f,h in read(RUN/'swinir-source-manifest.json').items():assert sha(RUN/'swinir-source'/f)==h
    sys.path.insert(0,str(RUN/'swinir-source'));import main_test_swinir as official
    args=SimpleNamespace(task='real_sr',scale=4,large_model=True,model_path=str(WEIGHT),tile=256,tile_overlap=32)
    model=official.define_model(args).eval().cuda()
    state_path=RUN/'superresolution.json'
    state=read(state_path) if state_path.exists() else dict(state='running',started=now(),records=[],source_sha256=WEIGHT_SHA)
    rows=read(RUN/'native-inputs.json')['records'];done={r['frame']:r for r in state['records']}
    start=time.monotonic()
    with torch.inference_mode():
        for r in rows[:1] if first_only else rows:
            image=RUN/'data/images'/r['name'];mask=RUN/'data/masks'/r['name']
            image.parent.mkdir(parents=True,exist_ok=True);mask.parent.mkdir(parents=True,exist_ok=True)
            if r['frame'] in done:
                assert sha(image)==done[r['frame']]['output_sha256'] and sha(mask)==done[r['frame']]['mask_sha256'];continue
            assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
            with Image.open(r['image']) as im:rgb=np.array(im.convert('RGB'))
            x=torch.tensor(rgb,device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255.
            h,w=rgb.shape[:2];hp=(h//8+1)*8-h;wp=(w//8+1)*8-w
            x=torch.cat([x,torch.flip(x,[2])],2)[:,:,:h+hp,:]
            x=torch.cat([x,torch.flip(x,[3])],3)[:,:,:,:w+wp]
            tick=time.monotonic();pred=official.test(x,model,args,8)[...,:h*4,:w*4]
            assert pred.shape==(1,3,3840,1904) and torch.isfinite(pred).all()
            out=np.rint(pred[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(out).save(image)
            with Image.open(r['mask']) as im:im.convert('L').resize((1904,3840),Image.Resampling.BILINEAR).save(mask)
            # Same-size interpolation is a display control, not a learned SR target.
            with Image.open(r['image']) as im:linear=np.array(im.convert('RGB').resize((1904,3840),Image.Resampling.LANCZOS))
            delta=float(np.abs(out.astype(np.float32)-linear.astype(np.float32)).mean()/255.)
            assert delta>1e-5,'Output unexpectedly identical to interpolation'
            item=dict(frame=r['frame'],native_sha256=r['source_sha256'],output_sha256=sha(image),mask_sha256=sha(mask),
                      output_wh=[1904,3840],l1_against_lanczos4x=delta,seconds=time.monotonic()-tick)
            state['records'].append(item);state.update(updated=now(),processed=len(state['records']),state='probe_completed' if first_only else 'running',peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            save(state_path,state);print('SWINIR',len(state['records']),'/59',json.dumps(item),flush=True)
            if first_only:
                preview=RUN/'sr-probe';preview.mkdir(exist_ok=True)
                Image.fromarray(linear).save(preview/'lanczos4x.png');shutil.copy2(image,preview/'swinir4x.png');shutil.copy2(r['image'],preview/'original.png')
                box=np.array(r['face_box_xyxy']).reshape(2,2)*4;pad=np.maximum((box[1]-box[0])*.3,32)
                box=np.r_[np.floor(box[0]-pad),np.ceil(box[1]+pad)].astype(int)
                for name in ['lanczos4x','swinir4x']:
                    with Image.open(preview/(name+'.png')) as im:im.crop(tuple(box)).save(preview/(name+'_face.png'))
            del x,pred
    if not first_only:
        assert len(state['records'])==59;state.update(state='completed',finished=now(),resume_session_seconds=time.monotonic()-start);save(state_path,state)
        stage_inputs()
    print('SWINIR_PROBE_READY' if first_only else 'SWINIR_ALL_59_COMPLETED',flush=True)


def stage_inputs():
    state=read(RUN/'superresolution.json');assert state['state']=='completed'
    manifest={r['frame']:r for r in state['records']};rows=copy.deepcopy(read(RUN/'native-inputs.json')['records'])
    for r in rows:
        m=manifest[r['frame']];r.update(native_image=r['image'],native_mask=r['mask'],native_image_sha256=r['source_sha256'])
        r['image']=str(RUN/'data/images'/r['name']);r['mask']=str(RUN/'data/masks'/r['name'])
        r['source_sha256']=m['output_sha256'];r['mask_sha256']=m['mask_sha256'];r['wh']=[1904,3840]
        k=np.diag([4.,4.,1.])@np.array(r['K']);ki=k.copy();ki[:2,2]-=.5
        r['K']=k.tolist();r['K_integer_centers']=ki.tolist()
        for key in ['face_box_xyxy','initial_box_xyxy','other_person_boxes_xyxy']:
            if key in r:r[key]=(np.array(r[key])*4).tolist()
        r['upsampling_scale']=4;r['upsampling_method']='SwinIR-L real_sr x4 official params_ema'
        assert sha(r['image'])==r['source_sha256'] and sha(r['mask'])==r['mask_sha256']
    save(RUN/'training-inputs.json',dict(records=rows,coordinate_contract='K in pixel-corner coordinates; same original background w2c',learned_superresolution=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','probe','superresolve']);a=p.parse_args()
    if a.action=='create':create()
    else:superresolve(first_only=a.action=='probe')
