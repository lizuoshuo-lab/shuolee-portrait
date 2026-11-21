"""Bounded native DUSt3R forward on fixed real pairs; no camera/body fitting."""
from pathlib import Path
import argparse, ast, datetime as dt, hashlib, json, os, subprocess, sys, time, traceback
import numpy as np
from PIL import Image, ImageDraw

H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
O=H/'native-smoke'
R=Path('/opt/gs-slam-lab/shuolee-hsfm116')
sys.path[:0]=[str(R/'dust3r'),str(R/'croco'),str(R/'python')]
PAIRS=[(75,100),(100,75),(13,163),(163,13)]
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(name,value):
    p=O/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def literal(n):
    if isinstance(n,ast.Constant):return n.value
    if isinstance(n,(ast.Tuple,ast.List)):
        vals=[literal(x) for x in n.elts];return tuple(vals) if isinstance(n,ast.Tuple) else vals
    if isinstance(n,ast.Name) and n.id=='inf':return float('inf')
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -literal(n.operand)
    raise ValueError(ast.dump(n))

def main():
    assert not O.exists(),'Do not overwrite or repeat completed/partial native forward'
    O.mkdir()
    inp=H/'input-pilot/inputs.json';rows={r['raw_frame']:r for r in json.loads(inp.read_text())['records']}
    ids=[75,100,13,163]
    save('protocol.json',dict(created=now(),stage='native_tensor_and_input_check_only',
        pairs=PAIRS,input_manifest_sha256=sha(inp),size=512,batch_size=1,
        precision='official inference default float32, AMP off',seed=116,
        selection='Two fixed directions on a near-front pair and a wide opposite-side pair; no quality-based selection',
        hypotheses=['Pinned fork can run actual portrait images on existing Torch/CUDA',
                    'Native image tensor exactly matches the documented PIL resize/crop and RGB normalization'],
        limits={'wall_seconds':300,'smoke_total_P3_gpu_seconds':1800},
        stop_on=['Existing GPU job','Changed source hashes','Nonfinite outputs','Input identity or coordinate mismatch','Timeout or OOM'],
        calibration=False,subject_fit=False,gaussian_training=False,
        acceptance='Finite correctly shaped native outputs are an integration pass only, never material/camera/ghosting acceptance',
        source_inputs=[{k:rows[i][k] for k in ('raw_frame','image','image_sha256','processed_wh','center_affine')} for i in ids]))
    # Avoid competing work; retain actual process commands instead of trusting stale PIDs.
    proc=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        try:cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not cmd:continue
        if any(s in cmd for s in ('python','train','supervis','nvidia')):proc.append(dict(pid=int(d.name),command=cmd))
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    save('process-check.json',dict(updated=now(),self_pid=os.getpid(),processes=proc,gpu_returncode=gpu.returncode,gpu_compute=gpu.stdout))
    assert gpu.returncode==0 and not gpu.stdout.strip(),'GPU compute process already present'
    save('progress.json',dict(updated=now(),state='loading_native_model',pid=os.getpid(),pairs_done=0))
    start=time.monotonic()
    import torch
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    torch.manual_seed(116);torch.set_num_threads(2)
    p=Path('/mnt/d/GS-SLAM-Data/models/hsfm/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    checked=json.loads((H/'native-model-check.json').read_text());assert p.stat().st_size==checked['bytes']
    with torch.serialization.safe_globals([argparse.Namespace]):ckpt=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    call=ast.parse(ckpt['args'].model,mode='eval').body
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='AsymmetricCroCo3DStereo' and not call.args
    kw={k.arg:literal(k.value) for k in call.keywords};assert None not in kw
    kw['landscape_only']=False
    if kw.get('patch_embed_cls')=='ManyAR_PatchEmbed':kw['patch_embed_cls']='PatchEmbedDust3R'
    model=AsymmetricCroCo3DStereo(**kw);model.load_state_dict(ckpt['model'],strict=True,assign=True)
    del ckpt
    for i in ids:assert sha(rows[i]['image'])==rows[i]['image_sha256']
    images=load_images([rows[i]['image'] for i in ids],size=512,verbose=False)
    image_check=[]
    for i,v in zip(ids,images):
        raw=Image.open(rows[i]['image']).convert('RGB');scale=512/max(raw.size)
        resized=raw.resize(tuple(round(x*scale) for x in raw.size),Image.Resampling.LANCZOS)
        cx,cy=resized.width//2,resized.height//2;hx,hy=((2*cx)//16)*8,((2*cy)//16)*8
        ref=np.asarray(resized.crop((cx-hx,cy-hy,cx+hx,cy+hy)))
        actual=v['img'][0].permute(1,2,0).numpy()
        expected=ref.astype(np.float32)/255*2-1
        error=float(abs(expected-actual).max());assert error<1e-6
        assert list(ref.shape[1::-1])==rows[i]['processed_wh']
        image_check.append(dict(raw_frame=i,rgb_normalization_max_abs=error,shape=list(actual.shape)))
    model=model.eval().cuda();torch.cuda.reset_peak_memory_stats()
    gpu_start=time.monotonic();outputs=[];stats=[];panels=[]
    import matplotlib
    for pi,(a,b) in enumerate(PAIRS):
        # The pinned native inference function returns point maps in view-a's camera gauge.
        out=inference([(images[ids.index(a)],images[ids.index(b)])],model,'cuda',batch_size=1,verbose=False)
        outputs.append(out)
        pred1=out['pred1'];pred2=out['pred2']
        maps=[pred1['pts3d'][0].numpy(),pred2['pts3d_in_other_view'][0].numpy()]
        conf=[pred1['conf'][0].numpy(),pred2['conf'][0].numpy()]
        for xyz,c in zip(maps,conf):
            assert xyz.shape==(512,288,3) and c.shape==(512,288)
            assert np.isfinite(xyz).all() and np.isfinite(c).all()
        stats.append(dict(pair=[a,b],shape=list(maps[0].shape),
            confidence_quantiles=[np.quantile(c,[0,.5,.9,1]).tolist() for c in conf],
            positive_Z_fraction=[float((xyz[...,2]>0).mean()) for xyz in maps]))
        # Review-only display with shared pair scale, no source image edits or learned RGB.
        vals=np.concatenate([x[...,2].ravel() for x in maps]);lo,hi=np.quantile(vals,[.02,.98])
        clim=float(np.quantile(np.concatenate([np.log(c).ravel() for c in conf]),.98))
        for j,k in enumerate((a,b)):
            rgb=((images[ids.index(k)]['img'][0].permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
            zn=np.clip((maps[j][...,2]-lo)/max(hi-lo,1e-8),0,1)
            cn=np.clip(np.log(conf[j])/max(clim,1e-8),0,1)
            zr=(matplotlib.colormaps['viridis'](zn)[...,:3]*255).astype(np.uint8)
            cr=(matplotlib.colormaps['magma'](cn)[...,:3]*255).astype(np.uint8)
            canvas=Image.new('RGB',(864,548),'white');d=ImageDraw.Draw(canvas)
            for col,(arr,title) in enumerate(((rgb,f'{a:06d}->{b:06d} | RGB {k:06d}'),(zr,'Z in first camera | relative'),(cr,'log confidence | uncalibrated'))):
                canvas.paste(Image.fromarray(arr),(288*col,36));d.text((288*col+5,8),title,fill='black')
            canvas.save(O/f'pair-{pi+1}-view-{j+1}.jpg',quality=95)
        save('progress.json',dict(updated=now(),state='native_forward_running',pid=os.getpid(),pairs_done=pi+1,pairs_total=len(PAIRS),seconds=time.monotonic()-start))
        assert time.monotonic()-start<300
    torch.cuda.synchronize();gpu_seconds=time.monotonic()-gpu_start
    torch.save(outputs,O/'native-pair-output.pt')
    save('result.json',dict(updated=now(),state='native_gpu_forward_passed_visual_review_pending',
        pairs=PAIRS,images=ids,output_point_frame='Each directed pair has its own first-camera gauge; not a global scene',
        gpu_seconds=gpu_seconds,total_seconds=time.monotonic()-start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        pixel_checks=image_check,pair_statistics=stats,output_sha256=sha(O/'native-pair-output.pt'),
        physical_camera_accuracy_verified=False,material_identity_verified=False,calibration_candidate=False,
        subject_fit=False,gaussian_training=False,point_cloud_ply_exported=False,root_cause_resolved=False))
    save('progress.json',dict(updated=now(),state='completed',pid=os.getpid(),pairs_done=4))
    print(json.dumps(dict(state='completed_native_forward_only',gpu_seconds=gpu_seconds)),flush=True)

try:main()
except Exception as e:
    if O.exists():save('failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
    raise
