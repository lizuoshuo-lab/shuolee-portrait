"""Validate official downloaded DUSt3R checkpoint on CPU without subject data."""
from pathlib import Path
import argparse, ast, datetime as dt, hashlib, json, os, sys, time, traceback
H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
R=Path('/opt/gs-slam-lab/shuolee-hsfm116')
sys.path[:0]=[str(R/'dust3r'),str(R/'croco'),str(R/'python')]
os.environ['CUDA_VISIBLE_DEVICES']=''
def save(name,x):(H/name).write_text(json.dumps(x,indent=2)+'\n')
def literal(n):
    if isinstance(n,ast.Constant):return n.value
    if isinstance(n,(ast.Tuple,ast.List)):
        vals=[literal(x) for x in n.elts];return tuple(vals) if isinstance(n,ast.Tuple) else vals
    if isinstance(n,ast.Name) and n.id=='inf':return float('inf')
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -literal(n.operand)
    raise ValueError('Unexpected checkpoint model expression: '+ast.dump(n))

def main():
    assert not (H/'native-model-check.json').exists()
    now=dt.datetime.now(dt.timezone.utc).isoformat();t=time.monotonic()
    receipt=json.loads((H/'weight-acquisition.json').read_text(encoding='utf-8-sig'))
    p=Path('/mnt/d/GS-SLAM-Data/models/hsfm/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    assert p.stat().st_size==receipt['bytes']==2285005731
    save('native-model-check-progress.json',dict(updated=now,pid=os.getpid(),state='cpu_validation_running',gpu=False))
    import torch
    torch.set_num_threads(2)
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.inference import inference
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(p)
    assert set(unsafe)<= {'argparse.Namespace'},unsafe
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    expr=ckpt['args'].model;call=ast.parse(expr,mode='eval').body
    assert isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='AsymmetricCroCo3DStereo' and not call.args
    kwargs={kw.arg:literal(kw.value) for kw in call.keywords};assert None not in kwargs
    kwargs['landscape_only']=False
    if kwargs.get('patch_embed_cls')=='ManyAR_PatchEmbed':kwargs['patch_embed_cls']='PatchEmbedDust3R'
    assert kwargs['head_type']=='dpt' and kwargs['img_size']==(512,512)
    model=AsymmetricCroCo3DStereo(**kwargs)
    state=ckpt['model']; loaded=model.load_state_dict(state,strict=True,assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
    result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='passed_strict_cpu_model_load',
        checkpoint_path=str(p),bytes=p.stat().st_size,download_sha256=receipt['sha256'],
        publisher_checksum_available=False,unsafe_globals=unsafe,model_expression=expr,
        native_loader_compatibility=['explicit safe argparse.Namespace for PyTorch2.7 weights_only',
            'official ManyAR_PatchEmbed to PatchEmbedDust3R alias','official landscape_only=False'],
        matched_state_tensors=len(state),all_tensors_finite=True,imports_verified=True,
        parameters=sum(v.numel() for v in state.values()),seconds=time.monotonic()-t,
        gpu=False,subject_inference=False,calibration_candidate=False,root_cause_resolved=False)
    save('native-model-check.json',result)
    save('native-model-check-progress.json',dict(state='completed',updated=result['updated'],pid=os.getpid()))
    print(json.dumps(result),flush=True)
try:main()
except Exception as e:
    save('native-model-check-failure.json',dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        error=repr(e),traceback=traceback.format_exc(),subject_inference=False))
    raise
