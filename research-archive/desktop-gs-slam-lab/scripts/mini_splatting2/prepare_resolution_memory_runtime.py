"""Isolated implementation-only memory changes; preserve the loss and old repository."""
import hashlib
import json
from pathlib import Path
import shutil
import time
ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')


def main():
    source=RUNTIME/'repo-depthguide';target=RUNTIME/'repo-depth-memory';assert not target.exists()
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    train=target/'msv2/train_depth_guided.py';text=train.read_text()
    text=text.replace('MOTION_CACHE={}','from collections import OrderedDict\nMOTION_CACHE=OrderedDict()\nMOTION_HOST={}\nMOTION_CACHE_LIMIT=4')
    old="""        x=motion_np.asarray(MotionImage.open(MOTION_ROOT/(camera.image_name+'.png')).convert('L')).copy()
        mask=torch.from_numpy((x>0).astype('float32'))[None,None].to(gt.device)"""
    new="""        while len(MOTION_CACHE)>=MOTION_CACHE_LIMIT:MOTION_CACHE.popitem(last=False)
        if camera.image_name not in MOTION_HOST:
            x=motion_np.asarray(MotionImage.open(MOTION_ROOT/(camera.image_name+'.png')).convert('L')).copy()
            MOTION_HOST[camera.image_name]=torch.from_numpy((x>0).astype('float32'))[None,None]
        mask=MOTION_HOST[camera.image_name].to(gt.device)"""
    assert old in text;text=text.replace(old,new)
    text=text.replace('    return MOTION_CACHE[key]','    MOTION_CACHE.move_to_end(key)\n    return MOTION_CACHE[key]')
    old="""        loss = loss + depth_guidance(viewpoint_cam,gaussians,iteration,valid,gaussians._culling[:,viewpoint_cam.uid])
        loss.backward()"""
    new="""        # The parameters are unchanged between these backward calls.
        # Release RGB raster/SSIM saved tensors before allocating the depth graph.
        loss.backward()
        rgb_loss_value = loss.detach()
        depth_loss = depth_guidance(viewpoint_cam,gaussians,iteration,valid,gaussians._culling[:,viewpoint_cam.uid])
        if depth_loss.requires_grad:depth_loss.backward()
        loss = rgb_loss_value + depth_loss.detach()
        if iteration % 100 == 0:
            free_bytes,total_bytes=torch.cuda.mem_get_info()
            with (MotionPath(dataset.model_path)/'memory-usage.jsonl').open('a') as memory_log:
                memory_log.write(motion_json.dumps(dict(iteration=iteration,gaussians=len(gaussians.get_xyz),
                    allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,free_mib=free_bytes/2**20,
                    prior_cache_entries=len(depth_guidance.cache),motion_cache_entries=len(MOTION_CACHE)))+'\\n')"""
    assert old in text;text=text.replace(old,new)
    train.write_text(text)
    helper=target/'native_depth_guidance.py';text=helper.read_text()
    text=text.replace('import json','from collections import OrderedDict\nimport json',1)
    text=text.replace('self.coefficient=float(coefficient);self.cache={};self.usage={};self.records=[]',
        'self.coefficient=float(coefficient);self.cache=OrderedDict();self.host={};self.cache_limit=4;self.usage={};self.records=[]')
    old="""            with np.load(self.prior_root/(camera.image_name+'.npz')) as n:
                inverse=torch.tensor(n['inverse_depth'].astype('float32'),device='cuda')[None,None]
                confidence=torch.tensor(n['weight'].astype('float32'),device='cuda')[None,None]"""
    new="""            while len(self.cache)>=self.cache_limit:self.cache.popitem(last=False)
            if camera.image_name not in self.host:
                with np.load(self.prior_root/(camera.image_name+'.npz')) as n:
                    self.host[camera.image_name]=(torch.from_numpy(n['inverse_depth'].astype('float32').copy())[None,None],
                        torch.from_numpy(n['weight'].astype('float32').copy())[None,None])
            inverse,confidence=(x.to('cuda') for x in self.host[camera.image_name])"""
    assert old in text;text=text.replace(old,new)
    text=text.replace('        return self.cache[key]','        self.cache.move_to_end(key)\n        return self.cache[key]')
    helper.write_text(text)
    out=ROOT/'native-resolution-memory-runtime';out.mkdir(exist_ok=False)
    inherited={}
    for p in source.rglob('*.py'):
        if '__pycache__' in p.parts:continue
        rel=p.relative_to(source)
        if str(rel) in ['msv2/train_depth_guided.py','native_depth_guidance.py']:continue
        assert p.read_bytes()==(target/rel).read_bytes();inherited[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    state=dict(status='prepared_pending_equivalence',created=time.time(),repository=str(target),
        source_repository=str(source),changes=['Bound GPU caches to4camera-resolution pairs; keep raw priors/masks in CPU memory',
            'Accumulate RGB and depth gradients in separate backward calls before the same optimizer step',
            'Record allocated/reserved/peak memory and cache entries every100steps'],
        loss_weights_and_schedules_unchanged=True,rasterizer_unchanged=True,
        changed_files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [train,helper]},
        inherited_files=inherited)
    (out/'manifest.json').write_text(json.dumps(state,indent=2));print(json.dumps({k:v for k,v in state.items() if k!='inherited_files'}))


if __name__=='__main__':main()
