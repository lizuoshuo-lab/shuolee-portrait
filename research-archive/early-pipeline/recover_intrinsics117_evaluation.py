"""Recover ONLY the registered held-out evaluation. Never rerun either solve."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import time
import numpy as np
from PIL import Image,ImageDraw
import solve_shared_intrinsics117 as s

# The old evaluator has a numpy.bool_ in its gates. Preserve its boolean value
# when serializing; this is an output-type repair, not a changed decision rule.
original_save=s.save
def save_native(path,value):
    def native(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [native(v) for v in x]
        return x
    original_save(path,native(value))
s.save=save_native


def visuals(detail,rows):
    lookup={}
    for arm,dd in detail.items():
        for r in dd['records']:
            for f,obs,pred,label in zip(r['frames'],r['observed'],r['projected'],r['labels']):
                lookup.setdefault((arm,f),[]).append((obs,pred,label))
    for f in s.read(s.D/'preregistration.json')['visual_frames']:
        src=Image.open(s.local(rows[f-1]['image'])).convert('RGB')
        assert src.size==(1514,2694)
        mask=np.array(Image.open(s.local(rows[f-1]['mask'])).convert('L'))>127
        yy,xx=np.where(mask)
        face=[o for o,p,g in lookup.get(('fixed074',f),[]) if g in s.LABELS[:2]]
        cx,cy=np.mean(face,axis=0) if face else ((xx.min()+xx.max())/2,yy.min()+.12*(yy.max()-yy.min()))
        x0=int(np.clip(cx-250,0,1514-500));y0=int(np.clip(cy-250,0,2694-500))
        box=(x0,y0,x0+500,y0+500)
        # Labels belong in names/manifest, not in the scientific image pixels.
        board=Image.new('RGB',(1512,1407),'white')
        for col,arm in enumerate(['original']+s.ARMS):
            im=src.copy();dr=ImageDraw.Draw(im)
            if arm!='original':
                for obs,pred,g in lookup.get((arm,f),[]):
                    x,y=obs;u,v=pred
                    dr.ellipse((x-3,y-3,x+3,y+3),outline=(0,220,230),width=2)
                    dr.line((x,y,u,v),fill=(255,40,60),width=2)
                    dr.line((u-4,v,u+4,v),fill=(255,40,60),width=2)
                    dr.line((u,v-4,u,v+4),fill=(255,40,60),width=2)
            board.paste(im.resize((504,897),Image.Resampling.LANCZOS),(col*504,0))
            board.paste(im.crop(box),(col*504,907))
        board.save(s.D/f'review/{f:06d}_original_fixed074_sharedK_body_face.jpg',quality=95)
    s.save(s.D/'review/image-manifest.json',dict(columns=['original','fixed074','shared_K_only'],rows=['full image, uniform display scaling','identical 500x500 face crop, no deformation'],cyan='observed source point',red='held-out projection and residual',text_burned_into_images=False,validation_reused=True))


def main():
    assert not (s.D/'result.json').exists(), 'Evaluation already completed; inspect it.'
    pr=s.read(s.D/'preregistration.json')
    assert all(s.sha(s.local(p))==h for p,h in pr['source_hashes'].items())
    assert s.sha(s.Path(s.__file__))==pr['code_sha256']
    with np.load(s.P/'admitted-tracks.npz') as z:a={k:z[k] for k in z.files}
    matrices={};native={};saved={}
    for arm in s.ARMS:
        native[arm]=s.read(s.D/(arm+'-result.json'))
        with np.load(s.D/(arm+'.npz')) as z:
            matrices[arm]=z['K'].copy()
            assert np.max(abs(z['w2c']-a['w2c']))<1e-10
        assert np.array_equal(matrices[arm],np.array(native[arm]['K']))
        saved[arm]=s.sha(s.D/(arm+'.npz'))
    q=s.read(s.O/'queue-state.json')
    q.update(updated=s.now(),state='D1_K_evaluation_recovery',active_pid=os.getpid(),active_log='D1-intrinsics-only/recovery.log',training_started=False)
    s.save(s.O/'queue-state.json',q)
    s.save(s.D/'evaluation-recovery.json',dict(started=s.now(),pid=os.getpid(),completed_solver_matrices=saved,new_solver_runs=0,training_started=False,script_sha256=s.sha(s.Path(__file__))))
    start=time.monotonic()
    # The original evaluator sliced archive['observations'] once per track.
    # Each NpzFile lookup decompressed a fresh full array; each retained view kept
    # that full array alive. Materialize each source array exactly once instead.
    # All observations, labels, splits, point fits and acceptance gates are identical.
    with np.load(s.SRC) as archive: cached={k:archive[k] for k in archive.files}
    class CachedSourceNumpy:
        def __getattr__(self,name):return getattr(np,name)
        def load(self,path,*args,**kwargs):
            return cached if s.Path(path)==s.SRC else np.load(path,*args,**kwargs)
    s.save(s.D/'evaluation-memory-repair.json',dict(updated=s.now(),original_code_sha256=pr['code_sha256'],cached_array_bytes=sum(v.nbytes for v in cached.values()),changed='Materialize the immutable source NPZ once instead of per-track decompression. No numerical algorithm, observations, splits, solver or threshold changed.',source_sha256=s.sha(s.SRC),prior_recovery='Process exited after WSL kernel OOM. The earlier historical interruption remains unconfirmed.'))
    s.np=CachedSourceNumpy()
    try:result,detail,rows=s.evaluate(a,matrices,native)
    finally:s.np=np
    s.secondary(matrices,a['w2c'])
    visuals(detail,rows)
    assert all(s.sha(s.D/(arm+'.npz'))==h for arm,h in saved.items())
    assert all(s.sha(s.local(p))==h for p,h in pr['source_hashes'].items())
    result.update(state='completed_visual_review_pending',evaluation_recovery_seconds=time.monotonic()-start,sources_unchanged=True,solves_repeated=False)
    s.save(s.D/'result.json',result)
    q.update(updated=s.now(),state='D1_K_review_pending',stage='D1_review',active_pid=None,result='D1-intrinsics-only/result.json',next_action='Inspect source overlays and held statistics. No camera promotion or training unless all frozen gates pass.')
    s.save(s.O/'queue-state.json',q)
    print(__import__('json').dumps(result,indent=2,default=lambda x:x.item()),flush=True)

if __name__=='__main__':main()
