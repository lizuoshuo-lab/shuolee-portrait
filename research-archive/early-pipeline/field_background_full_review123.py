"""Produce the preregistered full-view material review; no geometry or matching."""
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
import os,sys,json,hashlib,time,signal,traceback
import numpy as np
from PIL import Image,ImageDraw
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
F=T/'shuolee/current/local-clothed-surface-123/field-preflight'
O=F/'native-material';B=O/'topology-review';OUT=B/'background-full-review'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def mapped(p):
    p=p.replace('\\','/')
    return Path('/mnt/'+p[0].lower()+p[2:]) if len(p)>2 and p[1]==':' else Path(p)

def main():
    assert not OUT.exists(),'Already prepared: inspect existing review, do not rerun.'
    plan=read(F/'background-material-review-plan.json')
    for p,h in plan['inputs'].items():assert sha(mapped(p))==h,p
    assert len(plan['selected'])==10 and plan['observations']==102
    start=time.monotonic();OUT.mkdir()
    save(OUT/'execution.json',dict(started=now(),source_sha256=sha(Path(__file__)),plan_sha256=sha(F/'background-material-review-plan.json'),
        pid=os.getpid(),command=sys.argv,CPU_only=True,feature_calls=0,geometry_calls=0,training_calls=0))
    def timeout(*_):raise TimeoutError('120-second preregistered material review preparation budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    manifest=read(O/'feature-manifest.json')['frames']
    segments={q['id']:q for q in read(B/'segments.json')['records']}
    verified={};features={};records=[]
    @lru_cache(maxsize=4)
    def original(slot):
        row=manifest[slot];p=Path(row['image']);digest=sha(p)
        assert digest==row['source_sha256'];verified[slot]=digest
        im=Image.open(p).convert('RGB');assert im.size==(2704,1520)
        return im
    def xy(slot,k):
        if slot not in features:
            row=manifest[slot];p=O/'features'/row['feature_file'];assert sha(p)==row['sha256']
            with np.load(p,allow_pickle=False) as z:features[slot]=z['xy'].copy()
        return features[slot][k]
    for item in plan['selected']:
        s=segments[item['segment_id']]
        assert all(s[k]==item[k] for k in ['family_id','split','nodes','development_only'])
        assert s['topologically_eligible'] and s['development_only']
        tiles=[];views=[]
        for slot,k in s['nodes']:
            im=original(slot);uv=xy(slot,k);cx,cy=np.floor(uv).astype(int)
            box=[int(cx-96),int(cy-96),int(cx+96),int(cy+96)]
            context=im.crop(box);smallbox=[int(cx-32),int(cy-32),int(cx+32),int(cy+32)]
            center=im.crop(smallbox);mark=ImageDraw.Draw(center)
            u,v=uv-[cx-32,cy-32];mark.ellipse((u-3,v-3,u+3,v+3),outline=(0,160,255),width=1)
            tile=Image.new('RGB',(256,192),'white');tile.paste(context,(0,0));tile.paste(center,(192,64));tiles.append(tile)
            views.append(dict(slot=slot,source_frame=manifest[slot]['source_frame'],frame=Path(manifest[slot]['image']).stem,keypoint=k,
                source_image=manifest[slot]['image'],source_sha256=manifest[slot]['source_sha256'],feature_sha256=manifest[slot]['sha256'],
                uv_native=uv.tolist(),context_xyxy=box,center_xyxy=smallbox,source_pixels_resampled=False))
        # A fixed 4-column native contact sheet per family, external text index.
        canvas=Image.new('RGB',(1024,192*((len(tiles)+3)//4)),'white')
        for j,tile in enumerate(tiles):canvas.paste(tile,((j%4)*256,(j//4)*192))
        name=f'family_row_{item["review_row"]:03d}_all_native.png';canvas.save(OUT/name)
        records.append(dict(**item,views=views,file=name,sha256=sha(OUT/name),columns=4,
            native_context_panel=[192,192],native_center_panel=[64,64],center_marker='blue ring, no coordinate adjustment',
            physical_review_state='pending',independent_accuracy_verified=False))
        save(OUT/'progress.json',dict(updated=now(),stage='preparing_native_review',pid=os.getpid(),families=len(records),total=10))
    assert sum(len(q['views']) for q in records)==102
    for p,h in plan['inputs'].items():assert sha(mapped(p))==h,p
    save(OUT/'index.json',dict(rows=records,all_original_selected_observations_preserved=True,
        pixel_domain='native MP4',new_features=0,new_matches=0,resampling=False,burned_text=False,physical_review_pending=True))
    save(OUT/'result.json',dict(updated=now(),state='complete_actual_review_pending',families=10,observations=102,
        unique_source_frames=len(verified),all_source_hashes_verified=True,full_original_splits_preserved=True,
        camera_solver_calls=0,SMPLX_fit_calls=0,training_calls=0,physical_accuracy_accepted=False,
        seconds=time.monotonic()-start,source_sha256=sha(Path(__file__))))
    signal.alarm(0);save(OUT/'progress.json',dict(updated=now(),stage='completed',pid=None,families=10,total=10))
    print(json.dumps(read(OUT/'result.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
            save(OUT/'progress.json',dict(updated=now(),stage='failed',pid=None,error=repr(e)))
        raise
