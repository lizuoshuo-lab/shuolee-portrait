"""Prepare the single preregistered physical-material review from saved segments."""
from pathlib import Path
from datetime import datetime,timezone
from functools import lru_cache
import json,hashlib,time,os,signal,fcntl,traceback
import numpy as np
from PIL import Image
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
O=REC/'pattern-preflight/native-material';B=O/'bridge-control';V=B/'physical-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert not V.exists(),'Existing crops must be reviewed, never regenerated.'
    plan=read(B/'physical-review-plan.json');assert sha(B/'segments.json')==plan['input_sha256']
    assert sha(B/'head-family-catalog.json')==plan['catalog_sha256']
    lock=open('/tmp/shuolee-pattern-retained-review123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    V.mkdir();start=time.monotonic()
    def alarm(*a):raise TimeoutError('Review preparation600second limit')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    save(V/'execution.json',dict(started=now(),pid=os.getpid(),source_sha256=sha(Path(__file__)),plan_sha256=sha(B/'physical-review-plan.json'),native_wh=[2704,1520],crop_wh=[192,192],no_text=True,no_warp=True))
    catalog=read(B/'head-family-catalog.json')['families'];head_ids=sorted(catalog)[:160]
    ss=[s for s in read(B/'segments.json')['records'] if s['topologically_eligible']];lookup={s['id']:s for s in ss}
    manifest=read(O/'feature-manifest.json')['frames'];xy=[]
    for f in manifest:
        p=O/'features'/f['feature_file'];assert sha(p)==f['sha256'];xy.append(np.load(p,allow_pickle=False)['xy'])
    def region(node):
        i,j=node;x,y=xy[i][j];h,b=manifest[i]['regions'][0]['bounds'],manifest[i]['regions'][1]['bounds']
        return 'head' if h[0]<=x<h[2] and h[1]<=y<h[3] else ('person' if b[0]<=x<b[2] and b[1]<=y<b[3] else 'background')
    selected=[];seen=set(head_ids)
    for fid in head_ids:
        for seg in sorted(catalog[fid],key=lambda x:x['segment_id']):selected.append(dict(category='head',selection_anchor=None,segment=lookup[seg['segment_id']]))
    prior={x['family_id'] for x in read(O/'visual-review/index.json')['rows']}
    bg_gaps=[]
    for anchor in [0,24,49,74,98]:
        for split in ['FIT','HELD']:
            possible=[]
            for seg in ss:
                if seg['split']!=split or seg['family_id'] in seen or seg['family_id'] in prior:continue
                if any(i==anchor and region([i,j])=='background' for i,j in seg['nodes']):possible.append(seg)
            possible.sort(key=lambda s:(s['family_id'],s['id']));picked=0
            for seg in possible:
                if seg['family_id'] in seen:continue
                selected.append(dict(category='background',selection_anchor=anchor,segment=seg));seen.add(seg['family_id']);picked+=1
                if picked==4:break
            if picked<4:bg_gaps.append(dict(anchor=anchor,split=split,available=picked))
    # Selection is frozen before reading/cropping any source image.
    save(V/'selection.json',dict(registered=now(),head_families=head_ids,head_families_not_reviewed=sorted(catalog)[160:],
        records=[dict(category=x['category'],anchor=x['selection_anchor'],segment_id=x['segment']['id'],family_id=x['segment']['family_id'],split=x['segment']['split'],development_only=x['segment']['development_only']) for x in selected],background_gaps=bg_gaps))
    @lru_cache(maxsize=16)
    def source(slot):
        p=Path(manifest[slot]['image']);assert sha(p)==manifest[slot]['source_sha256'];im=Image.open(p).convert('RGB');assert im.size==(2704,1520);return im
    def crop(node):
        i,j=node;uv=xy[i][j];left,top=np.floor(uv).astype(int)-96;box=[int(left),int(top),int(left+192),int(top+192)]
        info=dict(slot=i,source_frame=manifest[i]['source_frame'],keypoint=j,uv=uv.tolist(),ROI_region=region(node),crop_xyxy=box,source_sha256=manifest[i]['source_sha256'],outside_image_padding=any([box[0]<0,box[1]<0,box[2]>2704,box[3]>1520]))
        return source(i).crop(box),info
    rows=[];extras=[]
    for rowid,item in enumerate(selected):
        seg=item['segment'];nodes=seg['nodes'];which=sorted({0,len(nodes)//2,len(nodes)-1});assert len(which)==3
        strip=Image.new('RGB',(576,192));views=[]
        for col,k in enumerate(which):im,v=crop(nodes[k]);strip.paste(im,(col*192,0));views.append(v)
        name=f'row_{rowid:03d}_native.png';strip.save(V/name)
        regionchange=set()
        for k in range(1,len(nodes)):
            if region(nodes[k-1])!=region(nodes[k]):regionchange.update([k-1,k])
        extra_indices=sorted(regionchange-set(which))
        extra_views=[]
        for k in extra_indices:
            im,v=crop(nodes[k]);name2=f'extra_row{rowid:03d}_slot{nodes[k][0]:03d}.png';im.save(V/name2)
            extra_views.append(dict(**v,image=name2));extras.append(dict(row=rowid,view=v,image=name2))
        rows.append(dict(row=rowid,category=item['category'],selection_anchor=item['selection_anchor'],family_id=seg['family_id'],segment_id=seg['id'],split=seg['split'],development_only=seg['development_only'],nodes=nodes,views=views,extra_views=extra_views,image=name,physical_annotation='pending'))
        if rowid%25==0:save(V/'progress.json',dict(updated=now(),stage='cropping',pid=os.getpid(),done=rowid+1,total=len(selected),training=False))
    # Ten native rows per page; no resampling, labels, arrows or synthetic pixels except black outside-source padding.
    for startrow in range(0,len(rows),10):
        subset=rows[startrow:startrow+10];page=Image.new('RGB',(576,len(subset)*192))
        for r,row in enumerate(subset):page.paste(Image.open(V/row['image']),(0,r*192));row['page']=f'retained_{startrow//10+1:02d}.png';row['page_row']=r
        page.save(V/subset[0]['page'])
    for startrow in range(0,len(extras),20):
        subset=extras[startrow:startrow+20];page=Image.new('RGB',(960,192*((len(subset)+4)//5)))
        for k,ex in enumerate(subset):page.paste(Image.open(V/ex['image']),(192*(k%5),192*(k//5)));ex['page']=f'crossings_{startrow//20+1:02d}.png';ex['page_cell']=k
        page.save(V/subset[0]['page'])
    save(V/'index.json',dict(updated=now(),rows=rows,extra_views=extras,head_families=160,head_families_unreviewed=len(catalog)-160,head_segments=sum(r['category']=='head' for r in rows),background_rows=sum(r['category']=='background' for r in rows),total_rows=len(rows),pages=(len(rows)+9)//10,crossing_pages=(len(extras)+19)//20,physical_review_done=False,physical_face_HELD=0,training=False))
    signal.alarm(0);save(V/'progress.json',dict(updated=now(),stage='crops_complete_actual_review_pending',pid=None,seconds=time.monotonic()-start,training=False))
    for root in [R,REC]:
        q=read(root/'queue-state.json');q.update(updated=now(),current_stage='pattern_retained_material_actual_review',running=None,worker_pid=None,
            previous_goal_turn_classification='progress: prior turn isolated confirmed false identity bridge and updated original report/viewer/PPT',latest_evidence=str(V/'index.json'),training_running=False)
        save(root/'queue-state.json',q)
    print(json.dumps(dict(rows=len(rows),head_families=160,head_segments=sum(r['category']=='head' for r in rows),background_rows=sum(r['category']=='background' for r in rows),extras=len(extras),pages=(len(rows)+9)//10,crossing_pages=(len(extras)+19)//20,seconds=time.monotonic()-start),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except Exception as e:
        if V.exists():save(V/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
