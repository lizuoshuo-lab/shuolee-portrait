"""One read-only comparison of existing scene support; no camera solver.

Image-height and estimated-depth labels are not physical material truth.
Samples are selected without residual values, and whole-track splits stay fixed.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time
import numpy as np
from PIL import Image,ImageDraw
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'grayT-scene-support-cross-run-review'
A=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
B=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(a):
    a=np.asarray(a,float).ravel()
    return dict(n=int(a.size),minimum=float(a.min()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()),ge1=int((a>=1).sum())) if a.size else dict(n=0)
def spaced(a,n):
    a=sorted(a);return [a[i] for i in np.unique(np.linspace(0,len(a)-1,min(n,len(a))).astype(int))] if a else []
def isin(xy,b,margin=20):
    b=np.asarray(b);return (xy[:,0]>=b[0]-margin)&(xy[:,0]<=b[2]+margin)&(xy[:,1]>=b[1]-margin)&(xy[:,1]<=b[3]+margin)
def nativepath(p):return Path('//wsl.localhost/Ubuntu-20.04'+p) if p.startswith('/opt/') else Path(p[5].upper()+':/'+p[7:])

def main():
    if (O/'started.json').exists():raise RuntimeError('This audit already started; do not repeat.')
    plan=read(O/'plan.json');start=time.monotonic()
    assert all(sha(Path(s['path']))==s['sha256'] for s in plan['sources'])
    save(O/'started.json',dict(updated=now(),plan_sha256=sha(O/'plan.json'),source_sha256=sha(Path(__file__))))
    ar=read(A/'data/inputs.json')['records'];results={};index=[];selected=[]
    for label,root in [('research',A),('company00',B)]:
        rows=read(root/'data/inputs.json')['records'];z=dict(np.load(root/'camera/solver-output.npz'))
        records=read(root/'camera/appearance-tracks.json')['records'];tm={t['id']:t for t in records}
        held=read(root/'camera/held-evaluation.json')['records'];fitids=[str(x) for x in z['track_ids']]
        fit=np.load(root/'fit/fitted.npz');n=len(rows);E=z['w2c'];K=z['K']
        assert n==95 and [r['source_frame_index'] for r in rows]==[r['source_frame_index'] for r in ar]
        assert np.max(np.abs(E-np.array([r['w2c'] for r in rows])))==0
        assert np.max(np.abs(K-np.array([r['K_integer_centers'] for r in rows])))==0
        assert all(not tm[x]['held'] for x in fitids) and all(tm[x['id']]['held'] for x in held)
        assert not set(fitids)&{x['id'] for x in held}
        scales=np.array([r['wh'] for r in rows])/np.array([r['wh'] for r in ar]);features=[];native=[];flags=[];hashes=[];background_hits=[]
        for i,r in enumerate(rows):
            fp=root/f'camera/features-{i:03d}.npz';xy=np.load(fp)['xy'];features.append(xy);hashes.append(sha(fp))
            nv=(xy+.5)/scales[i]-.5;native.append(nv)
            mp=nativepath(r['mask']);assert sha(mp)==r['mask_sha256'];mask=np.asarray(Image.open(mp).convert('L'))>127
            ij=np.rint(xy).astype(int);person=mask[ij[:,1],ij[:,0]]
            anybox=isin(nv,ar[i]['initial_box_xyxy'])
            for b in ar[i].get('other_person_boxes_xyxy',[]):anybox|=isin(nv,b)
            fb=ar[i]['face_box_xyxy'];fh=fb[3]-fb[1]
            headband=(nv[:,1]>=fb[1]-.5*fh)&(nv[:,1]<=fb[3]+.5*fh)
            pb=ar[i]['initial_box_xyxy'];pw=pb[2]-pb[0]
            nearx=(nv[:,0]>=pb[0]-.5*pw)&(nv[:,0]<=pb[2]+.5*pw)
            flags.append(np.stack([person,~anybox,headband,headband&nearx&~anybox],axis=1))
            if label=='company00':
                bp=root/'camera/background-masks'/r['name'];bg=np.asarray(Image.open(bp).convert('L'))>127
                background_hits.append(int((~bg[ij[:,1],ij[:,0]]).sum()))
        cat={}
        for t in records:
            fl=np.array([flags[i][j] for i,j in t['nodes']]);ni=np.array([i for i,j in t['nodes']])
            cat[t['id']]=dict(outside_all=bool(fl[:,1].all()),near_head=bool((fl[:,3]&(ni>=39)&(ni<=79)).any()),nodes=len(ni),span=int(ni.max()-ni.min()))
        ci=z['obs_cam'];pi=z['obs_point'];uv=z['observations'];q=np.einsum('nij,nj->ni',E[ci,:3,:3],z['points'][pi])+E[ci,:3,3]
        pp=q@K.T;err=(pp[:,:2]/pp[:,2,None]-uv)/scales[ci]
        ff=[];identity=0.
        for a,(i,p) in enumerate(zip(ci,pi)):
            j=dict(tm[fitids[p]]['nodes'])[int(i)];identity=max(identity,float(np.abs(uv[a]-features[i][j]).max()));ff.append(flags[i][j])
        assert identity<1e-9
        hci=[];hids=[];herr=[];hf=[];hz=[];hp={t['id']:np.asarray(t['point']) for t in held}
        for tr in held:
            assert tr['test_frames']==[i for i,j in tr['nodes'][1::2]]
            for (i,j),saved_error in zip(tr['nodes'][1::2],tr['errors']):
                cam=E[i,:3,:3]@hp[tr['id']]+E[i,:3,3];proj=K@cam;res=proj[:2]/proj[2]-features[i][j]
                assert abs(np.linalg.norm(res)-saved_error)<1e-8
                hci.append(i);hids.append(tr['id']);herr.append(np.linalg.norm(res/scales[i]));hf.append(flags[i][j]);hz.append(cam[2])
        data={'FIT':dict(ci=ci,ids=np.array(fitids)[pi],errors=np.linalg.norm(err,axis=1),flags=np.array(ff),depth=q[:,2]),
              'HELD':dict(ci=np.array(hci),ids=np.array(hids),errors=np.array(herr),flags=np.array(hf),depth=np.array(hz))}
        centers=-np.einsum('nji,nj->ni',E[:,:3,:3],E[:,:3,3]);extent=np.max(np.linalg.norm(centers-centers[0],axis=-1))
        # Head joint only supplies a conditional depth reference, not a truth label.
        head=fit['joints_world'][:,15];headz=np.einsum('nij,nj->ni',E[:,:3,:3],head)[:,2]+E[:,2,3]
        result=dict(original_feature_hashes=hashes,frames=n,working_wh=rows[0]['wh'],native_wh=ar[0]['wh'],fit_tracks=len(fitids),held_tracks=len(held),groups={},samples={},
                    background_mask_feature_violations=sum(background_hits) if background_hits else None,source_identity_max_abs=identity)
        pointmap={t:x for t,x in zip(fitids,z['points'])};pointmap.update(hp)
        for split,d in data.items():
            outside=np.array([cat[t]['outside_all'] for t in d['ids']]);band=d['flags'][:,2];near=d['flags'][:,3]
            checks={'all':np.ones(len(outside),bool),'person_mask':d['flags'][:,0],'whole_track_outside_boxes':outside,
                    'head_height_band_outside_boxes':outside&band,'near_head_band_during_M0':outside&near&(d['ci']>=39)&(d['ci']<=79)}
            groups={}
            for name,m in checks.items():
                ids=sorted(set(d['ids'][m]));spans=[cat[t]['span'] for t in ids]
                groups[name]=dict(native_reprojection=stats(d['errors'][m]),tracks=len(ids),views=len(set(d['ci'][m])),track_span_selected_frames=stats(spans),
                    estimated_depth_to_camera_path_extent=stats(d['depth'][m]/extent),estimated_depth_to_fitted_head=stats(d['depth'][m]/headz[d['ci'][m]]),
                    nonpositive_depth=int((d['depth'][m]<=0).sum()))
            result['groups'][split]=groups
            eligible=[t for t in sorted(set(d['ids'])) if cat[t]['outside_all'] and cat[t]['near_head'] and cat[t]['nodes']>=3 and cat[t]['span']>=6]
            ids=spaced(eligible,2 if split=='FIT' else 4);result['samples'][split]=dict(eligible_count=len(eligible),selected=ids)
            for k,ident in enumerate(ids):
                nodes=tm[ident]['nodes'];chosen=[nodes[i] for i in np.unique(np.linspace(0,len(nodes)-1,3).astype(int))];parts=[];sources=[]
                point=pointmap[ident];rays=[]
                for i,j in chosen:
                    xy=native[i][j];cx,cy=np.rint(xy).astype(int);crop=(int(cx-128),int(cy-128),int(cx+128),int(cy+128))
                    ip=nativepath(ar[i]['image']);im=Image.open(ip).convert('RGB');tile=im.crop(crop);parts.append(tile)
                    sources.append(dict(frame=ar[i]['frame'],xy_native=xy.tolist(),crop=crop,image_sha256=sha(ip),role=('HELD_CHECK' if [i,j] in nodes[1::2] else 'TRIANGULATION_ANCHOR') if split=='HELD' else 'CAMERA_FIT'))
                    ray=point-centers[i];rays.append(ray/np.linalg.norm(ray))
                angles=[float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1)))) for ai,a in enumerate(rays) for b in rays[ai+1:]]
                sheet=Image.new('RGB',(256*len(parts),256),'white');loc=Image.new('RGB',sheet.size,'white')
                for j,tile in enumerate(parts):
                    sheet.paste(tile,(j*256,0));mark=tile.copy();dr=ImageDraw.Draw(mark);dr.ellipse((123,123,133,133),outline=(255,150,0),width=1);loc.paste(mark,(j*256,0))
                name=f'{label}_{split}_{k+1:02d}_{ident[:10]}';sheet.save(O/(name+'_native.png'));loc.save(O/(name+'_locator.png'))
                index.append(dict(frame=ident,region=label+'_'+split+'_头部高度附近背景候选，尚未物理验收',file=name+'_native.png',raw=name+'_native.png',locator=name+'_locator.png',columns=[s['frame'] for s in sources],sources=sources))
                selected.append(dict(run=label,split=split,track_id=ident,nodes=nodes,span=cat[ident]['span'],conditional_three_view_parallax_degrees=angles,sample=name))
        save(O/(label+'-audit.json'),result);results[label]=result
        if time.monotonic()-start>plan['cpu_seconds']:raise RuntimeError('Audit budget exhausted; no repeat.')
    save(O/'index.json',dict(images=index));save(O/'selected-samples.json',dict(selection_uses_error=False,samples=selected))
    assert all(sha(Path(s['path']))==s['sha256'] for s in plan['sources'])
    save(O/'result.json',dict(updated=now(),seconds=time.monotonic()-start,state='completed_actual_material_review_pending',
        runs={k:{kk:vv for kk,vv in v.items() if kk!='original_feature_hashes'} for k,v in results.items()},source_hashes_unchanged=True,
        no_new_features=True,no_camera_or_point_optimization=True,physical_material_identity_verified=False,quality_accepted=False))
    print(json.dumps({k:dict(head_band=v['groups']['HELD']['near_head_band_during_M0'],samples=v['samples'],background_mask_feature_violations=v['background_mask_feature_violations']) for k,v in results.items()},ensure_ascii=False))
if __name__=='__main__':main()
