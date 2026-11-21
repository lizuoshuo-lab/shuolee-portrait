"""One fixed-camera point-only test on a reviewed physical skin-mark candidate.

Observation localization remains uncertain; this is not subpixel ground truth.
"""
from datetime import datetime,timezone
import time,json
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native,stats

OUT=REC/'grayT-skin-texture-probe'
def main():
    started=time.monotonic()
    if (OUT/'geometry-result.json').exists():raise RuntimeError('Point test already executed; no repeat.')
    p=read(OUT/'plan.json');r=read(OUT/'tracking-result.json');v=read(OUT/'actual-review.json')
    assert r['algorithmic_checks_passed'] and v['all41_actually_reviewed'] and v['same_visible_pigment_mark_region']
    assert not v['subpixel_location_certified']
    assert [o['frame_number'] for o in r['observations']]==p['native_frame_numbers']
    sources=[ROOT/'camera/solver-output.npz',OUT/'plan.json',OUT/'tracking-result.json',OUT/'actual-review.json']
    hashes=[sha(f) for f in sources]
    z=np.load(sources[0]);K=z['K'];E=z['w2c'];obs=r['observations'];ci=np.array([o['frame_number']-1 for o in obs]);uv=np.array([o['xy'] for o in obs])
    anchor=np.array([o['role']=='ANCHOR' for o in obs]);assert anchor.sum()==6
    assert [o['frame_number'] for o in obs if o['role']=='HELD']==p['held_frame_numbers']
    # Multi-view linear triangulation from the6preregistered anchors only.
    P=np.einsum('ij,njk->nik',K,E[ci,:3,:]);A=[]
    for mat,pt in zip(P[anchor],uv[anchor]):A.extend([pt[0]*mat[2]-mat[0],pt[1]*mat[2]-mat[1]])
    _,s,V=np.linalg.svd(np.array(A));q=V[-1];x0=q[:3]/q[3]
    def project(x):
        xyz=np.einsum('nij,j->ni',E[ci,:3,:3],x)+E[ci,:3,3]
        px=xyz@K.T
        return px[:,:2]/px[:,2,None],xyz[:,2]
    opt=least_squares(lambda x:(project(x)[0][anchor]-uv[anchor]).ravel(),x0,max_nfev=100,method='lm',
        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    pred,depth=project(opt.x);err=np.linalg.norm(pred-uv,axis=1)
    entries=[]
    for o,u,pp,e,d in zip(obs,uv,pred,err,depth):
        entries.append(dict(frame=o['frame'],frame_number=o['frame_number'],role=o['role'],xy=u.tolist(),projected=pp.tolist(),
            residual=(pp-u).tolist(),euclidean_error_px=float(e),depth=float(d)))
    background=read(REC/'grayT-material-static-check/result.json')['tracks'];comparisons=[]
    for t in background:
        if not any(t['track_id'].startswith(a) for a in p['original_scene_negative_controls']):continue
        for b in t['checks']:
            f=next((o for o in entries if o['frame']==b['frame'] and o['role']=='HELD'),None)
            if f:comparisons.append(dict(frame=b['frame'],skin_px=f['euclidean_error_px'],background_px=b['euclidean_error_px'],
                skin_residual=f['residual'],background_residual=b['residual'],skin_depth=f['depth'],background_depth=b['depth'],
                background_track=t['track_id'],same_frame=True,both_held_from_point_estimation=True))
    assert all(sha(f)==h for f,h in zip(sources,hashes))
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
        point=opt.x.tolist(),anchor_fit=stats(err[anchor]),held_check=stats(err[~anchor]),observations=entries,
        matched_frame_scene_controls=comparisons,positive_depth_count=int((depth>0).sum()),frames=41,
        optimization=dict(variables=3,loss='linear all6anchors',nfev=opt.nfev,success=bool(opt.success),
            initial_squared_error=float(np.sum((project(x0)[0][anchor]-uv[anchor])**2)),final_squared_error=float(2*opt.cost),
            jacobian_singular_values=np.linalg.svd(opt.jac,compute_uv=False).tolist(),DLT_singular_values=s.tolist()),
        sources=[dict(path=str(f),sha256=h) for f,h in zip(sources,hashes)],
        no_camera_parameter_updated=True,no_held_observation_used_in_point_fit=True,
        subpixel_location_certified=False,absolute_motion_or_camera_cause_identified=False,
        warning='The same visible pigment-mark region was tracked, but point-center drift of a few pixels is visually possible. Flow roundtrip<0.5px is not a physical uncertainty bound. Neither this point nor the sparse scene proxies certifies true cameras.',
        geometry_is_diagnostic_only=True,training_admitted=False,quality_accepted=False)
    assert result['seconds']<120
    save(OUT/'geometry-result.json',result)
    # The4same-frame face/control comparisons were defined by existing test IDs.
    # Produce raw native128px crops and separate observation/projection overlays.
    rows=read(ROOT/'data/training-inputs.json')['records'];canvas=Image.new('RGB',(256*2,256*2),(245,245,245));images=[]
    for n,c in enumerate(comparisons):
        assert n<4
        o=next(a for a in entries if a['frame']==c['frame']);i=o['frame_number']-1
        xy=np.array(o['xy']);pp=np.array(o['projected']);cx,cy=map(int,np.round(xy));crop=[cx-128,cy-128,cx+128,cy+128]
        with Image.open(native(rows[i]['image'])) as im:tile=im.convert('RGB').crop(crop)
        rawname=f'same_frame_{o["frame_number"]:03d}_native.png';tile.save(OUT/rawname)
        d=ImageDraw.Draw(tile);a=xy-crop[:2];b=pp-crop[:2]
        d.line((*a,*b),fill=(255,176,0),width=1);d.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4),outline=(0,255,255),width=1)
        d.line((b[0]-5,b[1],b[0]+5,b[1]),fill=(255,176,0),width=1);d.line((b[0],b[1]-5,b[0],b[1]+5),fill=(255,176,0),width=1)
        name=f'same_frame_{o["frame_number"]:03d}_projection.png';tile.save(OUT/name);canvas.paste(tile,((n%2)*256,(n//2)*256))
        images.append(dict(frame=o['frame'],region='皮肤色素点留出：青圈观测／橙十字冻结相机下的静态三维点',file=name,raw=rawname,columns=['native256px'],crop=crop))
    canvas.save(OUT/'same_frame_skin_projection.png')
    idx=read(OUT/'index.json');idx['images'].extend(images);save(OUT/'index.json',idx)
    print(json.dumps({k:v for k,v in result.items() if k not in ['observations','sources']},ensure_ascii=False))

if __name__=='__main__':main()
