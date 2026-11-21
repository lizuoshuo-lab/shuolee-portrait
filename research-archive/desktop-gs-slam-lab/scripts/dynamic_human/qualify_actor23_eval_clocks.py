"""Disjoint calibration-only camera/clock diagnostics from native feature correspondences.

No reserved appearance-window images or author surfaces; empirical clocks are not hardware timestamps.
"""
from pathlib import Path
import json,hashlib,time,itertools
import cv2,numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont

L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';Q=O/'evaluation-time';Q.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (Q/'summary.json').exists()
plan=read(O/'validation-reservation.json');cams=sorted(read(O/'cameras.json'));audit={(r['camera'],r['frame']):r for r in read(Q/'calibration-input-audit.json')['records']}
cv2.setNumThreads(4);sift=cv2.SIFT_create(nfeatures=6000);matcher=cv2.BFMatcher(cv2.NORM_L2);cal=read(O/'cameras.json');centers=np.array([cal[c]['camera_center_world'] for c in cams]);pairs=set()
for i in range(len(cams)):
    for j in np.argsort(np.linalg.norm(centers-centers[i],axis=1))[1:4]:pairs.add(tuple(sorted((i,int(j)))))
pairs=sorted(pairs);anchors=[584,588,592];features={};background={};source={};start=time.monotonic()
def load(c,f):
    p=O/audit[c,f]['input'];assert sha(p)==audit[c,f]['input_sha256'];z=np.load(p)
    return {k:z[k] for k in ['rgb','matte','valid','K','w2c']}
def describe(gray,mask):
    kp,des=sift.detectAndCompute(gray,mask.astype(np.uint8)*255)
    assert len(kp)>15 and des is not None
    return np.float32([p.pt for p in kp]),des
def mutual(a,b):
    ab={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(a,b,k=2) if m.distance<.7*n.distance}
    ba={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(b,a,k=2) if m.distance<.7*n.distance}
    return np.array([(i,j) for i,j in ab.items() if ba.get(j)==i],dtype=np.int64).reshape(-1,2)
def track(gray,other,xy):
    p0=xy.reshape(-1,1,2);p1,s,_=cv2.calcOpticalFlowPyrLK(gray,other,p0,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
    back,sb,_=cv2.calcOpticalFlowPyrLK(other,gray,p1,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
    err=np.linalg.norm(back[:,0]-xy,axis=1);p1=p1[:,0]
    good=(s[:,0]>0)&(sb[:,0]>0)&(err<.75)&np.isfinite(p1).all(1)&(p1[:,0]>=0)&(p1[:,0]<1920)&(p1[:,1]>=0)&(p1[:,1]<1080)
    return p1,good,err
for c in cams:
    for f in anchors:
        z=load(c,f);gray=cv2.cvtColor(z['rgb'],cv2.COLOR_RGB2GRAY);xy,de=describe(gray,(z['matte']>.9)&z['valid'])
        before=cv2.cvtColor(load(c,f-2)['rgb'],cv2.COLOR_RGB2GRAY);after=cv2.cvtColor(load(c,f+2)['rgb'],cv2.COLOR_RGB2GRAY)
        xm,gm,em=track(gray,before,xy);xp,gp,ep=track(gray,after,xy)
        good=gm&gp
        features[c,f]=dict(xy=xy,des=de,minus=xm,plus=xp,good=good)
        if f==588:
            bgmask=cv2.dilate((z['matte']>.05).astype(np.uint8),np.ones((31,31),np.uint8))==0
            background[c]=describe(gray,bgmask&z['valid']);source[c]=z
        print('FEATURES',c,f,len(xy),int(good.sum()),flush=True)
    save(Q/'progress.json',dict(cameras=len(source),elapsed_seconds=time.monotonic()-start))
def skew(t):return np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]],dtype=float)
def fundamental(i,j):
    a,b=source[cams[i]],source[cams[j]];Wi,Wj=a['w2c'],b['w2c'];R=Wj[:3,:3]@Wi[:3,:3].T;t=Wj[:3,3]-R@Wi[:3,3];F=np.linalg.inv(b['K']).T@skew(t)@R@np.linalg.inv(a['K'])
    P1=a['K']@Wi[:3];P2=b['K']@Wj[:3];e=P2@np.r_[centers[i],1];F2=skew(e)@P2@np.linalg.pinv(P1);F/=np.linalg.norm(F);F2/=np.linalg.norm(F2)
    error=min(np.max(abs(F-F2)),np.max(abs(F+F2)));assert error<1e-9,error
    return F,error
def residual(F,x,y):
    xx=np.c_[x,np.ones(len(x))];yy=np.c_[y,np.ones(len(y))];l2=xx@F.T;l1=yy@F
    return (yy*l2).sum(1)/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1)+1e-30)
def stats(x):
    x=np.abs(x);return dict(count=len(x),median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None,fraction_below1p5=float(np.mean(x<1.5)) if len(x) else None)
bgrows=[];tr=[];records=[];geometry=[];colors=[];fm={}
for i,j in pairs:
    ca,cb=cams[i],cams[j];F,err=fundamental(i,j);fm[i,j]=F
    pa,da=background[ca];pb,db=background[cb];m=mutual(da,db);d=residual(F,pa[m[:,0]],pb[m[:,1]])
    bgrows.append(dict(cameras=[ca,cb],formula_difference=err,sampson_pixel_error=stats(d)))
    for f in anchors:
        a,b=features[ca,f],features[cb,f];m=mutual(a['des'],b['des']);ok=a['good'][m[:,0]]&b['good'][m[:,1]];m=m[ok]
        x,y=a['xy'][m[:,0]],b['xy'][m[:,1]];d=residual(F,x,y)
        # Fixed loose descriptor+geometry gate, before any clock optimization.
        keep=abs(d)<8;m=m[keep];x=x[keep];y=y[keep]
        rec=dict(cameras=[ca,cb],anchor=f,mutual_track_matches=len(ok),track_valid=int(ok.sum()),raw_track_sampson=stats(d),retained=int(keep.sum()))
        records.append(rec)
        if len(m)<8:continue
        path=Q/f'pair{ca}-{cb}-frame{f:06d}.npz'
        aa=np.stack([a[k][m[:,0]] for k in ['minus','xy','plus']],1).astype(np.float64);bb=np.stack([b[k][m[:,1]] for k in ['minus','xy','plus']],1).astype(np.float64)
        np.savez_compressed(path,a=aa,b=bb,F=F,cameras=np.array([ca,cb]),anchor=f)
        tr.append(dict(i=i,j=j,frame=f,a=aa,b=bb,F=F,file=path.name))
        if f==588:
            P1=source[ca]['K']@source[ca]['w2c'][:3];P2=source[cb]['K']@source[cb]['w2c'][:3];h=cv2.triangulatePoints(P1,P2,x.T.astype(float),y.T.astype(float)).T;xyz=h[:,:3]/h[:,3:]
            z1=xyz@source[ca]['w2c'][:3,:3].T+source[ca]['w2c'][:3,3];z2=xyz@source[cb]['w2c'][:3,:3].T+source[cb]['w2c'][:3,3];q1=z1@source[ca]['K'].T;q2=z2@source[cb]['K'].T;q1=q1[:,:2]/q1[:,2:];q2=q2[:,:2]/q2[:,2:];re=np.maximum(np.linalg.norm(q1-x,axis=1),np.linalg.norm(q2-y,axis=1));v1=xyz-centers[i];v2=xyz-centers[j];cos=(v1*v2).sum(1)/np.linalg.norm(v1,axis=1)/np.linalg.norm(v2,axis=1);valid=(z1[:,2]>.1)&(z2[:,2]>.1)&(re<2)&(cos<np.cos(np.deg2rad(2)))&np.isfinite(xyz).all(1)
            geometry.append(xyz[valid]);uv=np.rint(x[valid]).astype(int);colors.append(source[ca]['rgb'][uv[:,1],uv[:,0]]/255.)
            rec['initialization_points']=int(valid.sum());rec['retained_reprojection']=stats(re[valid])
    print('PAIR',ca,cb,'bg',bgrows[-1]['sampson_pixel_error'],flush=True)
assert tr and geometry
points=np.concatenate(geometry);color=np.concatenate(colors);np.savez_compressed(Q/'calibration_only_geometry_not_for_training.npz',points=points,colors=color)
reference=cams.index('04');variables=[i for i in range(18) if i!=reference]
def offsets(v):
    full=np.zeros(18);full[variables]=v;return full
def interpolate(a,dt):
    return a[:,1]+(dt/4)*(a[:,2]-a[:,0])+(dt*dt/8)*(a[:,2]-2*a[:,1]+a[:,0])
def evaluate(v,subset):
    d=offsets(v);return [residual(r['F'],interpolate(r['a'],-d[r['i']]),interpolate(r['b'],-d[r['j']])) for r in subset]
fit=[r for r in tr if r['frame'] in [584,592]];val=[r for r in tr if r['frame']==588]
assert fit and val
fit_result=least_squares(lambda v:np.concatenate(evaluate(v,fit)),np.zeros(17),bounds=(-2,2),loss='cauchy',f_scale=1.,max_nfev=150,xtol=1e-9,ftol=1e-9,gtol=1e-9)
zero=np.zeros(17);v=fit_result.x;before_fit=np.concatenate(evaluate(zero,fit));after_fit=np.concatenate(evaluate(v,fit));before_val=np.concatenate(evaluate(zero,val));after_val=np.concatenate(evaluate(v,val));rowstats=[]
for r,a,b in zip(val,evaluate(zero,val),evaluate(v,val)):rowstats.append(dict(cameras=[cams[r['i']],cams[r['j']]],anchor=r['frame'],before=stats(a),after=stats(b),file=r['file']))
candidate=dict(zip(cams,offsets(v).tolist()));save(Q/'candidate-clock-offsets.json',dict(status='candidate_not_applied',offsets=candidate,reference='04',sign='physical common index = image index + offset; evaluate observed tracks at common index - offset',fit_anchors=[584,592],validation_anchor=588,parameter_bound_frames=2,optimizer_success=bool(fit_result.success),optimizer_message=str(fit_result.message),not_hardware_timestamps=True))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);sheet=Image.new('RGB',(1500,1240),'white');draw=ImageDraw.Draw(sheet)
for k,c in enumerate(cams):
    z=source[c];raw=Image.fromarray(z['rgb']);matte=np.rint(z['rgb']*z['matte'][...,None]+127.5*(1-z['matte'][...,None])).astype(np.uint8);x=(k%5)*300;y=(k//5)*310
    draw.text((x+4,y+3),f'Camera{c} | frame588 | source / matte',font=font,fill='black');sheet.paste(raw.resize((300,169)),(x,y+24));sheet.paste(Image.fromarray(matte).resize((200,112)),(x,y+196))
sheet.save(Q/'native-training-input-contact.png')
summary=dict(status='disjoint_calibration_only_clock_diagnostic_stationarity_check_required',dataset='actor2_3',native_resolution=[1920,1080],calibration_cameras=cams,reserved_appearance_RGB_opened=False,heldout_cameras_calibration_visible=True,background_pairs=bgrows,foreground_pairs=records,same_index_initialization_points=len(points),point_quantiles=np.quantile(points,[.005,.5,.995],axis=0).tolist(),clock_fit=dict(reference='04',fit_anchors=[584,592],validation_anchor=588,fit_before=stats(before_fit),fit_after=stats(after_fit),validation_before=stats(before_val),validation_after=stats(after_val),validation_pairs=rowstats,offsets=candidate,bound_hits=[c for c,x in candidate.items() if abs(x)>1.999],candidate_applied=False),limitations=['Feature descriptors and optical-flow cycle checks are not semantic correspondence ground truth.','Background errors qualify calibration only where matching is reliable; static shadows are not tracked material points.','Three-time quadratic feature interpolation is a local approximation.','Frame-index offsets do not establish hardware synchronization, rolling-shutter parameters or metric scale.','No reserved600..624appearance image, author surface or actor5_6 geometry/time/pose was used. Derived geometry is calibration-only and must not initialize training.'],goal_complete=False,elapsed_seconds=time.monotonic()-start)
save(Q/'summary.json',summary);print(json.dumps(dict(points=len(points),fit=summary['clock_fit']['fit_after'],validation_before=stats(before_val),validation_after=stats(after_val),bounds=summary['clock_fit']['bound_hits'])),flush=True)
