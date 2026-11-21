"""Known-camera static-background triangulation from training views only.

Does not consume released boost/surfs geometry or heldout camera images.
Produces candidate initialization, not a trained scene or ground-truth mesh.
"""
from pathlib import Path
import json,cv2,numpy as np
from audit_enerf_input import read_cameras,digest

LAB=Path(__file__).resolve().parents[2];root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');out=LAB/'outputs/dynamic-human-enerf-background-032';out.mkdir(exist_ok=True);assert not (out/'triangulation.json').exists();cv2.setNumThreads(8);cv2.setRNGSeed(73)
cameras=read_cameras(root/'optimized');held=['00','06','12'];train=sorted(set(cameras)-set(held));ordered=sorted(train,key=lambda c:cameras[c]['camera_center_world'][0]);pairs=[(ordered[i],ordered[j]) for i in range(len(ordered)) for j in range(i+1,min(i+4,len(ordered)))];sift=cv2.SIFT_create(nfeatures=8000);features={};images={};hashes={}
protocol=dict(status='frozen_before_training_background_features',dataset='actor5_6',train_cameras=train,heldout_cameras=held,source='Author released bkgd/{camera}.jpg static references; do not claim raw empty-capture provenance beyond the release',calibration='optimized/intri.yml andextri.yml; fixed OpenCV world-to-camera',features='Native rectified SIFT, <=8000per view',matching='Mutual FLANN nearest matches, ratio0.7 both directions, known-calibration Sampson <=1px',track_filter='At least3unique cameras, no duplicate camera feature in a component, multiview normalized DLT, all reprojection errors<=1.5px, all camera depths0.1..20released units, maximum ray angle>=1degree',source_geometry_used=False,heldout_RGB_used=False,pairs=pairs,output='Local candidate points and colors, no trained background or joint-person scene claim')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
for cam in train:
    path=root/'bkgd'/f'{cam}.jpg';rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);assert rgb.shape==(1080,1920,3);K=np.array(cameras[cam]['K']);D=np.array(cameras[cam]['D']);maps=cv2.initUndistortRectifyMap(K,D,None,K,(1920,1080),cv2.CV_32FC1);rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);kp,des=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None);xy=np.array([p.pt for p in kp]);features[cam]=(xy,des);images[cam]=rgb;hashes[cam]=dict(path=path.relative_to(root).as_posix(),sha256=digest(path));np.savez_compressed(out/f'features-{cam}.npz',xy=xy,descriptors=des);print('FEATURES',cam,len(xy),flush=True)
parent={}
def find(x):
    if x not in parent:parent[x]=x
    while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
    return x
def join(x,y):
    a,b=find(x),find(y)
    if a!=b:parent[b]=a
checks=[]
for a,b in pairs:
    xa,da=features[a];xb,db=features[b];matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=96));ab=matcher.knnMatch(da,db,k=2);ba=matcher.knnMatch(db,da,k=2);reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.7*n.distance};matches=[(m.queryIdx,m.trainIdx) for m,n in ab if m.distance<.7*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    wa=np.array(cameras[a]['w2c']);wb=np.array(cameras[b]['w2c']);R=wb[:3,:3]@wa[:3,:3].T;T=wb[:3,3]-R@wa[:3,3];F=np.linalg.inv(np.array(cameras[b]['K'])).T@np.cross(np.eye(3),T)@R@np.linalg.inv(np.array(cameras[a]['K']));m=np.array(matches,dtype=int).reshape(-1,2);x=np.c_[xa[m[:,0]],np.ones(len(m))];y=np.c_[xb[m[:,1]],np.ones(len(m))];l2=x@F.T;l1=y@F;dist=np.abs((y*l2).sum(1))/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1));keep=dist<=1
    for i,j in m[keep]:join((a,int(i)),(b,int(j)))
    np.savez_compressed(out/f'matches-{a}-{b}.npz',pairs=m,sampson=dist,accepted=keep)
    checks.append(dict(source=a,target=b,mutual_matches=len(m),geometric_inliers=int(keep.sum()),median_sampson_px=float(np.median(dist)) if len(dist) else None));print('PAIR',a,b,int(keep.sum()),flush=True)
components={}
for key in parent:components.setdefault(find(key),[]).append(key)
points=[];colors=[];tracks=[];rejected=dict(fewer_than3views=0,duplicate_camera=0,reprojection_or_depth=0,small_ray_angle=0)
for track in components.values():
    if len(track)<3:rejected['fewer_than3views']+=1;continue
    if len({cam for cam,_ in track})!=len(track):rejected['duplicate_camera']+=1;continue
    A=[];observations=[]
    for cam,i in track:
        xy=features[cam][0][i];w2c=np.array(cameras[cam]['w2c']);K=np.array(cameras[cam]['K']);u=np.linalg.solve(K,np.r_[xy,1]);A.extend([u[0]*w2c[2]-w2c[0],u[1]*w2c[2]-w2c[1]]);observations.append((cam,i,xy,w2c,K))
    _,_,vh=np.linalg.svd(np.array(A));q=vh[-1]
    if abs(q[3])<1e-12:rejected['reprojection_or_depth']+=1;continue
    X=q[:3]/q[3];errs=[];depths=[];rays=[];cols=[]
    for cam,i,xy,w2c,K in observations:
        xc=w2c[:3,:3]@X+w2c[:3,3];uv=K@xc;errs.append(float(np.linalg.norm(uv[:2]/uv[2]-xy)));depths.append(float(xc[2]));ray=X-np.array(cameras[cam]['camera_center_world']);rays.append(ray/np.linalg.norm(ray));px=np.rint(xy).astype(int);cols.append(images[cam][px[1],px[0]])
    if not np.isfinite(X).all() or max(errs)>1.5 or min(depths)<=.1 or max(depths)>=20:rejected['reprojection_or_depth']+=1;continue
    angle=float(np.degrees(np.arccos(np.clip(np.min(np.array(rays)@np.array(rays).T),-1,1))))
    if angle<1:rejected['small_ray_angle']+=1;continue
    points.append(X);colors.append(np.median(cols,axis=0)/255);tracks.append(dict(observations=[dict(camera=cam,feature=i,xy=xy.tolist()) for cam,i,xy,_,_ in observations],errors_px=errs,max_ray_angle_deg=angle,depths=depths))
assert len(points)>500,'Insufficient reliable training-only background geometry; do not train blindly'
points=np.array(points,dtype='float32');colors=np.array(colors,dtype='float32');np.savez_compressed(out/'background-init.npz',points=points,colors=colors);result=dict(status='training_camera_background_candidate_geometry_complete_not_trained',points=len(points),sources=hashes,train_cameras=train,heldout_images_read=0,source_geometry_used=False,calibration_sha256={n:digest(root/'optimized'/n) for n in ['intri.yml','extri.yml']},pairs=checks,track_components=len(components),rejected=rejected,tracks=tracks,bounds_world=[points.min(0).tolist(),points.max(0).tolist()],coordinate_scope='actor5_6 optimized camera world only; no unverified merge with actor1_4 foreground coordinates',limits=['Known-camera triangulation, not full SfM/pose estimation','SIFT matching and released calibration may still contain outliers; reprojection checks and visual review required','No model, heldout rendering or joint foreground/background training yet'])
(out/'triangulation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=result['status'],points=len(points),rejected=rejected),indent=2))
