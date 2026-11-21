from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2
from scipy.spatial import cKDTree
from audit_enerf_input import read_cameras
LAB=Path(__file__).resolve().parents[2];SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';MOTION=LAB/'outputs/dynamic-human-enerf-continuous-body-063';OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');OUT.mkdir(exist_ok=True)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert not (OUT/'input-audit.json').exists()
save(OUT/'protocol.json',dict(status='fixed_before_training',branches=['fixed_offsets','estimated_offsets'],steps_per_branch=15000,
 training='15cameras x7evenframes744..756=105originalnativeimages; no00/06/12photos. Sharedoriginalreleasedforegroundmatte; noSAM750maskreusedatothertimes.',
 evaluation='Same15cameras x6oddframes745..755=90native-timevalidationimages; plus12even-time trainingdiagnostics. Oddposespreviouslyexaminedasdevelopment, notfreshfinaltest.',
 model='Canonicalanisotropic3DGaussians, fixednearestSMPLvertexbindings inheritedbydensityoperations, continuousbodyaffinetransformsincludingposecorrectivevertexmotion, exactcovariancepushforward, canonicaldirectionSH2.',
 initialization='Identical050frame750train-onlyvisualhull52109points/colors,scale0.009,identityquaternion,opacity0.1,zeroSHhigherbands; same060referencebody andbindings.',
 intervention='Only063continuousmotionandcapturetimehypothesis changes; corresponding3Dtrajectoryfit changeswithtimehypothesis. SameRGBs/masks/optimizer/density/budget/draws. Noimagereindexing/interpolation/sharpening.',
 optimization='0.8RGBL1+0.2DSSIM+0.1matteL1, randomuniformbackground. Nativepersonboundingcrop32pxmargin withtranslatedK; fullnativeevaluation. Adam andDefaultStrategy density400..9900, resets3000, refine100; pointcap800000.',
 limitations='SMPLbindings approximatemotionofclothes/hair/heldobjects, noindependenttoy orfingerarticulation. Poseandtimefrozen. Unknownheldoutcamera clockspreventclaimingold00/06/12synchronizedevaluation. Fullbackgroundnotmodeled.',goal_complete=False))
init=np.load(LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame/train-visual-hull-init.npz');ref=np.load(MOTION/'reference.npz');verts=ref['vertices'].reshape(-1,3);dist,ids=cKDTree(verts).query(init['points'],workers=4)
np.savez_compressed(OUT/'initial.npz',points=init['points'],colors=init['colors'],anchor=ids.astype(np.int64),anchor_distance=dist)
save(OUT/'binding-audit.json',dict(initial_points=len(ids),anchor_count=len(verts),distance_quantiles_world=np.quantile(dist,[0,.5,.9,.99,1]).tolist(),person_counts=[int((ids//6890==i).sum()) for i in range(2)],scope='Nearestvertexbinding, notsemanticinstanceGT; toys/hair/clothescanbindtoapproximatebodyregion. Geometryoffsetsremaintrainable.',goal_complete=False))
cal=read_cameras(RAW/'optimized');rows=[];inputdir=OUT/'native-input';inputdir.mkdir(exist_ok=True)
source=read_cameras(RAW/'optimized');recorded=json.loads((SOURCE/'native-input/input-audit.json').read_text())['records']
for r in recorded:
 c,t=r['camera'],r['frame'];native=SOURCE/'native-input'/f'camera{c}-frame{t:06d}.npz';rgb=np.load(native)['rgb'];K=np.array(cal[c]['K']);D=np.array(cal[c]['D']);W=np.array(cal[c]['w2c']);maskfile=RAW/'bgmtv2'/c/f'{t:06d}.jpg'
 mask=cv2.imread(str(maskfile),0).astype(np.float32)/255;matte=cv2.undistort(mask,K.astype(np.float32),D.astype(np.float32));mx,my=cv2.initUndistortRectifyMap(K.astype(np.float32),D.astype(np.float32),None,K.astype(np.float32),(1920,1080),cv2.CV_32FC1);valid=(mx>=0)&(mx<1919)&(my>=0)&(my<1079)
 ys,xs=np.where((matte>.5)&valid);bbox=[max(0,int(xs.min())-32),max(0,int(ys.min())-32),min(1920,int(xs.max())+33),min(1080,int(ys.max())+33)]
 path=inputdir/f'camera{c}-frame{t:06d}.npz';np.savez_compressed(path,rgb=rgb,matte=matte,valid=valid,K=K,w2c=W,bbox=np.array(bbox))
 rows.append(dict(camera=c,frame=t,split='train' if t%2==0 else 'validation_time',source_rgb_sha256=r['source_sha256'],source_matte_sha256=sha(maskfile),rectified_rgb_sha256=r['rectified_sha256'],input_sha256=sha(path),bbox=bbox))
 if t==756:print('ARTICULATED_INPUT',c,flush=True)
assert sum(r['split']=='train' for r in rows)==105 and sum(r['split']=='validation_time' for r in rows)==90
save(OUT/'input-audit.json',dict(status='195native_inputs_prepared_105train90timevalidation',records=rows,goal_complete=False));shutil.copy2(__file__,OUT/Path(__file__).name)
