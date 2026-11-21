"""Read-only native vs regressed head-point comparison on existing fits."""
from pathlib import Path
import json,hashlib
import numpy as np

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/smplx-face-surface-fresh-047';F=B/'current/smplx-native-face-observations-046'
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def quant(x):return {'n':int(x.size),'median_px':float(np.median(x)),'p95_px':float(np.quantile(x,.95))}
data=load(R/'control/initialization.npz');obs=load(B/'current/gaussianavatars-smplx-from-scratch-001/data/observations.npz')
fan=load(F/'face-observations.npz');m=load(F/'reviewed-support.npz')
assert obs['keypoints'].shape==(59,133,2)
names=obs['joint_names'].tolist()
assert names[23:26]==['face-0','face-1','face-2'],names[23:26]
res={'body_detector':'RTMW COCO-WholeBody133','head_points_native_joint_indices':[55,57,56,59,58],
     'native_names':['nose','left_eye','right_eye','left_ear','right_ear'],'is_fitting':False,'independent_GT':False}
res['rtmw_fan_surface_detector_difference']=quant(np.linalg.norm(obs['keypoints'][:,23:91]-fan['xy'],axis=-1)[m['selected_mask']])
res['rtmw_fan_nose_tip_difference']=quant(np.linalg.norm(obs['keypoints'][:,53]-fan['xy'][:,30],axis=-1)[fan['valid']])
res['rtmw_body_nose_vs_own_face_nose']=quant(np.linalg.norm(obs['keypoints'][:,0]-obs['keypoints'][:,53],axis=-1)[fan['valid']])
visibility=data['body_weights'][:,:5]>0;visibility[25:,3]=False;visibility[:18,4]=False
arms={}
for arm in ['control','face_surface']:
    f=load(R/arm/'fitted.npz');native=f['projected_joints'][:,[55,57,56,59,58]];target=data['original_keypoints'][:,:5]
    ne=np.linalg.norm(native-target,axis=-1);re=np.linalg.norm(f['projected_coco17'][:,:5]-target,axis=-1)
    arms[arm]={'native_head_error':quant(ne[visibility]),'regressed_head_error':quant(re[visibility]),
        'per_joint':{name:{'native':quant(ne[:,j][visibility[:,j]]),'regressed':quant(re[:,j][visibility[:,j]])} for j,name in enumerate(res['native_names'])}}
res['arms']=arms
res['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out=R/'diagnosis/head-definition-audit.json';assert not out.exists();out.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(res,ensure_ascii=False))
