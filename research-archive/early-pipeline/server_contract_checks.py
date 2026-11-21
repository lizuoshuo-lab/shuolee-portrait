from pathlib import Path
import sys,os,json,tempfile,importlib.util,hashlib
P=Path('/mnt/d/00/工作/人体重建');sys.path.insert(0,str(P/'src'))
from shuolee_portrait import runtime as rt
rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
import numpy as np
from scipy.spatial.transform import Rotation
from shuolee_portrait.training import camera
from shuolee_portrait.export import rotate_ply
from plyfile import PlyData,PlyElement
from shuolee_portrait.cli import validate_config
import importlib.metadata as metadata

checks={}
E=np.eye(4);E[:3,:3]=Rotation.from_euler('xyz',[.3,-.4,.1]).as_matrix();E[:3,3]=[.2,-.1,3]
row=dict(frame='synthetic',wh=[731,1297],K=[[980,0,351.4],[0,987,624.7],[0,0,1]],w2c=E.tolist())
cam,receipt=camera(row,0,768);checks['actual_camera_projection']=receipt
rng=np.random.default_rng(15);n=37
fields=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
data=np.zeros(n,dtype=[(k,'<f4') for k in fields])
for k in fields:data[k]=rng.normal(size=n)
for i in range(3):data[f'scale_{i}']=rng.uniform(-5,-3,n)
q=Rotation.random(n,random_state=rng).as_quat()[:,[3,0,1,2]]
for i in range(4):data[f'rot_{i}']=q[:,i]
with tempfile.TemporaryDirectory(prefix='shuolee-contract-') as temp:
 d=Path(temp);src=d/'source.ply';dst=d/'rotated.ply';PlyData([PlyElement.describe(data,'vertex')]).write(src)
 R=Rotation.from_euler('xyz',[.4,.8,-.3]).as_matrix();checks['synthetic_ply_transform']=rotate_ply(src,dst,R,np.array([3.,4.,5.]))
 loaded=PlyData.read(dst)['vertex'].data
 # Coefficients, not only basis math, must predict equal directional colors.
 def coeff(v):return np.concatenate([np.column_stack([v[f'f_dc_{i}'] for i in range(3)])[:,None,:],np.column_stack([v[f'f_rest_{i}'] for i in range(45)]).reshape(-1,3,15).transpose(0,2,1)],axis=1)
 from utils.sh_utils import eval_sh
 dirs=rng.normal(size=(n,3));dirs/=np.linalg.norm(dirs,axis=1)[:,None]
 a=eval_sh(3,coeff(data).transpose(0,2,1),dirs@R);b=eval_sh(3,coeff(loaded).transpose(0,2,1),dirs)
 checks['actual_SH_color_rotation_max']=float(abs(a-b).max());assert checks['actual_SH_color_rotation_max']<1e-6
 spec=importlib.util.spec_from_file_location('restore',P/'scripts/restore_weights.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 bundle=d/'parts';bundle.mkdir();payload=b'weight_bytes_not_a_real_checkpoint'*131
 parts=[]
 for i,start in enumerate(range(0,len(payload),1000)):
  block=payload[start:start+1000];p=bundle/f'{i}.part';p.write_bytes(block);parts.append(dict(path=p.name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
 manifest={'files':[dict(target='model/weight.bin',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),parts=parts)]}
 (bundle/'manifest.json').write_text(json.dumps(manifest));m.restore(bundle,d/'assets');assert (d/'assets/model/weight.bin').read_bytes()==payload
 (bundle/'0.part').write_bytes(b'damaged');caught=False
 try:m.restore(bundle,d/'other')
 except RuntimeError:caught=True
 assert caught;checks['split_restore_and_corruption_rejection']=True
checks['versions']={}
for name in ['torch','torchvision','pytorch-lightning','hydra-core','hydra-zen','timm','einops','roma','yacs','rtmlib','onnxruntime','iopath','joblib','rich','hydra-colorlog']:
 try:checks['versions'][name]=metadata.version(name)
 except metadata.PackageNotFoundError:checks['versions'][name]='not_discoverable'
checks['quality_accepted']=False
rt.save(P/'provenance/server-contract-validation.json',checks)
print(json.dumps(checks,indent=2))
