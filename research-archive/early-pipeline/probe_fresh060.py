from pathlib import Path
import numpy as np,json
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-clothing-pipeline-fresh-060'
z=np.load(R/'data/observations.npz')
print('fields', {k:list(z[k].shape) for k in z.files})
print('scores body/face/hand', [np.quantile(z['scores'][:,s],[.1,.5,.9]).tolist() for s in [slice(0,17),slice(23,91),slice(91,133)]])
print('flip body px',np.quantile(z['flip_disagreement'][:,:17],[.1,.5,.9,.99]).tolist())
print('decoded/excluded', {k:v for k,v in json.loads((R/'data/inputs.json').read_text()).items() if k in ['decoded_total_frames','excluded_unbracketed_frames']})
z=np.load(B/'current/smplx-observation-support-014/surface-support.npz')
print('original 3d support',{k:list(z[k].shape) for k in z.files})
