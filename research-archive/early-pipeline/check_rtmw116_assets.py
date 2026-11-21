"""Verify the existing whole-body detector with isolated CPU dependencies."""
from pathlib import Path
import ast, datetime as dt, hashlib, json, sys, zipfile
import numpy as np
sys.path.insert(0,'/opt/gs-slam-lab/shuolee-hsfm116/python')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
import onnxruntime as ort
from rtmlib.visualization.skeleton.coco133 import coco133
from rtmlib import RTMPose
assert json.loads((R/'cpu-setup-result.json').read_text())['exit_code']==0
path=Path('/mnt/d/GS-SLAM-Data/models/rtmw/rtmw-l-384x288.onnx')
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
sess=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
inp=[dict(name=x.name,shape=x.shape,type=x.type) for x in sess.get_inputs()]
out=[dict(name=x.name,shape=x.shape,type=x.type) for x in sess.get_outputs()]
assert inp[0]['shape'][-3:]==[3,384,288]
assert [x['name'] for x in out]==['simcc_x','simcc_y']
assert [x['shape'][-1] for x in out]==[576,768]
# Export stores the keypoint axis as a symbolic dimension. Verify concrete
# output sizes with a zero image on CPU instead of guessing the symbol value.
zero_outputs=sess.run(None,{inp[0]['name']:np.zeros((1,3,384,288),dtype=np.float32)})
assert [list(x.shape) for x in zero_outputs]==[[1,133,576],[1,133,768]]
assert all(np.isfinite(x).all() for x in zero_outputs)
names=[coco133['keypoint_info'][i]['name'] for i in range(133)]
tree=ast.parse((R/'source/joint_names.py').read_text()); source_names={}
def constant_value(n):
    if isinstance(n,ast.List):return ast.literal_eval(n)
    if isinstance(n,ast.Name):return source_names[n.id]
    if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):return constant_value(n.left)+constant_value(n.right)
    raise ValueError(type(n).__name__)
for n in tree.body:
    if isinstance(n,ast.Assign) and isinstance(n.value,(ast.List,ast.BinOp)):
        for t in n.targets:
            if isinstance(t,ast.Name):source_names[t.id]=constant_value(n.value)
assert names[:23]==source_names['COCO_WHOLEBODY_KEYPOINTS'][:23]
assert all(n in source_names['SMPL_45_KEYPOINTS'] for n in names[:23])
with zipfile.ZipFile(path.with_suffix('.zip')) as z:pipeline=json.loads(z.read('pipeline.json'))
transforms=pipeline['pipeline']['tasks'][0]['transforms']
norm=next(v for v in transforms if v['type']=='Normalize')
# Reuse official decoding; no model predictions. One peak per known channel
# verifies SimCC x/y order and its two-bin-per-network-pixel convention.
class Detector(RTMPose):
    def __init__(self):
        self.model_input_size=(288,384);self.mean=np.array(norm['mean']);self.std=np.array(norm['std'])
        self.onnx_model=str(path);self.session=sess
        self.backend,self.device,self.to_openpose='onnxruntime','cpu',False
d=Detector()
x=np.zeros((1,133,576),dtype=np.float32);y=np.zeros((1,133,768),dtype=np.float32)
x[:,:,240]=1;y[:,:,560]=1
center=np.array([320.,470.]);scale=np.array([576.,768.])
decoded,scores=d.postprocess([x,y],center,scale)
expected=np.array([272.,646.])
assert np.max(abs(decoded[0]-expected))<1e-5
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='passed_cpu_graph_body23_and_simcc_contract',
    path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),providers=sess.get_providers(),
    inputs=inp,outputs=out,concrete_zero_image_shapes=[list(x.shape) for x in zero_outputs],
    normalization=norm,body23_names=names[:23],
    body23_to_smpl45=[source_names['SMPL_45_KEYPOINTS'].index(n) for n in names[:23]],
    synthetic_decode_expected=expected.tolist(),synthetic_decode_actual=decoded[0,0].tolist(),
    full_face133_material_mapping_verified=False,score_is_calibrated_probability=False,
    subject_inference=False,old_subject_fit_loaded=False,gpu=False,
    disclosure='RTMW substituted for HSfM ViTPose. SameCOCO body23 schema; labels are detector estimates, not measured material points.')
(R/'rtmw-compatibility.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
a=json.loads((R/'asset-compatibility.json').read_text());a['findings']['rtmw']=result
a['findings']['rtmw_deps']={'onnxruntime':True,'rtmlib':True,'isolated_target':'/opt/gs-slam-lab/shuolee-hsfm116/python'}
a['updated']=result['updated'];(R/'asset-compatibility.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result),flush=True)
