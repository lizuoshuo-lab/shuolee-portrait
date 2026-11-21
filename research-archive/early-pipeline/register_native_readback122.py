from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).resolve().parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-capacity/readback-diagnosis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
assert read(R/'native-capacity/result.json')['state']=='failed'
assert not O.exists();O.mkdir()
sources=read(R/'native-capacity/plan.json')['input_sha256']
for n in ['scene.json','point_cloud.ply','cameras.json']:
 p=R/'native-capacity/portrait-native-capacity/package'/n;sources[str(p)]=sha(p)
p=R/'native-capacity/portrait-native-capacity/native-float-response.npz';sources[str(p)]=sha(p)
plan=dict(stage='T7b_saved_parameter_readback',registered=datetime.now(timezone.utc).isoformat(),
 hypothesis='T7 export response differs despite tiny attribute deltas. Isolate constant-package reproducibility and exact-array NPZ from one additional PLY quantization cycle; no optimizer.',
 intervention='At same recorded native1520x2704/200k camera, render saved T7package once and compare its savedfloatresponse; render bit-identical NPZarrays; serialize savedparameters once through unchangedPLYcodec and render. Exactly3 forward renders, no backward/training.',
 limits=dict(wall_seconds=120,forward_renders=3,optimizer_steps=0),
 controls='Fullarrays and both floatimages retained before analysis. SameK/E/kernel/pointcount/resolution. Attribute differences/counts and fullframe deltas reported without threshold filtering.',
 stopping='Stop any failure/OOM/time limit, no fallback or repeat; do not loosen T7 acceptance threshold.',
 scope='OriginalT7 pre-export updated parameters and full reference were not saved. This follow-up cannot uniquely attribute that first mismatch; preserve that limitation.',
 source_sha256=sources,driver_sha256=sha(H/'probe_native_capacity_readback122.py'),
 physical_training_admitted=False)
(O/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
old=(H/'run_native_capacity122.sh').read_text(encoding='utf-8')
new=old.replace('probe_native_capacity122.py','probe_native_capacity_readback122.py').replace("native-capacity/run.log'","native-capacity/readback-diagnosis/run.log'")
(H/'run_native_capacity_readback122.sh').write_text(new,encoding='utf-8',newline='\n')
print('T7B_REGISTERED_NO_GPU_EXECUTION')
