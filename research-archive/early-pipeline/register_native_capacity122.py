"""Register the actual native-resolution engineering gap without admitting real data."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-capacity';G=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if O.exists():assert {p.name for p in O.iterdir()}=={'registration-before-fix.py','registration-failure.json'}
assert read(R/'queue-state.json')['running'] is None
old=read(R/'native-renderer-v2/review.json');assert old['large']['points']==200000 and old['large']['full_resolution_capacity_tested'] is False
assert read(R/'native-renderer-v2/viewer-ui-review.json')
inputs=[R/'native-training-integration/subpixel-culling-diagnosis/frozen-large-arrays.npz',R/'native-training-integration/subpixel-culling-diagnosis/camera.json',R/'native-renderer-v2/plan.json']
with np.load(inputs[0]) as z:
 assert z['means'].shape==(200000,3);sh=z['sh'].shape
 assert sh==(200000,16,3)
inputs+=[H/n for n in ['native_portrait_renderer122_v2.py','native_scene_package122_v2.py','native_training122_v2.py','native_gaussian_ply122.py','native_portrait_renderer122.py','native_training122.py']]
inputs+=[G/n for n in ['gsplat/rendering.py','gsplat/cuda/_wrapper.py','gsplat/cuda/csrc/ProjectionUT3DGSFused.cu','gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu','gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu']]
O.mkdir(exist_ok=True);registered_at=datetime.now(timezone.utc).isoformat()
plan=dict(stage='T7_native_resolution_capacity',registered=registered_at,
 gap='Current verified200k forward/backward/update used64x64. Need native-size feasibility, complete attributes/cameras readback and bounded memory/time. This is required independent engineering, not a substitute for physical alignment.',
 intervention='Use unchangedv2adapter/kernel with the same frozen200k synthetic input arrays, two explicitly known synthetic full-size cameras. Fresh parameter copy and empty optimizer for each orientation, exactlyone update. No realhuman/bodyfit orrealcamera.',
 cameras=[dict(id='portrait-native-capacity',width=1520,height=2704,K=[[1494.4425475491018,0,761.3168340952368],[0,1493.0941798544786,1357.2026900985354],[0,0,1]]),
          dict(id='landscape-native-capacity',width=2704,height=1520,K=[[1494.4425475491018,0,1407.375],[0,1493.0941798544786,738.625],[0,0,1]])],
 camera_semantics='Known synthetic pinhole cameras; portrait numericalK equals conditionalsourceK as a fixture only, landscape principalpoint intentionallyoffcenter. Use frozen synthetic w2c. These are not registeredMP4cameras or physicalcalibration.',
 renderer=dict(adapter='122-ray3d-geometry-bounds-v2',projection='pinhole',gaussianEvaluation='ray-3d',bounds='ut-3.33-sigma-with-eps2d-no-opacity-cull',sh_degree=3,changed=False),
 input_sha256={str(p):sha(p) for p in inputs},optimizer=read(R/'native-renderer-v2/plan.json')['optimizer'],
 controls=['For eachorientation: native-resolutionRGB/alpha finite/nonempty, fixedfullK/w2cunchanged,finitegradientstoallattributeclasses,nonzeroactualcenterupdateswithfreshAdam.',
 'RecordCUDAallocated/reservedpeaks andactualelapsedforward/backwardtime. No lowerresolutionorpointcount fallback.',
 'Exportfullcovariance/SHPLYandSHA-boundscene+cameras usingv2 unchangedcodec. Render recordedcamera andcomparefullfloatRGB/alpha to directupdatedparameters.',
 'Save actual fullsize images forreview. No comparisonagainstupscaled64x64 orclaimofphysicalquality.',
 'Preserveoldcode/inputs/packages/livev1/v2services and119preview bysourcehash;no temporarythresholdorpixelconventionchanges.'],
 limits=dict(wall_seconds=300,orientations=2,points=200000,optimizer_steps_per_orientation=1,roundtrip_max_abs=2e-5,GPU_jobs=1),
 execution='Before running inspect realPID/log,00coordination andGPUcomputeapps. Hold existing /tmp/shuolee-native122-gpu.lock via flock for wholejob. Record new driver hash before importing CUDA. If existing progress/result exists inspect; never duplicate.',
 stop='AnyOOM,timeout,nonfinitevalue,roundtripfailureordiskissue:saveexactterminalfailure,nosequentialautomaticretrieswithfewerpoints/resolution orchangedkernel. Repair demonstratedcodebugs only. Stop GPUafterboundedrun.',
 scope='Componentcapacityandexportonly. Doesnotadmitanyperson,proveboundsforarbitraryscenes,wideanglefisheyeaccuracy,18ktrainingcapacity,physicalsubpixelaccuracyormulti-persongeneralization.',
 physical_training_admitted=False,source_images_or_K_changed=False)
save(O/'plan.json',plan)
print(json.dumps(dict(registered=True,path=str(O/'plan.json'),running=False,point_count=200000,shape=sh),ensure_ascii=False))
