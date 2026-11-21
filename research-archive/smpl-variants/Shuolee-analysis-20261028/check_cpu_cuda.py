from pathlib import Path
import fcntl,json,os,sys
import cv2
sys.path.insert(0,'/mnt/d/Projects/Shuolee/src')
from shuolee_portrait import runtime as rt
rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
from shuolee_portrait.observations import load_detector
from shuolee_portrait.video119 import _cpu_person_boxes
import torch
image=cv2.imread('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de/raw/frame_000001.png')
assert image is not None
guarded='--unguarded' not in sys.argv
report=dict(guarded=guarded,visible_before=os.environ.get('CUDA_VISIBLE_DEVICES'))
with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    detector=load_detector()
    if guarded:
        boxes=_cpu_person_boxes(detector,image)
    else:
        boxes=detector.predict(image,classes=[0],conf=.25,verbose=False,device='cpu')[0].boxes.xyxy.cpu().numpy()
    report.update(boxes=boxes.tolist(),visible_after=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        x=torch.ones((32,32),device='cuda',requires_grad=True)
        (x*x).sum().backward()
        report['cuda_backward']=bool(torch.isfinite(x.grad).all())
        if guarded:
            from sam2.build_sam import build_sam2
            model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(rt.ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
            report['sam2_cuda_loaded']=next(model.parameters()).is_cuda
    except RuntimeError as exc:
        report['cuda_backward']=False
        report['error']=str(exc)
    fcntl.flock(lock,fcntl.LOCK_UN)
path=Path('/mnt/d/Projects/Shuolee-analysis-20261028')/('cpu-cuda-fixed.json' if guarded else 'cpu-cuda-original.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
assert report['cuda_backward']==guarded
