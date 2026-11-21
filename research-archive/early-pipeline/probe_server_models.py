from pathlib import Path
import sys,json
P=Path('/mnt/d/00/工作/人体重建')
sys.path.insert(0,str(P/'src'))
from shuolee_portrait import runtime as r
r.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
import torch
for name in ['detection/yolov8n.pt','gvhmr/hmr2/epoch=10-step=25000.ckpt','gvhmr/vitpose/vitpose-h-multi-coco.pth','gvhmr/gvhmr/gvhmr_siga24_release.ckpt']:
 print(json.dumps({'file':name,'unsafe_globals':torch.serialization.get_unsafe_globals_in_checkpoint(r.ASSETS/name)}),flush=True)
from shuolee_portrait.observations import load_detector
model=load_detector()
print('Pinned detector constructed with scoped safe globals.',flush=True)
