"""Bounded numerical and timing check of TF32 for the unchanged SwinIR FP32 weights."""
import importlib.util, sys, time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
P=Path('/opt/gs-slam-lab/msv2-demo-fullframe-20261104')
s=importlib.util.spec_from_file_location('full',P/'experiment.py');ex=importlib.util.module_from_spec(s);s.loader.exec_module(ex)
torch=ex.runtime();sys.path.insert(0,str(P/'swinir-source'));import main_test_swinir as official
args=SimpleNamespace(task='real_sr',scale=4,large_model=True,model_path=str(ex.WEIGHT),tile=256,tile_overlap=32)
model=official.define_model(args).eval().cuda();rows=ex.read(P/'native-inputs.json')['records'];checks=[]
with torch.inference_mode():
    for i,(cx,cy) in zip([0,29,58],[(1400,650),(1300,720),(1000,900)]):
        rgb=np.array(Image.open(rows[i]['image']).convert('RGB'))[cy-128:cy+128,cx-128:cx+128]
        x=torch.from_numpy(rgb).to(device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255
        outputs=[];times=[]
        for enabled in [False,True]:
            torch.backends.cuda.matmul.allow_tf32=enabled;torch.backends.cudnn.allow_tf32=enabled
            t=time.monotonic();out=model(x);torch.cuda.synchronize();times.append(time.monotonic()-t);outputs.append(out.cpu().numpy())
        d=np.abs(outputs[0]-outputs[1]);checks.append(dict(frame=rows[i]['frame'],box=[cx-128,cy-128,cx+128,cy+128],max_abs=float(d.max()),mean_abs=float(d.mean()),p999=float(np.quantile(d,.999)),fp32_seconds=times[0],tf32_seconds=times[1]))
passed=all(c['mean_abs']<.0005 and c['max_abs']<.01 and c['p999']<.004 for c in checks)
ex.save(P/'swinir-tf32-probe.json',dict(passed=passed,checks=checks,weights_changed=False,weight_sha256=ex.sha(ex.WEIGHT),note='Numerical approximation only, not identical pixels. Not enabled by this probe.'))
print(checks,passed)
