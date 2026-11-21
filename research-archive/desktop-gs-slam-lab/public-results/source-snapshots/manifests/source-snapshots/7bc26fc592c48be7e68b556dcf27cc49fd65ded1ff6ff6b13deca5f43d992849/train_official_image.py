"""Run gsplat v1.5.3's unmodified SimpleTrainer on one public TUM RGB image."""
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from gsplat import rasterization

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'official-image-fit-001'
OUT.mkdir(parents=True, exist_ok=False)
source = ROOT / 'repos' / 'gsplat' / 'examples' / 'image_fitting.py'
spec = importlib.util.spec_from_file_location('gsplat_official_image_fitting', source)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
image_path = sorted((ROOT / 'data' / 'tum' / 'rgbd_dataset_freiburg1_desk' / 'rgb').glob('*.png'))[0]
with Image.open(image_path) as original:
    target = np.array(original.convert('RGB').resize((128, 96), Image.Resampling.LANCZOS), dtype=np.float32) / 255.
torch.manual_seed(0)
trainer = official.SimpleTrainer(torch.from_numpy(target), num_points=2000)
K = torch.tensor([[trainer.focal, 0, trainer.W / 2], [0, trainer.focal, trainer.H / 2], [0, 0, 1]], device='cuda')[None]

def render():
    return rasterization(trainer.means, torch.nn.functional.normalize(trainer.quats, dim=-1),
                         trainer.scales, trainer.opacities.sigmoid(), trainer.rgbs.sigmoid(),
                         trainer.viewmat[None], K, trainer.W, trainer.H, packed=False)[0][0]

with torch.no_grad():
    before = render().cpu().numpy()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
trainer.train(iterations=500, lr=.01, save_imgs=False, model_type='3dgs')
torch.cuda.synchronize()
seconds = time.perf_counter() - started
with torch.no_grad():
    after = render().cpu().numpy()
assert np.isfinite(after).all()
initial_mse = float(np.mean((before-target)**2))
final_mse = float(np.mean((after-target)**2))
images = [('target', target), ('initial', before), ('trained', after)]
panel = Image.new('RGB', (768, 224), 'white')
draw = ImageDraw.Draw(panel)
for index, (name, array) in enumerate(images):
    frame = Image.fromarray(np.round(np.clip(array, 0, 1)*255).astype(np.uint8))
    frame.save(OUT / f'{name}.png')
    panel.paste(frame.resize((256, 192)), (index*256, 28))
    draw.text((index*256+8, 8), name, fill='black')
panel.save(OUT / 'comparison.png')
result = {'status':'completed', 'scope':'official single-image Gaussian fitting; not 3D reconstruction or SLAM',
          'source':'https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/image_fitting.py',
          'dataset':'TUM RGB-D fr1_desk; CC BY 4.0', 'source_image':image_path.name,
          'seed':0, 'iterations':500, 'gaussians':2000, 'resolution':[128,96],
          'initial_mse':initial_mse, 'final_mse':final_mse,
          'initial_psnr_db':-10*math.log10(initial_mse), 'final_psnr_db':-10*math.log10(final_mse),
          'training_seconds_excludes_compile':seconds,
          'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
          'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__}
torch.save({name:getattr(trainer,name).detach().cpu() for name in ['means','scales','quats','rgbs','opacities']},OUT/'gaussians.pt')
(OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
