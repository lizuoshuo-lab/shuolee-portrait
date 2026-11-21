"""Zero-update 4x native-loss capacity check; never initializes the 4x training run."""
from pathlib import Path
from types import SimpleNamespace
import copy,importlib.util,json,time
import numpy as np
from PIL import Image

SCRIPT=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/standard_msv2_demo.py')
spec=importlib.util.spec_from_file_location('experiment',SCRIPT);ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)

def main():
    torch=ex.runtime()
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render_imp
    from fused_ssim import fused_ssim
    native=ex.ROOT/'original';assert ex.read(native/'execution.json')['state']=='training_completed'
    ply=native/'model/point_cloud/iteration_18000/point_cloud.ply';before=ex.sha(ply)
    row=copy.deepcopy(ex.read(native/'training-inputs.json')['records'][0]);row['image']=str(ex.ROOT/'swinir4x/dataset/images'/row['name']);row['wh']=[10816,6080]
    row['K']=(np.diag([4.,4.,1.])@np.asarray(row['K'])).tolist()
    done={r['name']:r for r in ex.read(ex.ROOT/'swinir4x/superresolution.json')['records']}
    assert ex.sha(row['image'])==done[row['name']]['output_sha256']
    cache=ex.ROOT/'swinir4x/rgb-cache-1';cache.mkdir(exist_ok=True)
    cachefile=cache/(row['frame']+'.npy')
    if not cachefile.exists():
        with Image.open(row['image']) as im:np.save(cachefile,np.asarray(im))
    state=dict(started=ex.now(),scope='Capacity only: frozen 1x model, native 4x target, zero optimizer updates; not a 4x fit or quality result',
        model_sha256=before,optimizer_updates=0,model_unchanged=True,full_frame=True,wh=row['wh'])
    result=ex.ROOT/'native-4x-memory-probe.json';torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    try:
        g=GaussianModel(3);g.load_ply(str(ply));g.active_sh_degree=3
        args,data,opt,pipe=ex.options(4);g.training_setup(opt)
        # Account for the native sparse Adam's two moment buffers without an update.
        moments=[torch.zeros_like(parameter) for group in g.optimizer.param_groups for parameter in group['params'] for _ in range(2)]
        camera=ex.LazyCamera(row,0);output=render_imp(camera,g,pipe,torch.zeros(3,device='cuda'));image=output['render'];target=camera.original_image
        l1=(image-target).abs().mean();loss=.8*l1+.2*(1-fused_ssim(image[None],target[None]));loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and torch.isfinite(g._xyz.grad).all()
        state.update(state='passed',points=len(g.get_xyz),loss=float(loss),adam_moment_bytes=sum(m.numel()*m.element_size() for m in moments))
    except torch.cuda.OutOfMemoryError as error:
        state.update(state='memory_limit',error=str(error))
    finally:
        state.update(finished=ex.now(),seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        assert ex.sha(ply)==before;ex.save(result,state);print(json.dumps(state),flush=True)
if __name__=='__main__':main()
