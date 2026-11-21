"""Zero-update full-size capacity probe on a read-only completed baseline."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

RUN = Path('/opt/gs-slam-lab/msv2-video15-20261105/runs/video15')
sys.path.insert(0, str(RUN / 'pipeline-source'))
from msv2.engine import Engine, LazyCamera
from msv2.io import read_json, save_json, sha256, now


def main():
    out = RUN / 'recovery/full4-capacity.json'
    assert not out.exists()
    engine = Engine(RUN / 'job.json')
    torch = engine.runtime()
    torch.cuda.set_per_process_memory_fraction(engine.config['gpu_memory_fraction'])
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render_imp
    from fused_ssim import fused_ssim
    from utils.loss_utils import l1_loss
    model_file = RUN / 'original/model/point_cloud/iteration_18000/point_cloud.ply'
    before = sha256(model_file)
    model = GaussianModel(3)
    model.load_ply(str(model_file))
    model.active_sh_degree = 3
    _, _, opt, pipe = engine.options(4)
    model.training_setup(opt)
    moments = [torch.zeros_like(p) for group in model.optimizer.param_groups for p in group['params'] for _ in range(2)]
    padding = torch.zeros(512 * 1024**2, dtype=torch.uint8, device='cuda')
    rows = read_json(RUN / 'swinir4x/training-inputs.json')['records']
    state = dict(state='running', started=now(), diagnostic_only=True,
        model_updates=0, optimizer_updates=0, trained_initializer=False,
        model_source=str(model_file), model_sha256=before, points=len(model.get_xyz),
        manual_adam_moment_bytes=sum(t.numel()*t.element_size() for t in moments),
        additional_padding_bytes=padding.numel(), views=[])
    save_json(out, state)
    try:
        cameras = [LazyCamera(rows[i], i) for i in (0, len(rows)//2, len(rows)-1)]
        render_pkg = image = gt = l1 = ssim_value = loss = None
        bg = torch.zeros(3, device='cuda')
        for cam in cameras:
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            render_pkg = render_imp(cam, model, pipe, bg)
            image = render_pkg['render']
            gt = cam.original_image
            l1 = l1_loss(image, gt)
            ssim_value = fused_ssim(image[None], gt[None])
            loss = .8*l1+.2*(1-ssim_value)
            loss.backward()
            torch.cuda.synchronize()
            assert torch.isfinite(loss) and torch.isfinite(model._xyz.grad).all()
            item = dict(frame=cam.image_name, wh=[cam.width,cam.height], loss=float(loss),
                seconds=time.monotonic()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            torch.cuda.empty_cache()
            model.optimizer.zero_grad(set_to_none=True)
            state['views'].append(item)
            save_json(out,state)
            print('FULL4_CAPACITY',json.dumps(item),flush=True)
        assert sha256(model_file)==before
        state.update(state='passed',finished=now(),model_file_unchanged=True,
            scope='Completed baseline read-only capacity proxy; actual fresh 4x full-stage training is still required')
        save_json(out,state)
    except BaseException as error:
        state.update(state='failed',error=repr(error),finished=now())
        save_json(out,state)
        raise


if __name__=='__main__':main()
