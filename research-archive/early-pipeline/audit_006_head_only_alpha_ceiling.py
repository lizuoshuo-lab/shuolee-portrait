"""Test frozen stock geometry with an opacity ceiling on strict head rows only."""

import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_unified_face_opacity import preflight, sheet
from review_bound_alpha_ab import read, require, sha
from review_bound_scale_coverage import face_alpha, scale_pixels
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, save


def main():
    saved, pose, row, near_row, native, near, native_crop, crop, _, identity = preflight('006')
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    import torch

    output = LAB/'shuolee-smpl-unified-006-strict-head-alpha-ceiling'
    output.mkdir(exist_ok=False)
    with np.load(ASSET, allow_pickle=False) as asset:
        protected = np.isin(asset['weights'].argmax(axis=1),[12,15,22,23,24])
        lmk = np.concatenate((asset['lmk_faces_idx'].ravel(),asset['dynamic_lmk_faces_idx'].ravel()))
    faces = saved.fit['faces']
    protected[faces[lmk].ravel()] = True
    group = protected[faces].all(axis=1)
    original_path = LAB/'shuolee-smpl-unified-006-face-opacity-audit/result.json'
    original = read(original_path)
    plan = dict(identity, control='strict protected head opacity floor; no body opacity change',
                all_points_ceiling_receipt_sha256=sha(original_path), ceiling_code_sha256=sha(Path(__file__)))
    save(output/'plan.json',plan)
    rt.configure(ASSET.parents[1])
    from gaussian_renderer import render

    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count()==1, 'Expose one CUDA GPU.')
            model, parity = _load_model(saved.snapshot/'bound-local.ply',saved.snapshot,saved.fit,
                                        saved.mesh,saved.facts,ASSET)
            model.select_mesh_by_timestep(pose)
            head = torch.as_tensor(group,device='cuda')[model.binding.long()]
            raw = model._opacity.clone()
            control = raw.clone()
            control[head] = raw[head].clamp_min(math.log(999.))
            require(head.any() and torch.equal(raw[~head],control[~head]), 'Body opacity changed in ceiling control.')
            pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
            black,white = torch.zeros(3,device='cuda'),torch.ones(3,device='cuda')
            rows=[]
            for name, view, roi, box in (('native',row,native,native_crop),('dolly',near_row,near,crop)):
                cam, receipt = camera(view,0,pose_index=pose)
                def raster():
                    return scale_pixels(render,render_pose,cam,model,pipe,black,white,pose,1.)[0]
                before = raster()
                prior = next(r for r in original['rows'] if r['camera']==name)
                stats = face_alpha(before[2],roi)
                require(stats == prior['stock'], 'Stock alpha differs from the registered ceiling reference.')
                try:
                    model._opacity.copy_(control)
                    high = raster()
                finally:
                    model._opacity.copy_(raw)
                require(model.optimizer is None and torch.equal(model._opacity,raw)
                        and all(np.array_equal(a,b) for a,b in zip(before,raster(),strict=True)),
                        'Frozen stock state did not restore exactly.')
                rows.append(dict(camera=name, camera_receipt=receipt, stock=stats,
                                 head_only=face_alpha(high[2],roi),all_points=prior['opacity_floor'],
                                 panel=sheet(output/f'{name}.png',box,before,high)))
    report=output/'result.json'
    save(report,dict(plan,state='completed',head_owned=int(head.sum()),opacity_entries_raised=int((control!=raw).sum()),
                     geometry_readback=parity,rows=rows,restored_exact=True,optimizer_updates=0))
    print(json.dumps(dict(result=str(report),sha256=sha(report))))


if __name__ == '__main__':
    main()
