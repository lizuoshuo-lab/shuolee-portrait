"""Read-only current-view colour controls for the published immutable 120k PLY.

No training changes. Diagnostic PLY copies change colour fields only.
"""
import json
import argparse
import fcntl
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from plyfile import PlyData
from audit_body_sh_directions123 import paths, ply, unit, stats, R, rt, eval_sh
from export_demo_body_snapshot123 import process


def main():
    snap, delivery, audit = paths(120000)
    out = audit / 'user-side-view'
    out.mkdir(exist_ok=False)
    camera_source = Path('/mnt/c/Users/Administrator/Downloads/local-surface-123-demo-body-smplx-offset-s120000_view.json')
    camera = rt.read(camera_source)[0]
    shutil.copy2(camera_source, out / 'camera.json')
    src_xyz, src_coeff = ply(snap / 'world.ply')
    xyz, coeff = ply(delivery / 'point_cloud.ply')
    eye = np.asarray(camera['position'])
    ray = unit(xyz-eye)
    colour = eval_sh(3, coeff, ray)+.5
    source_colour = eval_sh(3, src_coeff, unit(src_xyz-eye@R))+.5
    error = float(abs(colour-source_colour).max())
    assert error < 2e-5
    reference_eye = np.asarray(rt.read(delivery/'cameras.json')[0]['position'])
    reference_colour = np.maximum(eval_sh(3, coeff, unit(xyz-reference_eye))+.5, 0)
    # These are diagnostic copies only; preserve every position/covariance/
    # opacity/normal field exactly and verify on reread.
    original = PlyData.read(delivery/'point_cloud.ply')
    original_data = original['vertex'].data.copy()
    colour_fields = [n for n in original_data.dtype.names if n.startswith(('f_dc_', 'f_rest_'))]
    preserve = [n for n in original_data.dtype.names if n not in colour_fields]
    variants = []
    for label, replacement in [('dc-only', None), ('reference-colour', reference_colour), ('white-opacity', np.ones_like(colour))]:
        data = original['vertex'].data
        data[:] = original_data
        for i in range(45): data[f'f_rest_{i}'] = 0
        if replacement is not None:
            for i in range(3): data[f'f_dc_{i}'] = (replacement[:, i]-.5)/0.28209479177387814
        target = out/(label+'.ply')
        original.write(target)
        reread = PlyData.read(target)['vertex'].data
        assert all(np.array_equal(reread[n], original_data[n]) for n in preserve)
        assert all(np.all(reread[f'f_rest_{i}']==0) for i in range(45))
        variants.append(dict(label=label, path=str(target), sha256=rt.sha(target), only_colour_fields_changed=True))
    report = dict(updated=rt.now(), step=120000, points=len(xyz), camera=camera,
        source_ply_sha256=rt.sha(snap/'world.ply'), published_ply_sha256=rt.sha(delivery/'point_cloud.ply'),
        camera_sha256=rt.sha(out/'camera.json'), export_sh_exact_user_view_max_abs=error,
        colour_range=[float(colour.min()), float(colour.max())],
        negative_channel_fraction=float(np.mean(colour<0)), above_one_channel_fraction=float(np.mean(colour>1)),
        absolute_sh_high_order_contribution=stats(abs(colour-(coeff[:,:,0]*0.28209479177387814+.5))),
        statistics_scope='All Gaussian channels before compositing; not visible-pixel error or quality acceptance.',
        controls=variants, geometry_camera_opacity_unchanged=True, training_changed=False,
        promotion=False, quality_accepted=False, actual_ui_review='pending')
    rt.save(out/'result.json', report)
    print(json.dumps({k: v for k,v in report.items() if k not in ['camera','controls']}, indent=2))


def native(pid):
    import torch
    from PIL import Image
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.training import camera, image
    assert process(pid)[1] == 'T'
    snap, delivery, audit = paths(120000)
    out = audit/'user-side-view'
    record = rt.read(out/'result.json')
    assert not (out/'native-result.json').exists()
    assert rt.sha(delivery/'point_cloud.ply') == record['published_ply_sha256']
    assert rt.sha(snap/'world.ply') == record['source_ply_sha256']
    c = record['camera']
    E = np.eye(4); E[:3,:3] = np.asarray(c['rotation']).T; E[:3,3] = -E[:3,:3]@c['position']
    row = dict(frame='user-side-view', wh=[c['width'],c['height']],
        K=[[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1]], w2c=E.tolist())
    cam,_ = camera(row,0)
    source_E = E.copy(); source_E[:3,:3] = E[:3,:3]@R
    source_row = dict(row,w2c=source_E.tolist()); source_camera,_ = camera(source_row,0)
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    bg = torch.zeros(3,device='cuda'); images=[]
    with torch.no_grad():
        for label,path in [('full-sh3',delivery/'point_cloud.ply')] + [(v['label'],Path(v['path'])) for v in record['controls']]:
            model = GaussianModel(3); model.load_ply(str(path)); model.active_sh_degree=3
            rgb=render(cam,model,pipe,bg)['render']
            im=image(rgb); im.save(out/(label+'.png')); images.append(im)
            if label=='full-sh3': original=rgb.clone()
            del model
        source = GaussianModel(3); source.load_ply(str(snap/'world.ply')); source.active_sh_degree=3
        src=render(source_camera,source,pipe,bg)['render']
        diff=abs(src-original); parity=dict(mean_abs=float(diff.mean()),max_abs=float(diff.max()))
        assert parity['mean_abs']<2e-6 and parity['max_abs']<.005,parity
    page=Image.new('RGB',(images[0].width*len(images),images[0].height),'black')
    for i,im in enumerate(images): page.paste(im,(i*im.width,0))
    page.save(out/'same-camera-sh3-dc-reference-alpha.png')
    rt.save(out/'native-result.json',dict(updated=rt.now(),input_sha256=rt.sha(out/'result.json'),
        source_vs_export_native_parity=parity,columns=['full-sh3','dc-only','reference-colour','white-opacity'],
        same_camera_geometry_opacity=True,training_changes=False,optimizer_created=False,quality_accepted=False))
    print('SIDE_COLOUR_NATIVE_REPLAY_COMPLETE',parity,flush=True)


def run_native(pid):
    out=paths(120000)[2]/'user-side-view'
    lock=open('/tmp/shuolee-body-export123.lock','a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    rt.save(out/'native-plan.json',dict(created=rt.now(),script_sha256=rt.sha(__file__),
        input_sha256=rt.sha(out/'result.json'),budget_seconds=90,optimization_steps=0,
        intent='Render all four colour-only controls using the exact user-view camera; preserve native source/export parity.',
        changed_variable='rendered colour only',geometry_camera_opacity_unchanged=True))
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'native','--pid',str(pid)],check=True,timeout=90)
    finally:
        if process(pid)[2]==identity:os.kill(pid,signal.SIGCONT)
        rt.save(out/'native-gpu-handoff.json',dict(updated=rt.now(),pid=pid,identity=identity,trainer_after_state=process(pid)[1],training_changed=False))
        print('TRAINER_RESUMED',pid,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',nargs='?',default='prepare',choices=['prepare','native','run-native']); p.add_argument('--pid',type=int); args=p.parse_args()
    if args.mode=='prepare':main()
    elif args.mode=='native':native(args.pid)
    else:run_native(args.pid)
