"""Fixed single-Gaussian original-CUDA boundary/gradient check; no training.

register is CPU-only. run requires an idle GPU and creates a fresh bounded
diagnostic. It does not read person Gaussians into CUDA, optimize, change any
model, change the renderer, or adopt a bound into a training experiment.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import export_leafwoman_final129 as final

H=Path(__file__).resolve().parent
ROOT=final.ROOT.parent/'leafwoman-half12-dcbox1000-130'
AUDIT=ROOT/'numerical-boundary-audit.json'
PLAN=ROOT/'numerical-boundary-probe-plan.json'
OUT=ROOT/'numerical-boundary-probe'
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
UP=final.T.parent.parent/'repos/GaussianAvatars_SMPLX'
C0=.28209479177387814
sha,read,save,now=final.sha,final.read,final.save,final.now


def coefficients():
    lower=np.float32(-.5/C0)
    zero=np.float32(np.float32(-.5)/np.float32(C0))
    inward=np.nextafter(np.nextafter(lower,np.float32(np.inf)),np.float32(np.inf))
    return [('unclipped_control',np.float32(0.)),('current_literal_lower',lower),
        ('original_RGB2SH_float32_zero',zero),('current_lower_inward_two_ULP',inward),
        ('current_literal_upper',np.float32(.5/C0))]


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not PLAN.exists() and not OUT.exists()
    tree=ast.parse(Path(__file__).read_text());compile(tree,str(__file__),'exec')
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    assert not any(isinstance(node.func,ast.Attribute) and node.func.attr in ('step','training_setup','save_ply') for node in calls)
    assert 'torch' not in sys.modules
    audit=read(AUDIT)
    expected=next(case for case in audit['cases'] if case['name']=='literal_lower_inward_2_ULP')
    assert float(coefficients()[3][1])==expected['DC']
    spec=importlib.util.find_spec('diff_gaussian_rasterization');assert spec and spec.origin
    package=Path(spec.origin);binaries=list(package.parent.glob('_C*.so'));assert len(binaries)==1
    paths={Path(__file__),AUDIT,package,binaries[0],UP/'utils/sh_utils.py',
        PROJECT/'src/shuolee_portrait/training.py',PROJECT/'src/avatar_core/scene/cameras.py',
        PROJECT/'src/avatar_core/utils/graphics_utils.py',Path(final.__file__)}
    paths.update(Path(item['path']) for item in audit['sources'])
    for item in audit['sources']:assert sha(item['path'])==item['sha256']
    assert sha(audit['PLY'])==audit['PLY_sha256']
    paths.add(Path(audit['PLY']))
    plan=dict(updated=now(),schema='fixed-native-single-SH0-boundary130-v1',helper_sha256=sha(__file__),
        immutable_source_hashes={str(path):sha(path) for path in sorted(paths,key=str)},
        cases=[dict(name=name,DC=float(value),float32_bits=hex(int(value.view(np.uint32)))) for name,value in coefficients()],
        fixed_geometry=dict(points=1,means3D=[0.,0.,2.],scales=[.08,.08,.08],quaternion_wxyz=[1.,0.,0.,0.],opacity=.8),
        fixed_camera=dict(wh=[32,32],K=[[32.,0.,16.],[0.,32.,16.],[0.,0.,1.]],w2c=np.eye(4).tolist()),
        fixed_renderer='Installed original diff_gaussian_rasterization; SH degree0, black background, no override RGB, no kernel change.',
        objective='Sum native image RGB; backward once per fixed case, inspect features_dc gradient. No optimizer exists.',
        comparison='Same geometry/radii and unclipped DC0 control; positive unclipped gradient must equal lower-inward and upper gradients exactly.',
        decoded_RGB_measurement='One Gaussian on black: infer its native decoded RGB from sum(image)/(2*sum(control image)); not a person opacity/quality metric.',
        candidate='Exactly two float32 nextafter steps inward from current lower. Keep current upper only if original native checks pass. Not a threshold search.',
        candidate_acceptance='Native inward-lower RGB strictly positive and gradient bit-equal to the positive control; upper gradient equal, decoded RGB within8*float32eps of1. Never adopt automatically.',
        coefficient_precision='Record exact float32 coefficient bits and native gradients, not a claim of exact double-precision RGB bounds.',
        budget_seconds=120,single_gpu_nonblocking_lock=True,output=str(OUT),fresh_once=True,
        no_person_model_GPU_load=True,no_optimizer=True,no_training=True,no_model_or_renderer_changes=True,
        GPU_executed=False,quality_accepted=False)
    save(PLAN,plan)
    print('NATIVE_DC_BOUNDARY130_CPU_REGISTERED',sha(__file__),sha(PLAN),flush=True)


def verify(plan):
    assert sha(__file__)==plan['helper_sha256']
    for path,digest in plan['immutable_source_hashes'].items():assert sha(path)==digest,path


def run():
    assert os.name!='nt' and not OUT.exists()
    plan=read(PLAN);verify(plan);final.no_training_process()
    import fcntl
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed native boundary120s budget')))
    signal.alarm(120);OUT.mkdir(exist_ok=False)
    try:
        import torch
        sys.path.insert(0,str(PROJECT/'src'))
        from run_official_gaussians123 import ASSETS
        from shuolee_portrait import runtime
        runtime.configure(ASSETS)
        from shuolee_portrait.training import camera
        from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        row=dict(frame='synthetic_fixed_single_SH0',**plan['fixed_camera'])
        cam=camera(row,0)[0]
        settings=GaussianRasterizationSettings(image_height=32,image_width=32,
            tanfovx=float(np.tan(cam.FoVx*.5)),tanfovy=float(np.tan(cam.FoVy*.5)),
            bg=torch.zeros(3,device='cuda'),scale_modifier=1.,
            viewmatrix=cam.world_view_transform.cuda(),projmatrix=cam.full_proj_transform.cuda(),
            sh_degree=0,campos=cam.camera_center.cuda(),prefiltered=False,debug=False)
        rasterizer=GaussianRasterizer(raster_settings=settings)
        geo=plan['fixed_geometry']
        means=torch.tensor([geo['means3D']],dtype=torch.float32,device='cuda')
        scales=torch.tensor([geo['scales']],dtype=torch.float32,device='cuda')
        rotation=torch.tensor([geo['quaternion_wxyz']],dtype=torch.float32,device='cuda')
        opacity=torch.tensor([[geo['opacity']]],dtype=torch.float32,device='cuda')
        records=[];images={};gradients={};control_radii=None
        for case in plan['cases']:
            dc=torch.full((1,1,3),case['DC'],dtype=torch.float32,device='cuda',requires_grad=True)
            means2D=torch.zeros_like(means,requires_grad=True)
            image,radii=rasterizer(means3D=means,means2D=means2D,shs=dc,colors_precomp=None,
                opacities=opacity,scales=scales,rotations=rotation,cov3D_precomp=None)
            assert bool(torch.isfinite(image).all()) and int(radii[0])>0
            image.sum().backward()
            assert dc.grad is not None and bool(torch.isfinite(dc.grad).all())
            if control_radii is None:control_radii=radii.detach().clone()
            assert torch.equal(radii,control_radii)
            images[case['name']]=image.detach().cpu().numpy()
            gradients[case['name']]=dc.grad.detach().cpu().numpy()
            records.append(dict(**case,native_image_min=float(image.min()),native_image_max=float(image.max()),
                native_image_sum_RGB=[float(x) for x in image.detach().double().sum((1,2)).cpu().numpy()],
                features_dc_gradient=dc.grad.detach().cpu().numpy().tolist(),radii=radii.cpu().tolist(),
                native_image_exact_zero=bool((image==0).all()),gradient_exact_zero=bool((dc.grad==0).all())))
            save(OUT/'progress.json',dict(updated=now(),completed_cases=len(records),records=records,quality_accepted=False))
        control=images['unclipped_control'];control_gradient=gradients['unclipped_control']
        denominator=2*control.astype(np.float64).sum((1,2))
        assert np.all(denominator>0) and np.all(control_gradient>0)
        for record in records:
            record['native_decoded_RGB_via_control_ratio']=(images[record['name']].astype(np.float64).sum((1,2))/denominator).tolist()
            record['gradient_bit_equal_to_unclipped_control']=bool(np.array_equal(gradients[record['name']],control_gradient))
        inward=next(r for r in records if r['name']=='current_lower_inward_two_ULP')
        upper=next(r for r in records if r['name']=='current_literal_upper')
        passed=bool(inward['gradient_bit_equal_to_unclipped_control'] and
            np.all(np.asarray(inward['native_decoded_RGB_via_control_ratio'])>0) and
            upper['gradient_bit_equal_to_unclipped_control'] and
            np.all(np.abs(np.asarray(upper['native_decoded_RGB_via_control_ratio'])-1.)<=8*np.finfo(np.float32).eps))
        torch.cuda.synchronize();verify(plan)
        result=dict(updated=now(),state='fixed_native_boundary_probe_completed',seconds=time.monotonic()-started,
            helper_sha256=sha(__file__),plan_sha256=sha(PLAN),records=records,
            inward_lower_and_existing_upper_native_checks_passed=passed,
            suggested_fixed_lower_if_separately_authorized=inward if passed else None,
            no_automatic_bound_adoption=True,loss_renderer_Adam_unchanged=True,
            no_optimizer_or_training=True,no_person_Gaussian_model_loaded=True,
            immutable_sources_unchanged=True,quality_accepted=False)
        save(OUT/'result.json',result)
        print('NATIVE_DC_BOUNDARY130_READONLY_COMPLETE',passed,sha(OUT/'result.json'),flush=True)
    except BaseException as error:
        save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False,quality_accepted=False))
        raise
    finally:
        signal.alarm(0);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','run'])
    globals()[parser.parse_args().mode]()
