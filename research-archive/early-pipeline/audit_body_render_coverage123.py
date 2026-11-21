"""Read-only, all-frame opacity coverage screening of an immutable checkpoint.

The original renderer's white override yields accumulated opacity. This avoids
mistaking black clothing for holes. No optimization, pruning, or input edits.
"""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw
from export_demo_body_snapshot123 import ROOT, process
from shuolee_portrait import runtime as rt

OBS = Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data/observations.npz')
POLICY = dict(version=1, maximum_region_soft_uncovered=.15, minimum_landmark_score=.5,
    line_width_over_person_height=.03, minimum_target_pixels=32,
    threshold_source='Existing fit screen maximum_target_uncovered_fraction=.15; newly applied to each observed limb.',
    result_action='review_required only; no automatic exclusion or quality acceptance',
    measurement_scope='Target-mask/2D-detector consistency, not independent physical truth.',
    occlusion='Low-confidence/missing support is unknown; self-occlusion can trigger false alarms.',
    rgb_replay_max_abs=2/255, rgb_replay_mean_abs=1e-4)


def dirs(step):
    snap=ROOT/'gaussian-run'/f'step-{step:06d}'
    return snap, snap/'render-coverage'


def plan(step):
    snap,out=dirs(step); out.mkdir(exist_ok=True)
    r=rt.read(snap/'result.json'); assert r['step']==step and rt.sha(snap/'world.ply')==r['world_ply_sha256']
    rows=rt.read(ROOT/'data/training-inputs.json')['records']
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z:
        assert list(z['frame_ids'])==[r['frame'] for r in rows]
    files=[snap/'result.json',snap/'bound-local.ply',snap/'binding.npz',snap/'current-meshes.npz',
        ROOT/'data/training-inputs.json',ROOT/'quality-screen-plan.json',OBS]
    value=dict(updated=rt.now(),step=step,frames=len(rows),policy=POLICY,
        input_hashes={str(p):rt.sha(p) for p in files},frames_budget=59,optimization_steps=0,
        colour_overrides=['ones for opacity','zero and half controls on reference33 and failure18'],
        target_scope='All59 registered frames, no selection by observed error',
        training_source_changes=False,threshold_scan=False,user_request='这种畸变的后面要加筛选机制')
    if (out/'plan.json').exists():
        old=rt.read(out/'plan.json');assert old['policy']==value['policy'] and old['input_hashes']==value['input_hashes']
    else: rt.save(out/'plan.json',value)
    print('RENDER_COVERAGE_PLAN_REGISTERED',step,flush=True)


def replay(step,pid):
    assert process(pid)[1]=='T' and 'train_demo_body_upstream_offset123.py train' in process(pid)[0]
    snap,out=dirs(step); saved=rt.read(out/'plan.json')
    assert saved['policy']==POLICY
    for p,h in saved['input_hashes'].items(): assert rt.sha(p)==h,p
    assert not (out/'native-replay.json').exists()
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=rt.read(ROOT/'data/training-inputs.json')['records']
    result=rt.read(snap/'result.json')
    with np.load(snap/'current-meshes.npz',allow_pickle=False) as z: mesh=dict(z)
    with np.load(snap/'binding.npz',allow_pickle=False) as z: bindings=np.array(z['triangle_ids'])
    model=SMPLXGaussianModel(3); model.load_ply(str(snap/'bound-local.ply'))
    assert np.array_equal(model.binding.cpu().numpy(),bindings)
    model.active_sh_degree=result['active_sh_degree']
    vertices=[]
    for r,v in zip(rows,mesh['vertices_camera']):
        E=np.asarray(r['w2c']); vertices.append((v-E[:3,3])@E[:3,:3])
    model.vertices_sequence=torch.tensor(np.asarray(vertices),device='cuda',dtype=torch.float32)
    model.faces=torch.tensor(mesh['faces'],device='cuda',dtype=torch.long)
    model.num_timesteps=len(rows); model.static_offset=None
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda'); ones=torch.ones_like(model._xyz)
    checks=[];started=time.monotonic()
    with torch.no_grad():
        for i,r in enumerate(rows):
            assert str(mesh['frame_ids'][i])==r['frame']
            model.select_mesh_by_timestep(i);c,_=camera(r,i,pose_index=i)
            rgb=render(c,model,pipe,bg)['render']
            expected=np.asarray(Image.open(snap/(r['frame']+'_gaussian.png')).convert('RGB'),dtype=np.float32)/255
            # Saved PNGs use the existing image() clamp+round. Compare the same
            # encoding, not unbounded SH radiance against an LDR file.
            actual=np.asarray(image(rgb),dtype=np.float32)/255; diff=np.abs(actual-expected)
            check=dict(frame=r['frame'],rgb_replay_mean_abs=float(diff.mean()),rgb_replay_max_abs=float(diff.max()),
                unclipped_radiance_min=float(rgb.min()),unclipped_radiance_max=float(rgb.max()),
                comparison_encoding='Existing training.image clamp0..1 and round to uint8, unchanged')
            assert check['rgb_replay_mean_abs']<POLICY['rgb_replay_mean_abs'] and check['rgb_replay_max_abs']<=POLICY['rgb_replay_max_abs'],check
            alpha3=render(c,model,pipe,bg,override_color=ones)['render']
            assert float((alpha3[0]-alpha3[1]).abs().max())==0 and float((alpha3[1]-alpha3[2]).abs().max())==0
            alpha=alpha3[0];assert float(alpha.min())>=0 and float(alpha.max())<=1.000001
            if r['frame'] in ('frame_000033','frame_000018'):
                half=render(c,model,pipe,bg,override_color=ones*.5)['render']
                zero=render(c,model,pipe,bg,override_color=ones*0)['render']
                check.update(half_colour_linearity_max=float((half-alpha3*.5).abs().max()),zero_colour_max=float(zero.abs().max()))
                assert check['half_colour_linearity_max']<1e-6 and check['zero_colour_max']==0
            arr=alpha.clamp(0,1).cpu().numpy()
            Image.fromarray(np.round(arr*65535).astype(np.uint16)).save(out/(r['frame']+'_alpha.png'))
            checks.append(check)
            if i%10==0: print('ALPHA_REPLAY',i+1,len(rows),flush=True)
        assert process(pid)[1]=='T'
        rt.save(out/'native-replay.json',dict(updated=rt.now(),step=step,frames=len(rows),checks=checks,
            passed=True,seconds=time.monotonic()-started,colour_is_not_coverage=True,optimizer_created=False,
            source_world_ply_sha256=result['world_ply_sha256'],source_bound_ply_sha256=rt.sha(snap/'bound-local.ply')))


def execute(step,pid):
    snap,out=dirs(step); saved=rt.read(out/'plan.json')
    assert saved['policy']==POLICY
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid);assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    start=time.monotonic()
    rt.save(out/'gpu-handoff.json',dict(started=rt.now(),pid=pid,command=cmd,identity=identity,state='suspending',training_sources_changed=False))
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--step',str(step),'--pid',str(pid)],check=True,timeout=180)
    finally:
        if process(pid)[2]==identity: os.kill(pid,signal.SIGCONT)
        d=rt.read(out/'gpu-handoff.json');d.update(finished=rt.now(),seconds=time.monotonic()-start,state='resumed',trainer_state=process(pid)[1]);rt.save(out/'gpu-handoff.json',d)
        print('TRAINER_RESUMED',pid,flush=True)


def score(step):
    snap,out=dirs(step); replay=rt.read(out/'native-replay.json');assert replay['passed'] and replay['frames']==59
    plan_data=rt.read(out/'plan.json'); assert plan_data['policy']==POLICY
    rows=rt.read(ROOT/'data/training-inputs.json')['records']
    with np.load(OBS,allow_pickle=False) as z:
        obs={str(f):(p,s) for f,p,s in zip(z['frame_ids'],z['keypoints'],z['scores'])}
    chains={'left_arm':[5,7,9],'right_arm':[6,8,10],'left_leg':[11,13,15],'right_leg':[12,14,16]}
    records=[]
    for r in rows:
        assert rt.sha(r['mask'])==r['mask_sha256']
        mask=np.asarray(Image.open(r['mask']).convert('L'))>127
        alpha=np.asarray(Image.open(out/(r['frame']+'_alpha.png')),dtype=float)/65535
        assert alpha.shape==mask.shape
        kp,confidence=obs[r['frame']]; h=r['initial_box_xyxy'][3]-r['initial_box_xyxy'][1]
        region_masks={'whole_foreground':(mask,True,None)}
        for name,ids in chains.items():
            support=Image.new('L',(mask.shape[1],mask.shape[0]),0)
            valid=np.isfinite(kp[ids]).all() and float(confidence[ids].min())>=POLICY['minimum_landmark_score']
            if valid: ImageDraw.Draw(support).line([tuple(v) for v in kp[ids]],fill=255,width=max(6,round(POLICY['line_width_over_person_height']*h)),joint='curve')
            region_masks[name]=((np.asarray(support)>0)&mask,valid,float(confidence[ids].min()))
        metrics={};warnings=[];unknown=[]
        for name,(region,valid,conf) in region_masks.items():
            pixels=int(region.sum())
            if not valid or pixels<POLICY['minimum_target_pixels']:
                unknown.append(name);metrics[name]=dict(target_pixels=pixels,minimum_confidence=conf,soft_uncovered=None);continue
            missing=float((1-alpha[region]).mean())
            metrics[name]=dict(target_pixels=pixels,minimum_confidence=conf,soft_uncovered=missing,coverage=1-missing)
            if missing>POLICY['maximum_region_soft_uncovered']: warnings.append(name)
        outside=float(alpha[~mask].sum()/max(alpha.sum(),1e-8))
        if outside>POLICY['maximum_region_soft_uncovered']:warnings.append('outside_target_mask')
        records.append(dict(frame=r['frame'],regions=metrics,outside_mask_alpha_fraction=outside,
            warning_regions=warnings,unknown_regions=unknown,status='review_required' if warnings or unknown else 'no_configured_alarm',quality_accepted=False))
    result=dict(updated=rt.now(),step=step,frames=len(rows),policy=POLICY,records=sorted(records,key=lambda r:r['frame']),
        warning_frames=[r['frame'] for r in records if r['warning_regions']],unknown_frames=[r['frame'] for r in records if r['unknown_regions']],
        input_plan_sha256=rt.sha(out/'plan.json'),native_replay_sha256=rt.sha(out/'native-replay.json'),script_sha256=rt.sha(__file__),
        training_inputs_changed=False,auto_rejected_frames=[],quality_accepted=False)
    rt.save(out/'result.json',result)
    print('RENDER_COVERAGE_SCORED',len(result['warning_frames']),'warn',len(result['unknown_frames']),'unknown',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run','replay','score']);p.add_argument('--step',required=True,type=int);p.add_argument('--pid',type=int)
    args=p.parse_args()
    if args.mode=='plan':plan(args.step)
    elif args.mode=='run':execute(args.step,args.pid)
    elif args.mode=='replay':replay(args.step,args.pid)
    else:score(args.step)
