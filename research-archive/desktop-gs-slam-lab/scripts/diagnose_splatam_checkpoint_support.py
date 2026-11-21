"""Preregister a fixed-map pose/support diagnostic; never resume or alter training."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/opt/gs-slam-lab/splatam')
LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
REPO = ROOT/'repos/SplaTAM'
NAME = 'freiburg1_room_full_320x240_rgbd_s0_20261010T125931Z_0d3dc918'
RUN = ROOT/'outputs/runs'/NAME
TRAJ = LAB/'outputs/splatam-repeat-evaluations-001'/NAME/'trajectory'
DATA = ROOT/'data/tum/rgbd_dataset_freiburg1_room'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(p.read_text(encoding='utf8'))
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def associations():
    with (TRAJ/'associations.csv').open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def cases():
    result=[{'id':'base','kind':'base','axis':None,'value':0.}]
    for axis in range(3):
        for value in (-.05,.05):result.append({'id':f't{"xyz"[axis]}{value:+.2f}m','kind':'translation','axis':axis,'value':value})
    for axis in range(3):
        for value in (-15.,-5.,5.,15.):result.append({'id':f'r{"xyz"[axis]}{value:+.0f}deg','kind':'rotation','axis':axis,'value':value})
    return result

def prepare():
    out=LAB/'outputs/splatam-checkpoint-support-diagnostic-001'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    execution=read(RUN/'execution.json'); config=read(RUN/'effective-config.json')
    assert config['mapping']['lrs']['cam_trans']==config['mapping']['lrs']['cam_unnorm_rots']==0
    assert config['tracking']['use_sil_for_loss'] and config['tracking']['sil_thres']==.99
    assert config['tracking']['use_l1'] and not config['tracking']['ignore_outlier_depth_loss']
    assert config['tracking']['loss_weights']=={'im':.5,'depth':1.}
    for relative,expected in execution['source_sha256_before'].items():assert sha(REPO/relative)==expected,relative
    assoc=associations()
    files=[Path(__file__).resolve(),RUN/'execution.json',RUN/'effective-config.json',TRAJ/'associations.csv',RUN/'params.npz',RUN/'sensor-config.yaml']
    assert sha(RUN/'params.npz')==execution['params_sha256']
    selected=[]
    for frame in (800,1100):
        a=assoc[frame]; selected.append({'frame':frame,'checkpoint':str(RUN/f'params{frame}.npz'),
            'rgb_path':str(DATA/a['rgb_path']),'depth_path':str(DATA/a['depth_path']), 'rgb_timestamp':float(a['rgb_timestamp'])})
        files += [RUN/f'params{frame}.npz', DATA/a['rgb_path'],DATA/a['depth_path']]
    renderer=execution['renderer_provenance']
    files += [Path(renderer['extension_file']),Path(renderer['module_file'])]
    assert sha(Path(renderer['extension_file']))==renderer['extension_sha256_before']
    assert sha(Path(renderer['module_file']))==renderer['module_sha256']
    plan={'status':'preregistered_before_any_new_GPU_render','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
       'local_only':True,'run_name':NAME,'output':str(out),'frames':selected,'cases':cases(),
       'inputs_sha256':{str(p):sha(p) for p in files},'official_sources_sha256':execution['source_sha256_before'],
       'renderer_extension_sha256':renderer['extension_sha256_before'],
       'hypothesis':'Pose-dependent hard silhouette support can change the scale of tracking summed RGB/depth losses. Measure it; do not infer that it caused historical jumps.',
       'protocol':{'map':'Exact saved post-mapping checkpoints800 and1100, fixed in every condition; not maps just before jumps838 and1133',
          'pose':'Saved same-frame estimated W2C, left-compose a camera-coordinate rigid perturbation; no optimization, GT pose or offset fitting',
          'translations_m':[-.05,.05],'rotations_deg':[-15,-5,5,15],'axes':['x','y','z'],
          'RGB':'Read exact frozen official TUMDataset __getitem__, float64 resize then float32 tensor /255; depth float64 resize /5000 then float32, no crop',
          'camera':'Saved final params intrinsics and first-frame w2c, including float32 near-identity values; base retains exact raw checkpoint camera parameters',
          'mask':'exact official depth>0 AND non-NaN rendered depth/variance AND silhouette>.99',
          'measure':'Original weighted sum, per-supported-pixel residual, counts and fixed-baseline-pixel weighted sum; independently recompute original loss on every condition',
          'zero_support':'Record as invalid-support diagnostic; never label it a successful or best pose',
          'image_selection':'Always display base, tx+0.05m and ry+15deg; do not select by measured loss'},
       'limits':['Sparse checkpoints do not reconstruct historical online loss masks',
          'Post-mapping same-frame loss is a fixed-map diagnostic, not the original pre-mapping tracking loss',
          'No training, optimizer step, RNG continuation, trajectory correction or new algorithm accuracy score']}
    save(out/'plan.json',plan)
    print(json.dumps({'status':plan['status'],'plan':str(out/'plan.json'),'plan_sha256':sha(out/'plan.json')}),flush=True)

def execute(plan_path):
    plan=read(plan_path);out=Path(plan['output']);assert plan['status']=='preregistered_before_any_new_GPU_render'
    assert not (out/'metrics.json').exists()
    for filename,expected in plan['inputs_sha256'].items():assert sha(Path(filename))==expected,filename
    for relative,expected in plan['official_sources_sha256'].items():assert sha(REPO/relative)==expected,relative
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(REPO))
    spec=importlib.util.spec_from_file_location('frozen_splatam_readonly',REPO/'scripts/splatam.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.is_available()
    with np.load(RUN/'params.npz',allow_pickle=False) as final:
        k=final['intrinsics'].copy();first_w2c=final['w2c'].copy()
    cam=official.setup_camera(320,240,k,first_w2c)
    config=read(RUN/'effective-config.json')
    dataset=official.get_dataset(config_dict=official.load_dataset_config(str(RUN/'sensor-config.yaml')),
        basedir=str(DATA.parent),sequence=DATA.name,start=0,end=-1,stride=1,
        desired_height=240,desired_width=320,device='cuda',relative_pose=True,
        ignore_bad=config['data'].get('ignore_bad',False),use_train_split=config['data'].get('use_train_split',True))
    assert len(dataset)==1315
    allrows=[];show={}; checks=[]
    for selected in plan['frames']:
        frame=selected['frame']; raw=dict(np.load(selected['checkpoint'],allow_pickle=False))
        required={'means3D','rgb_colors','unnorm_rotations','logit_opacities','log_scales','cam_unnorm_rots','cam_trans'}
        assert set(raw)==required,set(raw)
        params={name:torch.tensor(value,device='cuda',dtype=torch.float32,requires_grad=name.startswith('cam_')) for name,value in raw.items()}
        count=len(raw['means3D'])
        base=np.eye(4,dtype=np.float32)
        base[:3,:3]=official.build_rotation(F.normalize(params['cam_unnorm_rots'][...,frame].detach())).cpu().numpy()
        base[:3,3]=params['cam_trans'][...,frame].detach().cpu().numpy().reshape(3)
        assert Path(dataset.color_paths[frame]).resolve()==Path(selected['rgb_path']).resolve()
        assert Path(dataset.depth_paths[frame]).resolve()==Path(selected['depth_path']).resolve()
        color_tensor,depth_tensor,frame_intrinsics,_unused_target_GT=dataset[frame]
        assert np.array_equal(frame_intrinsics[:3,:3].cpu().numpy(),k)
        curr={'im':color_tensor.permute(2,0,1)/255,'depth':depth_tensor.permute(2,0,1),
              'w2c':torch.tensor(first_w2c,device='cuda'),'cam':cam}
        rgb=curr['im'].detach().cpu().numpy().transpose(1,2,0)
        depth=curr['depth'][0].detach().cpu().numpy()
        assert depth.shape==(240,320) and rgb.shape==(240,320,3) and np.isfinite(depth).all() and (depth>0).any()
        fixed_mask=None
        for case in plan['cases']:
            delta=np.eye(4,dtype=np.float32)
            if case['kind']=='translation':delta[case['axis'],3]=case['value']
            if case['kind']=='rotation':
                rv=np.zeros(3);rv[case['axis']]=np.deg2rad(case['value']);delta[:3,:3]=Rotation.from_rotvec(rv).as_matrix()
            pose=delta@base
            q_xyzw=Rotation.from_matrix(pose[:3,:3]).as_quat();q_wxyz=np.r_[q_xyzw[3],q_xyzw[:3]]
            if case['kind']!='base':
                with torch.no_grad():
                    params['cam_unnorm_rots'][0,:,frame].copy_(torch.tensor(q_wxyz,device='cuda',dtype=torch.float32))
                    params['cam_trans'][0,:,frame].copy_(torch.tensor(pose[:3,3],device='cuda'))
            variables={'max_2D_radius':torch.zeros(count,device='cuda')}
            loss,variables,weighted=official.get_loss(params,curr,variables,frame,{'im':.5,'depth':1.},True,.99,True,False,tracking=True)
            actual_loss=float(loss.detach());actual_im=float(weighted['im'].detach());actual_depth=float(weighted['depth'].detach())
            transformed=official.transform_to_frame(params,frame,gaussians_grad=False,camera_grad=True)
            rendervar=official.transformed_params2rendervar(params,transformed)
            im,_,_=official.Renderer(raster_settings=cam)(**rendervar)
            ds,_,_=official.Renderer(raster_settings=cam)(**official.transformed_params2depthplussilhouette(params,curr['w2c'],transformed))
            im_np=im.detach().cpu().numpy(); ds_np=ds.detach().cpu().numpy()
            rendered_depth=ds_np[0]; silhouette=ds_np[1]; variance=(ds[2]-ds[0]**2).detach().cpu().numpy()
            valid=depth>0
            finite_for_official=~np.isnan(rendered_depth)&~np.isnan(variance)
            mask=valid&finite_for_official&(silhouette>.99)
            if fixed_mask is None:
                assert mask.sum()>0 and np.isfinite(actual_loss) and actual_loss>0, 'Invalid baseline support/loss; no ratio or success claim'
                fixed_mask=mask.copy()
            rgb_res=np.abs(rgb.transpose(2,0,1)-im_np).sum(axis=0,dtype=np.float64)
            d_res=np.abs(depth-rendered_depth).astype(np.float64)
            computed_im=.5*float(rgb_res[mask].sum());computed_depth=float(d_res[mask].sum())
            close=bool(np.isclose(actual_loss,computed_im+computed_depth,rtol=2e-5,atol=.01) and
                np.isclose(actual_im,computed_im,rtol=2e-5,atol=.01) and np.isclose(actual_depth,computed_depth,rtol=2e-5,atol=.01))
            assert close,(frame,case,actual_loss,computed_im+computed_depth)
            pixels=int(mask.sum());intersection=int((mask&fixed_mask).sum())
            row={'frame':frame,'case':case['id'],'kind':case['kind'],'axis':case['axis'],'value':case['value'],
              'gaussians':count,'official_weighted_sum_loss':actual_loss,'weighted_rgb_sum':actual_im,'depth_sum_m':actual_depth,
              'loss_support_pixels':pixels,'valid_sensor_depth_pixels':int(valid.sum()),'silhouette_above_099_pixels':int((silhouette>.99).sum()),
              'loss_support_fraction_of_sensor_valid':pixels/int(valid.sum()),'intersection_with_base_pixels':intersection,
              'base_support_pixels':int(fixed_mask.sum()),'invalid_zero_support':pixels==0,
              'unweighted_rgb_channel_mae':computed_im*2/(3*pixels) if pixels else None,
              'depth_mae_m':computed_depth/pixels if pixels else None,
              'weighted_loss_per_supported_pixel':(computed_im+computed_depth)/pixels if pixels else None,
              'weighted_loss_on_fixed_base_pixels':float((.5*rgb_res+d_res)[fixed_mask].sum()),
              'nan_pixels':int((np.isnan(im_np).any(axis=0)|np.isnan(ds_np).any(axis=0)).sum()),
              'inf_pixels':int((np.isinf(im_np).any(axis=0)|np.isinf(ds_np).any(axis=0)).sum()),
              'render_variance_nan_pixels':int(np.isnan(variance).sum()),'render_variance_inf_pixels':int(np.isinf(variance).sum()),
              'official_loss_independently_recomputed':close}
            assert row['nan_pixels']==row['inf_pixels']==row['render_variance_nan_pixels']==row['render_variance_inf_pixels']==0
            allrows.append(row);checks.append(close)
            if case['id'] in ('base','tx+0.05m','ry+15deg'):
                show[(frame,case['id'])]=(rgb,im_np.transpose(1,2,0),silhouette,mask)
            del loss,variables,weighted,transformed,rendervar,im,ds
        del params,raw,curr
        torch.cuda.empty_cache()
        print(json.dumps({'frame':frame,'actual_conditions':len(plan['cases']),'official_loss_checks':len(checks)}),flush=True)
    table=out/'pose-support.csv'
    with table.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    fig,axs=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for ri,frame in enumerate((800,1100)):
        rr=[r for r in allrows if r['frame']==frame];base=rr[0];x=np.arange(len(rr))
        axs[ri,0].plot(x,[r['official_weighted_sum_loss']/base['official_weighted_sum_loss'] for r in rr],marker='.',label='Original variable-mask sum / base')
        axs[ri,0].plot(x,[r['weighted_loss_on_fixed_base_pixels']/base['weighted_loss_on_fixed_base_pixels'] for r in rr],marker='.',label='Fixed-base-pixel sum / base')
        axs[ri,1].plot(x,[r['loss_support_pixels']/base['loss_support_pixels'] for r in rr],marker='.',label='Support count / base')
        axs[ri,1].plot(x,[r['weighted_loss_per_supported_pixel']/base['weighted_loss_per_supported_pixel'] if not r['invalid_zero_support'] else np.nan for r in rr],marker='.',label='Per-supported-pixel loss / base')
        for ax in axs[ri]:
            ax.set_xticks(x,[r['case'] for r in rr],rotation=75,fontsize=7);ax.set_title(f'Fixed post-mapping checkpoint and input f{frame}');ax.grid(alpha=.2);ax.axhline(1,color='gray',ls='--',lw=.8);ax.legend(fontsize=8)
    fig.suptitle('Registered rigid pose probes: map and input fixed; no optimizer or GT tuning\nThese are sparse checkpoint diagnostics, not historical masks at the trajectory jumps')
    fig.savefig(out/'fixed-map-support-and-loss.png',dpi=150);plt.close(fig)
    for frame in (800,1100):
        fig,axes=plt.subplots(3,4,figsize=(13,8),constrained_layout=True)
        for ri,key in enumerate(('base','tx+0.05m','ry+15deg')):
            rgb,render,sil,mask=show[(frame,key)]
            for ci,(img,title) in enumerate(((rgb,'Actual input RGB'),(np.clip(render,0,1),'Frozen-map rendered RGB'),(sil,'Raw silhouette [0,1]'),(mask,'Official tracking support mask'))):
                axes[ri,ci].imshow(img,cmap='gray',vmin=0,vmax=1);axes[ri,ci].axis('off');axes[ri,ci].set_title(key+' / '+title,fontsize=9)
        fig.suptitle(f'Fixed saved post-mapping f{frame} checkpoint; three display poses chosen before renders\nMaps are unchanged, poses deliberately perturbed; not the original online tracking stage')
        fig.savefig(out/f'checkpoint-{frame}-registered-probes.png',dpi=150);plt.close(fig)
    for filename,expected in plan['inputs_sha256'].items():assert sha(Path(filename))==expected,filename
    for relative,expected in plan['official_sources_sha256'].items():assert sha(REPO/relative)==expected,relative
    comparisons=[]
    for frame in (800,1100):
        rr=[r for r in allrows if r['frame']==frame];base=rr[0]
        for row in rr[1:]:
            comparisons.append({'frame':frame,'case':row['case'],
              'original_sum_lower':not row['invalid_zero_support'] and row['official_weighted_sum_loss']<base['official_weighted_sum_loss'],
              'fixed_base_pixel_loss_higher':row['weighted_loss_on_fixed_base_pixels']>base['weighted_loss_on_fixed_base_pixels'],
              'support_smaller':row['loss_support_pixels']<base['loss_support_pixels'],'invalid_zero_support':row['invalid_zero_support']})
    outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    result={'status':'completed_registered_read_only_fixed_map_support_diagnostic','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
       'plan_sha256':sha(plan_path),'actual_condition_count':len(allrows),'actual_renderer_forward_count':len(allrows)*4,'official_loss_recomputation_checks':sum(checks),
       'outputs_sha256':outputs,'rows':allrows,'comparisons_to_base':comparisons,'inputs_and_official_sources_unchanged':True,
       'GPU':{'name':torch.cuda.get_device_name(),'torch':torch.__version__,'max_memory_allocated_bytes':torch.cuda.max_memory_allocated()},
       'limits':plan['limits'],'not_new_trajectory_or_accuracy':True}
    save(out/'metrics.json',result)
    print(json.dumps({'status':result['status'],'output':str(out),'conditions':len(allrows),'checks_passed':sum(checks)}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute-plan',type=Path);args=parser.parse_args()
    if args.prepare==bool(args.execute_plan):parser.error('Choose prepare OR execute-plan')
    if args.prepare:prepare()
    else:execute(args.execute_plan)

if __name__=='__main__':main()
