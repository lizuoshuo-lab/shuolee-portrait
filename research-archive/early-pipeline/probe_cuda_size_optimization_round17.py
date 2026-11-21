"""Round 17 synthetic preflight using the exact installed CUDA rasterizer."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-multiview-quality-20261111')
sys.path[:0] = [str(PROJECT/'src')]
from shuolee_portrait import runtime
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from shuolee_portrait.render_support import FootprintLimits, replacement_evaluator, source_weights
from shuolee_portrait.size_child_optimization import (
    measure_size_parent_priority, optimize_size_children, save_size_decisions,
)
from shuolee_portrait.size_child_commit import PARAMETERS, commit_size_children
from shuolee_portrait.training import camera
from shuolee_portrait.world_scale import WorldScale, scale_config
from utils.sh_utils import C0
from diff_gaussian_rasterization import _C

PROOF = Path('/mnt/c/Users/Administrator/Documents/New project/outputs/smpl-delivery-round17/input-audit/cuda-multiview-quality-proof.json')
SOURCES = ('src/avatar_core/scene/gaussian_model.py', 'src/shuolee_portrait/training.py',
           'src/shuolee_portrait/world_scale.py', 'src/shuolee_portrait/render_support.py',
           'src/shuolee_portrait/size_child_optimization.py', 'src/shuolee_portrait/size_child_commit.py')
MOMENTS = ('step', 'exp_avg', 'exp_avg_sq')
ATTRIBUTES = (*PARAMETERS.values(), 'binding', 'binding_counter', 'max_radii2D',
              'xyz_gradient_accum', 'denom', 'densification_ids', 'next_densification_id')

def require(value, message):
    if not value:
        raise AssertionError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def T(value):
    return torch.tensor(value, device='cuda', dtype=torch.float32)

class BoundFixture(GaussianModel):
    def select_mesh_by_timestep(self, slot):
        self.timestep = int(slot)
        self.face_center = self.face_origin + self.face_shift * (slot-11.5)
        self.face_scaling = torch.ones((2,1), device='cuda')
        self.face_orien_mat = torch.eye(3, device='cuda').repeat(2,1,1)
        self.face_orien_quat = T([[1,0,0,0]]).repeat(2,1)

def build_model():
    model = BoundFixture(0)
    n = 5
    model.face_origin = T([[0.,0.,2.], [0.,0.,1.76]])
    model.face_shift = T([[.0005,0.,0.],[-.0005,0.,0.]])
    model.binding = torch.tensor([0,1,1,1,1], device='cuda')
    model.binding_counter = torch.tensor([1,4], device='cuda', dtype=torch.int32)
    model.densification_ids = torch.tensor([21,43,51,72,88], device='cuda')
    model.next_densification_id = 100
    model._xyz = torch.nn.Parameter(T([[0.,0.,0.],[-.12,-.1,0.],[.12,-.1,.001],[-.12,.1,.002],[.12,.1,.003]]))
    model._scaling = torch.nn.Parameter(T([[.225,.17,.11],[.23,.17,.09],[.23,.17,.09],
                                           [.23,.17,.09],[.23,.17,.09]]).log())
    model._opacity = torch.nn.Parameter(T([.008,.99,.99,.99,.99]).logit()[:,None])
    model._features_dc = torch.nn.Parameter(((T([[.48,.51,.52]]).repeat(n,1)-.5)/C0)[:,None])
    model._features_rest = torch.nn.Parameter(torch.empty((n,0,3),device='cuda'))
    model._rotation = torch.nn.Parameter(T([[1,0,0,0]]).repeat(n,1))
    model.select_mesh_by_timestep(0)
    args = NS(percent_dense=.01,position_lr_init=.01,position_lr_final=.001,position_lr_delay_mult=1.,
              position_lr_max_steps=100,feature_lr=.01,opacity_lr=.01,scaling_lr=.01,rotation_lr=.01)
    model.training_setup(args)
    model.max_radii2D = T([34,9,10,11,12])
    model.xyz_gradient_accum[:] = T([[1],[2],[3],[4],[5]])
    model.denom[:] = T([[5],[4],[3],[2],[1]])
    for group in model.optimizer.param_groups:
        group['lr']=0.
    for _ in range(3):
        for i,group in enumerate(model.optimizer.param_groups):
            parameter, = group['params']
            parameter.grad = torch.arange(parameter.numel(), device='cuda', dtype=parameter.dtype).reshape_as(parameter)*.1+i+.1
        model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    model.point_limits = FootprintLimits(model._scaling)
    model.point_limits.set(torch.tensor([3],device='cuda'), T([[-1.3,-1.3,-1.3]]))
    return model

def snapshot(model):
    attrs = {name:getattr(model,name) for name in ATTRIBUTES}
    states = {parameter:{key:value.clone() for key,value in state.items()}
              for parameter,state in model.optimizer.state.items()}
    values = {key:value.detach().clone() if torch.is_tensor(value) else value for key,value in attrs.items()}
    limit_values = {key:value.detach().clone() if torch.is_tensor(value) else value
                    for key,value in vars(model.point_limits).items()}
    return attrs, values, states, limit_values

def assert_unchanged(model, original):
    attrs, values, states, limits = original
    for key, value in values.items():
        got = getattr(model,key)
        require(got is attrs[key], f'Object changed after a rejected decision: {key}')
        require(torch.equal(got,value) if torch.is_tensor(got) else got==value, f'Value changed: {key}')
    require(set(model.optimizer.state) == set(states),'Adam roster changed')
    for group in model.optimizer.param_groups:
        for parameter in group['params']:
            for key,value in states[parameter].items():
                require(torch.equal(model.optimizer.state[parameter][key],value),f'Adam changed: {group["name"]}.{key}')
    for key,value in limits.items():
        got = getattr(model.point_limits,key)
        require(torch.equal(got,value) if torch.is_tensor(got) else got==value,f'Limit changed: {key}')

@torch.no_grad()
def make_constraints(model, directory):
    w, h = 128, 112
    rows, cameras, views, matrices, protected = [], [], [], [], []
    black = torch.zeros(3,device='cuda')
    pipe = NS(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    for i in range(24):
        K = np.array([[86.,0.,63.1+(i%3)*.28],[0.,91.,55.3+(i%5)*.13],[0.,0.,1.]])
        pose = np.eye(4)
        theta = (i-11.5)*.010
        c,s = np.cos(theta),np.sin(theta)
        pose[0,:3] = [c,0,s]
        pose[2,:3] = [-s,0,c]
        pose[0,3] = -2*s+(i%4-1.5)*.012
        row = dict(frame=f'fixture-{i:02d}',wh=[w,h],K=K,w2c=pose,image=directory/f'source-{i:02d}.png')
        cam, proof = camera(row,i,pose_index=i)
        require(proof['projection_error_px']<.001,'Camera projection failed')
        model.select_mesh_by_timestep(i)
        rgb = render(cam,model,pipe,black)['render']
        alpha = render(cam,model,pipe,black,override_color=torch.ones_like(model.get_xyz))['render'][0]
        # Record raw RGB and the independently drawn alpha as immutable observations.
        straight = (rgb/alpha.clamp_min(1e-4)).clamp(0,1)
        raw = (straight*255).round().byte().cpu().permute(1,2,0).contiguous().numpy()
        Image.fromarray(raw).save(row['image'])
        source = torch.from_numpy(np.array(Image.open(row['image']))).permute(2,0,1).to('cuda').float().div(255)
        target = alpha.clone()
        inside = target>=.80
        outside = target<=.005
        require(bool(inside.any() and outside.any()),'Missing synthetic foreground or background')
        rows.append(row)
        cameras.append(cam)
        matrices.append((T(K),T(pose)))
        views.append(NS(rgb=source,mask=target[None],box=(0,0,w,h),inside=inside,outside=outside,face=None,
                        mesh_depth=torch.full_like(target,3.)))
        roi = inside.nonzero().T
        midpoint = roi.shape[1]//2
        protected.append({'fixture_center':roi[:,midpoint-8:midpoint+8]})
    model.select_mesh_by_timestep(0)
    return NS(cameras=cameras,pipe=pipe,black=black,views=views,matrices=matrices,rows=rows,
              slots=list(range(24)),bins=[i//4 for i in range(24)],protected_regions=protected,
              options=dict(foreground_target=.98,low_alpha=.65,reliable_foreground=.99,reliable_background=.01,
                           mesh_occlusion_slack_m=.025,required_source_bins=2))

def evaluate(model,constraints,parents,centers,logs,logits,dc,weights):
    config=scale_config()['render_support']
    with replacement_evaluator(model,constraints,parents,centers,logs,logits.sigmoid(),weights,config,
                               screen_limit_px=20,child_dc=dc) as trial:
        return trial([0])

def compact(trial):
    return dict(accepted=trial['accepted'],views=len(trial['frames']),native_radius_check=trial['native_radius_check'],
                max_foreground_loss_px=trial['max_foreground_loss_px'],max_background_gain_px=trial['max_background_gain_px'],
                checks_failed_by_frame={f['frame']:[x['item'] for x in f['failures']] for f in trial['frames'] if f['failures']},
                worst_rgb_increase=max(f['worst_rgb_tile']['value'] for f in trial['frames']))

def main():
    begin = perf_counter()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(4)
    with TemporaryDirectory(prefix='r17-cuda-preflight-') as tmp:
        directory = Path(tmp)
        model=build_model()
        constraints=make_constraints(model,directory)
        parents=torch.tensor([0],device='cuda')
        selected=torch.tensor([0],device='cuda')
        original=snapshot(model)
        pre_poses=[getattr(model,key) for key in ('face_center','face_orien_mat','face_scaling','face_orien_quat')]
        weights=source_weights(model,constraints,parents)
        priority=measure_size_parent_priority(model,constraints,parents,weights,20)
        require(int(priority.peak_radii[0])>20,'Fixture parent is not screen-oversized')
        require(float(weights[...,0].sum())>0.,'Fixture parent has no reliable foreground weight')
        sigma=model._scaling[parents].detach().exp()
        signs=T([[-1,-1],[-1,1],[1,-1],[1,1]])
        centers=model._xyz[parents,None].detach()+torch.cat((signs*sigma[0,:2]*.49,T([[0.]]).expand(4,1)),1)[None]
        logs=(sigma*.68).log()[:,None].expand(-1,4,-1).clone()
        upper=(sigma*.78).log()
        logits=T([.007]).logit().expand(1,4,1).clone()
        dc=model._features_dc[parents,None].detach().expand(-1,4,-1,-1).clone()
        # Negative control deliberately spreads large, nearly opaque children into reliable background.
        rejected=evaluate(model,constraints,parents,centers+T([.63,0,0]),logs+.9,
                          torch.full_like(logits,4.),dc,weights)
        require(len(rejected['frames'])==24 and not rejected['accepted'],'Actual evaluator accepted failing negative control')
        assert_unchanged(model,original)
        pre_poses=[getattr(model,key) for key in ('face_center','face_orien_mat','face_scaling','face_orien_quat')]
        fit_weights=weights.clone()
        fit_weights[0,0,0]=0
        result=optimize_size_children(model,constraints,parents,centers,logs,logits,dc,upper,fit_weights,
                                      max_cycles=8,guard_options=scale_config()['render_support'],diagnostic_dir=directory/'evidence')
        require(result.audit['optimizer_updates']>=24 and result.audit['center_max_local_delta']>0,
                f'Actual CUDA child optimization failed: {result.audit["status"]}')
        require(all(v>0 for v in result.audit['max_abs_parameter_deltas'].values()),'Some parameter groups did not learn')
        require(len(result.audit['views'])==24 and {view['view_index'] for view in result.audit['views']}==set(range(24)),
                'Optimizer did not measure every declared native view')
        masked_view=result.audit['views'][0]
        require(not masked_view['foreground_supervision'] and masked_view['exterior_supervision'],
                'Removing positive foreground support must retain independent exterior supervision.')
        require(all(getattr(model,key) is value for key,value in
                    zip(('face_center','face_orien_mat','face_scaling','face_orien_quat'),pre_poses)),
                'Optimizer changed live pose objects')
        assert_unchanged(model,original)
        optimized=evaluate(model,constraints,parents,result.centers,result.log_sigma,result.opacity_logits,result.dc,weights)
        print('optimized evaluator:',json.dumps(compact(optimized)),flush=True)
        require(len(optimized['frames'])==24 and optimized['accepted'],'Optimized fixture candidate failed the actual final guard')
        assert_unchanged(model,original)
        world_cap=T([[.4,.4,.4]]).log().expand(len(model.binding),3)
        real_prune=model.prune_points
        def fail_after_prune(mask):
            real_prune(mask)
            raise RuntimeError('injected after actual prune')
        with patch.object(model,'prune_points',side_effect=fail_after_prune):
            try:
                commit_size_children(model,model.point_limits,parents,result,selected,world_cap)
            except RuntimeError as exc:
                require(str(exc)=='injected after actual prune',f'Unexpected rollback error: {exc}')
            else:
                raise AssertionError('Missing injected post-prune failure')
        assert_unchanged(model,original)
        commit=commit_size_children(model,model.point_limits,parents,result,selected,world_cap)
        require(commit['children']==4 and commit['unchanged_adam_rows']==4,'Successful commit topology differs')
        require(model.binding_counter.tolist()==[4,4],'Face coverage changed')
        _, old, old_states, _ = original
        for name,key in PARAMETERS.items():
            old_param=original[0][key]
            new_param=getattr(model,key)
            require(torch.equal(new_param[:4],old[key][1:]),f'Commit mutated unrelated {name}')
            for moment in MOMENTS:
                new=model.optimizer.state[new_param][moment]
                before=old_states[old_param][moment]
                require(torch.equal(new if moment=='step' else new[:4],before if moment=='step' else before[1:]),
                        f'Commit mutated unrelated {name}.{moment}')
            if name in result.moments:
                betas=result.moments[name]['betas']
                for moment,beta in zip(('exp_avg','exp_avg_sq'),betas):
                    local=result.moments[name][moment][0]
                    expected=local*(1-beta**int(model.optimizer.state[new_param]['step']))/(1-beta**result.moments[name]['step'])
                    got=model.optimizer.state[new_param][moment][-4:]
                    require(torch.allclose(got,expected,atol=1e-7,rtol=2e-6),f'Incorrect migrated {name}.{moment}')
        decisions=save_size_decisions(model,constraints,result.audit['representatives'],directory/'evidence')
        require(len(decisions)==2,'Actual final renders missing')
        for item in result.audit['representatives']+decisions:
            require(item['sha256']==sha(directory/'evidence'/item['path']),'Diagnostic digest mismatch')
        fallback_model=build_model()
        fallback_world=object.__new__(WorldScale)
        fallback_world.model=fallback_model
        fallback_world.options=scale_config()
        fallback_world.limits=fallback_model.point_limits
        fallback_world.log_cap=T([.4,.4]).log()
        fallback_world.diagnostic_dir=None
        fallback_world.optimized_candidate_ids=set()
        fallback_event=dict(optimized_replacement_children=0,fallback_size_replaced=0,
                            fallback_replacement_children=0)
        # Inject a rejected optimizer result to exercise the normal dispatch
        # and exact fallback path in the same live synthetic scene.
        bad_result=NS(centers=centers+T([.63,0,0]),log_sigma=logs+.9,
                      opacity_logits=torch.full_like(logits,4.),dc=dc,
                      audit=dict(status='no_feasible_checkpoint',optimizer_updates=192))
        with patch('shuolee_portrait.size_child_optimization.optimize_size_children',return_value=bad_result):
            fallback_world._replace_supported_size(parents,weights,constraints,fallback_event,20,3000)
        attempt, = fallback_event['local_optimizations']
        require(attempt['decision_reason']=='native_guard_rejected'
                and len(attempt['replacement_trial']['frames'])==24
                and attempt['committed_parent_ids']==[],'Injected bad proposal was not rejected before fallback.')
        fallback=fallback_event['fallback_replacement']
        require(fallback['path']=='r15_fallback' and fallback['shortlist_parent_ids']==[21],
                'Legacy shortlist was not preserved.')
        require(fallback['committed_parent_ids']==[21] and len(fallback['child_ids'])==2
                and len(fallback['trial']['final_trial']['frames'])==24
                and fallback['trial']['final_trial']['accepted'],'Legacy path did not preserve the verified native replacement.')
        require(fallback_model.binding_counter.tolist()==[2,4]
                and fallback_event['optimized_size_replaced']==0 and fallback_event['fallback_size_replaced']==1,
                'Legacy fallback changed other face coverage or accounting.')
        audit = dict(result.audit)
        audit['synthetic_source_png_sha256']={row['frame']:sha(row['image']) for row in constraints.rows}
        evidence=dict(state='passed',scope='synthetic CUDA integration, no production subject training',
                      fixture_sha256=sha(Path(__file__)),gpu=torch.cuda.get_device_name(),
                      runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
                                   raster_extension_sha256=sha(Path(_C.__file__))),
                      source_sha256={path:sha(PROJECT/path) for path in SOURCES},
                      source_file_sha256=sha(PROJECT/'src/shuolee_portrait/size_child_optimization.py'),
                      source_parent_priority=dict(peak_radii=priority.peak_radii.tolist(),scores=priority.scores.tolist(),
                                                  foreground_pixels=float(weights[...,0].sum())),
                      optimization=audit,rejected=compact(rejected),optimized=compact(optimized),
                      fallback=dict(path=fallback['path'],parents=fallback['committed_parent_ids'],
                                    children=fallback['child_ids'],trial=compact(fallback['trial']['final_trial']),
                                    injected_optimized_rejection=compact(attempt['replacement_trial'])),
                      post_prune_rollback=dict(restored_model_and_optimizer=True,model_rows=len(original[0]['binding']),
                                               unrelated_adam_groups=list(PARAMETERS)),
                      commit=commit,actual_final_diagnostic_crops=decisions,elapsed_seconds=perf_counter()-begin)
        runtime.save(PROOF,evidence)
        print(json.dumps(dict(state='passed',sha256=sha(PROOF),proof=str(PROOF),
                              seconds=evidence['elapsed_seconds'],commit=commit)),flush=True)
if __name__=='__main__':
    main()

