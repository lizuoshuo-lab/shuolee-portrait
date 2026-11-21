"""Resume the original final two body stages from one explicitly saved pose.

No new optimizer. The existing upstream seed hook and rotation-energy helper
are reused; the shared pose is a soft estimate, not measured 3D truth.
"""
import difflib,inspect
import numpy as np
import torch
import official_smplifyx_entry123 as entry
import control_smplifyx_root_seed123 as seed
from smplify_temporal124 import rotation_energy

BASE_SOURCE=entry.adapted_entry_source
BASE_LOAD=entry.load_official
BASE_CONFIG=entry.load_official_config

def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def source(root):
    entry._root_seed_original_source=BASE_SOURCE
    original,delta=seed.adapted_source(root)
    code=original
    start=code.index('    init_t = fitting.guess_init(');end=code.index('    camera_loss = ',start)
    code=code[:start]+"    init_t = torch.as_tensor(kwargs['_saved_seed']['camera_translation'],device=device,dtype=dtype)\n\n"+code[end:]
    start=code.index('        # Reset the parameters to estimate the initial translation')
    end=code.index('        _seed_trace = dict(',start)
    # The seed has already been projected and verified at full K. Re-running
    # torso initialization would throw away its fitted placement and face/hand.
    replacement="""        _saved_seed = kwargs['_saved_seed']
        body_model.reset_params(**{name:_saved_seed[name] for name,_ in body_model.named_parameters()})
        with torch.no_grad():
            pose_embedding.copy_(torch.as_tensor(_saved_seed['pose_embedding'],device=device,dtype=dtype))
            camera.translation.copy_(init_t)
        try_both_orient = False
        _require_fixed_K(camera, kwargs['_fixed_K'])
        if kwargs.get('_seed_replay_only', False):
            _r={name:v.detach().cpu().numpy().copy() for name,v in body_model.named_parameters()}
            _r.update(camera_translation=camera.translation.detach().cpu().numpy().copy(),body_pose=pose_embedding.detach().cpu().numpy().copy())
            return {'winner':{'result':_r,'loss':0.0},'orientation_count':1}

"""
    code=code[:start]+replacement+code[end:]
    code=once(code,"_reset_body(body_model, kwargs['_shared_betas'], **new_params)",
        "body_model.reset_params(**{name:_saved_seed[name] for name,_ in body_model.named_parameters()})")
    loop="            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):\n"
    code=once(code,loop,loop+"                if opt_idx < 3:\n                    continue\n")
    code=once(code,'                body_params = list(body_model.parameters())',
        '                body_params = list(body_model.parameters())\n                body_params.append(camera.translation)')
    compile(code,'<original-final-stages-from-shared-pose>','exec')
    return code,delta+'\n'+''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),
        fromfile='existing-original-root-seed-interface',tofile='saved-pose-original-stage34'))

def load_with_anchor(root,anchor,receipt,*,root_anchor=None):
    utils,prior,fitting,optimizer=BASE_LOAD(root)
    factory=fitting.create_loss;base=fitting.SMPLifyLoss
    root_weight=30. if root_anchor is not None else 0.
    class AnchoredLoss(base):
        def forward(self,obj,*args,**kwargs):
            original=super().forward(obj,*args,**kwargs)
            target=torch.zeros_like(obj.full_pose)
            target[:,:3]=(obj.full_pose[:,:3].detach() if root_anchor is None else
                          torch.as_tensor(root_anchor,device=target.device,dtype=target.dtype))
            target[:,3:66]=torch.as_tensor(anchor,device=target.device,dtype=target.dtype)
            extra=rotation_energy(obj.full_pose,target,1.,1.,body_weight=30.,root_weight=root_weight)
            receipt['calls']=receipt.get('calls',0)+1
            receipt['_last_original']=original.detach();receipt['_last_anchor']=extra.detach()
            return original+extra
    def make(loss_type='smplify',**kwargs):
        if loss_type=='smplify':return AnchoredLoss(**kwargs)
        return factory(loss_type=loss_type,**kwargs)
    fitting.create_loss=make
    return utils,prior,fitting,optimizer

def fit(*,saved_seed,replay_only=False,**kwargs):
    log={}
    def config(root):
        c=BASE_CONFIG(root);assert c['maxiters']==30 and c['optim_type']=='lbfgsls'
        c.update(_saved_seed=saved_seed,_seed_pose_embedding=saved_seed['pose_embedding'],
            _seed_global_orient=saved_seed['global_orient'],_seed_root_enabled=True,_seed_replay_only=replay_only)
        return c
    scope=entry.__dict__.copy();scope.update(adapted_entry_source=source,load_official_config=config,
        load_official=lambda root:load_with_anchor(root,saved_seed['body_pose_axis_angle'],log))
    exec(compile(inspect.getsource(entry.fit_one),'<same-original-entry-seeded-refine>','exec'),scope)
    arrays,report=scope['fit_one'](**kwargs)
    report.update(five_stages=False,original_stage_indices=[] if replay_only else [3,4],
        zero_optimization_replay=bool(replay_only),new_solver=False,
        upstream_loss_monitor_optimizer_unchanged=False,
        actual_method='Original SMPLify-X loss, FittingMonitor, LBFGS and stage3/4 weights plus existing body rotation energy30 around the shared-pose estimate; root weight0. Per-view placement stays free.',
        initialization='All body, root, placement, face and hand parameters from the same frozen8-view shared-pose archive. No per-frame best-run or neighbour selection.',
        camera_initialization='Saved body placement, full K preserved; no new camera initialization or background-camera solve.',
        anchor_calls=log.get('calls',0),anchor_body_weight=30.,anchor_root_weight=0.,
        last_original_loss=float(log['_last_original']) if log else None,
        last_anchor_loss=float(log['_last_anchor']) if log else None)
    return arrays,report
