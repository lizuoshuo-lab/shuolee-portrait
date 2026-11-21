"""Exercise the actual pinned CPU initializer, without SMPL inference or fitting."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from update_leafwoman129_state import FIT, ROOT, atomic, read, sha
import official_smplifyx_entry123 as original
import smplify_camera_init_support134 as bridge

OUT=ROOT/'leafwoman-fit40-132/camera-init-bridge134'
OFFICIAL=ROOT.parent/'smplifyx-reference-control/official'
# Deliberately hypothetical positive-control fixture, not admitted 022 anchors.
FIXTURE_IDS=(2,5,3,6)
FIXTURE_EDGES=((2,5),(3,6))


def verify():
    import torch
    from torch import nn
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    archive=FIT/'selected-fit.npz'
    assert sha(archive)==read(FIT/'result.json')['fit_sha256']
    original.adapted_entry_source(OFFICIAL)  # verifies all pinned upstream files
    _,_,fitting,_=original.load_official(OFFICIAL)
    with np.load(archive,allow_pickle=False) as z:
        frames=z['frame_ids'].astype(str).tolist()
        xy=z['observed_native_keypoints'].copy()
        # Approximate frozen model coordinates only, never a new mean-pose fit.
        points=z['joints_camera']-z['camera_translation'][:,None,:]
        K=z['K_compute_float32'].copy()
    pin={str(p):sha(p) for p in (archive,Path(bridge.__file__),Path(original.__file__),Path(__file__))}

    def rejected(fn, expected):
        try:fn()
        except ValueError as e:
            assert expected in str(e), str(e)
        else:raise AssertionError('Missing rejection: '+expected)

    def checked(target, ids, edges, reviewed_joints=None, reviewed_edges=None):
        return bridge.checked_arguments(target,(2704,1520),init_joints=ids,depth_edges=edges,
            reviewed_joints=ids if reviewed_joints is None else reviewed_joints,
            reviewed_depth_edges=edges if reviewed_edges is None else reviewed_edges)

    def evaluate(target, joints, ids, edges):
        q=torch.as_tensor(target[...,:2],dtype=torch.float64)
        k=torch.as_tensor(K,dtype=q.dtype)
        ray=torch.cat((q,torch.ones_like(q[...,:1])),dim=-1)@torch.linalg.inv(k).T
        rays=ray[...,:2]/ray[...,2:]
        model=SimpleNamespace(joints=torch.as_tensor(joints,dtype=q.dtype))
        seed=fitting.guess_init(lambda **_:model,rays,edges,focal_length=1.,use_vposer=False,dtype=q.dtype)
        bridge.check_seed_geometry(joints,seed.numpy(),ids,edges)
        class Camera(nn.Module):
            def __init__(self):
                super().__init__();self.translation=nn.Parameter(seed.clone())
            def forward(self,j):
                projected=(j+self.translation[:,None,:])@k.T
                return projected[...,:2]/projected[...,2:]
        cam=Camera()
        loss=fitting.SMPLifyCameraInitLoss(torch.as_tensor(ids),trans_estimation=seed,
            data_weight=1000/1520,depth_loss_weight=100,dtype=q.dtype)(model,cam,q)
        grad,=torch.autograd.grad(loss,cam.translation)
        return seed,loss.detach(),grad.detach()

    checks=[]
    for i,frame in enumerate(frames):
        target=xy[i:i+1];joints=points[i:i+1]
        attacked=target.copy();attacked[:,[9,12],:2]+=[150,200]
        old=(bridge.ORIGINAL_JOINTS,bridge.ORIGINAL_EDGES)
        rejected(lambda:checked(target,*old,reviewed_joints=(2,5),reviewed_edges=()),'Unverified camera-init')
        default=evaluate(target,joints,*old)
        admitted=evaluate(target,joints,*checked(target,*old))
        assert all(torch.equal(a,b) for a,b in zip(default,admitted))
        changed=evaluate(attacked,joints,*old)
        assert all(not torch.equal(a,b) for a,b in zip(default,changed))
        scoped=checked(target,FIXTURE_IDS,FIXTURE_EDGES)
        a=evaluate(target,joints,*scoped);b=evaluate(attacked,joints,*scoped)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        checks.append(dict(frame=frame,full_support_exact=True,unverified_default_rejected=True,
            old_seed_delta_z=float(changed[0][0,2]-default[0][0,2]),
            scoped_seed_loss_and_translation_gradient_invariant=True))
    assert len(checks)==24
    rejected(lambda:checked(xy[:1],FIXTURE_IDS,()),'Two distinct')
    rejected(lambda:checked(xy[:1],FIXTURE_IDS,FIXTURE_EDGES,reviewed_joints=(2,3,6)), 'Unverified camera-init')
    flat=xy[:1].copy();flat[:,FIXTURE_IDS,1]=500
    rejected(lambda:checked(flat,FIXTURE_IDS,FIXTURE_EDGES),'collinear')
    line=points[:1].copy();line[:,FIXTURE_IDS,:]=np.asarray([[0,0,0],[1,0,0],[2,0,0],[3,0,0]])
    rejected(lambda:bridge.check_seed_geometry(line,np.array([[0,0,4]]),FIXTURE_IDS,FIXTURE_EDGES), 'rigid degrees')

    # Invoke the actual copied entry and original pre-optimizer code. A local
    # test hook stops after original depth and camera-loss construction.
    class StopBeforeOptimizer(Exception):pass
    class FixtureBody(nn.Module):
        def __init__(self):
            super().__init__();self.betas=nn.Parameter(torch.zeros((1,10)))
        def reset_params(self, **kwargs):pass
        def forward(self, **kwargs):return SimpleNamespace(joints=torch.from_numpy(points[:1]))
    class FixtureDecoder(nn.Module):
        def decode(self, latent):return {'pose_body':latent.new_zeros((1,63))}
    actual=bridge.compiled_entry(FIXTURE_IDS,FIXTURE_EDGES)
    loader=actual.__globals__['load_official']
    reached=[]
    def stop(seed, loss, edges):
        assert tuple(loss.init_joints_idxs.tolist())==FIXTURE_IDS and tuple(map(tuple,edges))==FIXTURE_EDGES
        assert seed.shape==(1,3) and bool(torch.isfinite(seed).all())
        reached.append(seed.detach().cpu().tolist())
        raise StopBeforeOptimizer
    def test_loader(root):
        u,p,f,o=loader(root);f._test_stop=stop;return u,p,f,o
    def intercepted_source(root):
        code,diff=original.adapted_entry_source(root)
        marker='    loss = fitting.create_loss(loss_type=loss_type,'
        assert code.count(marker)==1
        return code.replace(marker,'    fitting._test_stop(init_t, camera_loss, edge_indices)\n'+marker,1),diff
    actual.__globals__.update(load_official=test_loader,adapted_entry_source=intercepted_source)
    with tempfile.TemporaryDirectory() as temp:
        try:actual(official_root=OFFICIAL,image=np.zeros((1,1,3),np.uint8),keypoints=xy[:1],
            K_integer_centers=K,body_model=FixtureBody(),vposer2=FixtureDecoder(),result_path=Path(temp)/'never.pkl',
            shared_betas=None,freeze_betas=False,device='cpu',diff_path=Path(temp)/'entry.diff')
        except StopBeforeOptimizer:pass
        else:raise AssertionError('Original camera-init hook was not reached.')
    assert len(reached)==1 and not torch.cuda.is_initialized()
    assert pin=={p:sha(p) for p in pin}
    atomic(OUT/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),state='cpu_bridge_verified_real_fit_not_admitted',
        source_hashes=pin,upstream_pins=original.PINNED,torch_version=torch.__version__,frames=checks,
        actual_pinned_pre_optimizer_call_received_both_arguments=True,
        rank_and_visibility_rejections_verified=True,full_support_exact_on_frozen_geometry=True,
        supported_fixture_uses_hypothetical_shoulders_elbows_not_admitted_real_anchors=True,
        source_images_modified=False,optimizer_updates=0,CUDA_initialized=False,training_started=False,
        rear_views_available=0,fit_admitted=False,quality_accepted=False))
    print('VIDEO22_ORIGINAL_INIT_BRIDGE_CPU_PASSED',len(checks),'optimizer_updates=0')


if __name__=='__main__':
    verify()
