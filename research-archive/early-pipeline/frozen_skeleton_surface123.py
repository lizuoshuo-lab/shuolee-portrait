"""Clothing-only displacement bridge using original SMPL-X LBS transforms.

This is an optional local adapter. It is not imported by the live Gaussian run.
Initial pose, shape, expression, template offset and skeletal joints are frozen.
Only an additional shared canonical surface displacement is trainable.
"""
from unittest.mock import patch
import importlib
import numpy as np
import torch
import smplx
from torch.func import functional_call


class FrozenSkeletonSurface:
    def __init__(self, fit, initial_offset, model_path, device='cpu'):
        self.device=torch.device(device)
        self.model=smplx.SMPLX(str(model_path),batch_size=1,num_betas=10,
            num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
            flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,dtype=torch.float32).to(device).eval().requires_grad_(False)
        self.model.parents=self.model.parents.cpu()
        self.initial_offset=torch.as_tensor(initial_offset,device=device,dtype=torch.float32).reshape_as(self.model.v_template).detach().clone()
        self.fit=fit
        self.ids=list(fit['frame_ids'].astype(str));self.slots={f:i for i,f in enumerate(self.ids)}
        self.faces=torch.tensor(fit['faces'],device=device,dtype=torch.long)
        assert np.array_equal(self.model.faces,fit['faces'])
        self.params=[];self.base_vertices=[];self.base_skeleton=[];self.linear=[];self.base_rest_joints=[]
        self.full_model_joints=[]
        lbs=importlib.import_module('smplx.lbs')
        original=lbs.batch_rigid_transform
        with torch.no_grad():
            for i in range(len(self.ids)):
                values={k:self.tensor(fit[k][i:i+1]) for k in ('global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose')}
                values.update(body_pose=self.tensor(fit['body_pose_axis_angle'][i:i+1]),betas=self.tensor(fit['shared_betas']))
                self.params.append(values)
                captures=[]
                def observe(rot_mats,joints,parents,**kw):
                    result=original(rot_mats,joints,parents,**kw)
                    captures.append((result[1].detach().clone(),joints.detach().clone()))
                    return result
                # Observe original outputs; do not replace any original calculation.
                with patch.object(lbs,'batch_rigid_transform',observe):
                    output=self._official(i,self.initial_offset)
                assert len(captures)==1
                A,J=captures[0]
                T=(self.model.lbs_weights@A[0].reshape(A.shape[1],16)).reshape(-1,4,4)
                E=self.tensor(fit['w2c'][i]);translation=self.tensor(fit['camera_translation'][i])
                transform=lambda x:(x+translation-E[:3,3])@E[:3,:3]
                self.base_vertices.append(transform(output.vertices[0]))
                self.base_skeleton.append(transform(output.joints[0,:self.model.J_regressor.shape[0]]))
                self.full_model_joints.append(transform(output.joints[0]))
                self.linear.append(E[:3,:3].T@T[:,:3,:3])
                self.base_rest_joints.append(J)
        self.base_vertices=torch.stack(self.base_vertices)
        self.base_skeleton=torch.stack(self.base_skeleton)
        self.linear=torch.stack(self.linear)
        self.full_model_joints=torch.stack(self.full_model_joints)

    def tensor(self,x):return torch.as_tensor(x,device=self.device,dtype=torch.float32)

    def _official(self,slot,offset):
        return functional_call(self.model,{'v_template':self.model.v_template+offset},(),
            dict(self.params[slot],return_verts=True,return_shaped=False))

    def vertices(self,delta,slot=None):
        """One shared [10475,3] delta; never accept per-frame free offsets."""
        if tuple(delta.shape)!=tuple(self.initial_offset.shape):
            raise ValueError('Expected one shared canonical vertex displacement')
        if slot is None:return self.base_vertices+(self.linear@delta[None,:,:,None]).squeeze(-1)
        return self.base_vertices[slot]+(self.linear[slot]@delta[:,:,None]).squeeze(-1)

    def joints(self,slot=None):
        # Skeletal joints only. Surface-derived face/foot landmarks are not bones.
        return self.base_skeleton if slot is None else self.base_skeleton[slot]

    def independent_original_forward(self,delta,slot):
        """Test oracle: original full forward with only rest joints held fixed."""
        lbs=importlib.import_module('smplx.lbs');calls=[]
        def fixed_joints(regressor,vertices):
            calls.append(1)
            assert regressor.shape==self.model.J_regressor.shape
            return self.base_rest_joints[slot]
        with patch.object(lbs,'vertices2joints',fixed_joints):
            output=self._official(slot,self.initial_offset+delta)
        assert len(calls)==1
        E=self.tensor(self.fit['w2c'][slot]);translation=self.tensor(self.fit['camera_translation'][slot])
        convert=lambda x:(x+translation-E[:3,3])@E[:3,:3]
        return convert(output.vertices[0]),convert(output.joints[0,:self.model.J_regressor.shape[0]])
