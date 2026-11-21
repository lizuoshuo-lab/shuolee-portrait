"""Isolated neck/head/wrist refinements on top of the frozen VPoser body."""
import difflib
import torch

JOINT_NAMES = ('neck', 'head', 'left_wrist', 'right_wrist')
FULL_JOINT_IDS = (12, 15, 20, 21)
BODY_INDICES = tuple(i-1 for i in FULL_JOINT_IDS)
PARAMETER = 'local_joint_offset124'
FACE_HAND_NAMES = ('expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose')


class LocalDecoder:
    def __init__(self, original, body):
        self.original, self.body = original, body

    def decode(self, latent):
        result = self.original.decode(latent)
        pose = result['pose_body'].reshape(latent.shape[0],21,3)
        indices = torch.as_tensor(BODY_INDICES,device=pose.device,dtype=torch.long)
        delta = getattr(self.body,PARAMETER)
        assert delta.shape == (1,4,3) and pose.shape[0] == 1
        updated = pose.index_copy(1,indices,pose.index_select(1,indices)+delta)
        return {**result,'pose_body':updated}


def install(body, original):
    from smplx.joint_names import JOINT_NAMES as names
    assert tuple(names[i] for i in FULL_JOINT_IDS)==JOINT_NAMES
    assert not hasattr(body,PARAMETER)
    body.register_parameter(PARAMETER,torch.nn.Parameter(torch.zeros(1,4,3,device=body.betas.device,dtype=body.betas.dtype)))
    return LocalDecoder(original,body)


def adapt_source(source):
    hook='                body_optimizer, body_create_graph = optim_factory.create_optimizer(\n'
    assert source.count(hook)==1
    insertion="""                _local_delta = body_model.local_joint_offset124
                if opt_idx == len(opt_weights) - 1:
                    _local_names = ('expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'local_joint_offset124')
                    final_params = [p for name, p in body_model.named_parameters()
                                    if name in _local_names and p.requires_grad]
                    assert len(final_params) == 7
                    assert all(p is not pose_embedding and p is not body_model.global_orient for p in final_params)
                else:
                    final_params = [p for p in final_params if p is not _local_delta]
"""
    changed=source.replace(hook,insertion+hook)
    compile(changed,'<original-smplifyx-local-head-wrist-parameters>','exec')
    return changed,''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),
        fromfile='existing-seeded-original-entry',tofile='local-neck-head-wrist-final-stage'))
