"""Minimal parameter-list adapter; original SMPLify-X optimizer and loss remain."""
import difflib

ALLOWED = ('expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose')


def adapt_source(source):
    hook = '                body_optimizer, body_create_graph = optim_factory.create_optimizer(\n'
    assert source.count(hook) == 1
    insertion = """                if kwargs.get('_freeze_body_in_final_stage', False) and opt_idx == len(opt_weights) - 1:
                    _face_hand_names = ('expression', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose')
                    final_params = [p for name, p in body_model.named_parameters()
                                    if name in _face_hand_names and p.requires_grad]
                    assert len(final_params) == 6
                    assert all(p is not pose_embedding and p is not body_model.global_orient for p in final_params)
"""
    changed = source.replace(hook, insertion + hook)
    compile(changed, '<original-smplifyx-final-face-hand-parameters>', 'exec')
    diff = ''.join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
                   fromfile='existing-seeded-original-entry', tofile='final-face-hand-parameter-partition'))
    return changed, diff
