"""Precision repair of a CPU-only audit; original failure and all fit files preserved."""
import difflib
import inspect
from pathlib import Path
import audit_unique_wrist_gradients124 as previous

original=inspect.getsource(previous.main)
replacements=[
    ('torch.set_num_threads(1);started=time.monotonic()',
     'torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic()'),
    ('dtype=torch.float32)', 'dtype=torch.float64)'),
    ("vposer,_,_=loader.load_vposer('cpu');vposer.eval()", "vposer,_,_=loader.load_vposer('cpu');vposer.double();vposer.eval()"),
    ("K=torch.from_numpy(output['K_compute_float32']);translation=torch.from_numpy(output['camera_translation'])",
     "K=torch.from_numpy(output['K_compute_float32']).double();translation=torch.from_numpy(output['camera_translation']).double()"),
    ("latent=torch.from_numpy(archive[prefix+'_pose_embedding'].copy()).requires_grad_()",
     "latent=torch.from_numpy(archive[prefix+'_pose_embedding'].copy()).double().requires_grad_()"),
    ('target=torch.from_numpy(modified_targets if modified else original_targets)',
     'target=torch.from_numpy(modified_targets if modified else original_targets).double()'),
    ('loss=fitting.SMPLifyLoss(interpenetration=False,',
     'loss=fitting.SMPLifyLoss(dtype=torch.float64,interpenetration=False,'),
    ('frozen={str(p):c.sha(p) for p in paths};OUT.mkdir()',
     "paths += [Path(_audit_source_module.__file__),_audit_source_module.OUT/'plan.json',_audit_source_module.OUT/'failure.json'];frozen={str(p):c.sha(p) for p in paths};OUT.mkdir();(OUT/'precision-repair.diff').write_text(_precision_patch_text);(OUT/'executed-audit.py').write_text(_compiled_audit_source);frozen[str(OUT/'executed-audit.py')]=c.sha(OUT/'executed-audit.py')")]
source=original
for old,new in replacements:
    assert source.count(old)==1,old
    source=source.replace(old,new)
delta=''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='failed-float32-audit',tofile='same-saved-states-float64-audit'))
scope=dict(previous.main.__globals__)
scope.update(OUT=previous.c.OUT/'gradient-audit-fp64-completed',__file__=__file__,_audit_source_module=previous,_compiled_audit_source=source,_precision_patch_text=delta)
exec(compile(source,'same-saved-states-float64-audit','exec'),scope)
scope['main']()
