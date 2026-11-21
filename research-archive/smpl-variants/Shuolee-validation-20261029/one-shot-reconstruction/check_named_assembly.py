"""Historical CPU assembly check; no model forward, fit, or Gaussian execution."""
from pathlib import Path
import ast
import json
import sys
import time

import numpy as np
import torch
import smplx

P = Path('/mnt/d/Projects/Shuolee')
sys.path.insert(0, str(P/'src'))
from shuolee_portrait import official_sequence as seq, runtime as rt

root = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/full94-shared-shape')
reference_source = Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/run_official_sequence123.py')
assert rt.sha(reference_source) == '7190d70afc2317fd9d702e8d999140fd19a54ac476cb65e1980ad59405dbcaa0'
def function(path, name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
assert function(reference_source, 'root_to_world') == function(Path(seq.__file__), 'root_to_world')
provenance = rt.read(root/'input-provenance.json')
frames = next(Path(p) for p in provenance['input_files'] if p.endswith('/data/inputs.json'))
observations = next(Path(p) for p in provenance['input_files'] if p.endswith('/data/observations.npz'))
for p in (frames, observations):
    assert rt.sha(p) == provenance['input_files'][str(p)]
started = time.monotonic()
inputs = seq.load_inputs(frames, observations, camera_length_contract=provenance['source_camera_length_contract'],
                         reference_frame=provenance['reference_frame_id'])
assert not torch.cuda.is_available(), 'This check must remain CPU only.'
body = smplx.SMPLX('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz',
    batch_size=1, num_betas=10, num_expression_coeffs=10, use_pca=True, num_pca_comps=12,
    flat_hand_mean=False, use_face_contour=False, create_body_pose=False, create_transl=False,
    dtype=torch.float32)
basis = seq.pelvis_basis(body)
with np.load(root/'fit/fitted.npz', allow_pickle=False) as z:
    reference = {key: z[key].copy() for key in z.files}
winners = {}
for frame in inputs['ids']:
    folder = root/'frames'/frame
    winners[frame] = dict(npz=str(folder/'winner-camera-space.npz'),
        npz_sha256=rt.sha(folder/'winner-camera-space.npz'), result=str(folder/'result.json'),
        result_sha256=rt.sha(folder/'result.json'))
print(json.dumps(dict(inputs_checked=len(inputs['ids']), device='CPU', body_forwards=0)), flush=True)
output = Path(__file__).parent/'official-named-assembly'
receipt = seq.assemble_saved_winners(inputs=inputs, root=output, winners=winners,
    shared_betas=reference['shared_betas'], basis=basis)
with np.load(output/'fit/fitted.npz', allow_pickle=False) as z:
    actual = {key: z[key].copy() for key in z.files}
assert set(actual) == set(reference), (set(actual)-set(reference), set(reference)-set(actual))
equal = {key: np.array_equal(actual[key], reference[key]) for key in reference}
errors = {key: float(np.max(np.abs(actual[key]-reference[key]))) for key in reference
          if actual[key].dtype.kind in 'fiu'}
rows = rt.read(output/'data/training-inputs.json')['records']
old_rows = rt.read(root/'data/training-inputs.json')['records']
assert len(rows) == len(old_rows) == 94
assert all(a == b for a, b in zip(rows, old_rows))
result = dict(state='passed' if all(equal.values()) else 'assembly_difference',
    historical_readonly_assembly=True, frames=94, fields=len(equal), all_fields_exact=all(equal.values()),
    fields_exact=equal, numeric_max_errors=errors, training_records_exact=True,
    root_to_world_AST_identical=True, sources_unchanged=True, GPU_executions=0, body_forwards=0,
    fitting_calls=0, gaussian_steps=0, receipt=str(output/'fit/official-fit-result.json'),
    seconds=time.monotonic()-started, quality_accepted=False, server_verified=False)
seq.check_bound(inputs['bound'])
rt.save(output/'comparison.json', result)
print(json.dumps({k: result[k] for k in ['state','frames','fields','all_fields_exact','training_records_exact',
                                      'GPU_executions','fitting_calls','seconds']}), flush=True)
assert all(equal.values()), errors
