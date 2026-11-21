"""Read the actual worst R24 diagnostic pair in its native fitted view."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from audit_r23_numerical import garment_seed, read_state, _pin, _read
from audit_r24_motion_pairs import endpoint, frames_and_scales, load_motion, native_graph, OUTPUT


subject = sys.argv[1]
worst = _read(OUTPUT/f'{subject}-motion-pairs-18k.json')['shared_edge_owner']['r24_actual_graph']['worst']
path = endpoint(subject,24)
state = read_state(path/'main-state.pt')
motion = load_motion(path)
graph = {key:value.numpy() for key,value in state['surface_graph'].items()}
seed, baseline, _ = garment_seed(subject)
sources = {row['frame']:row for row in _read(baseline/'data/inputs.json')['records']}
row = sources[worst['frame']]
_pin(Path(row['mask']),row['mask_sha256'])
with Image.open(row['mask']) as image:
    mask = np.asarray(image.convert('L'))
origin,basis,length = frames_and_scales(motion['vertices_world'],motion['faces'])
ref = motion['reference_index'].item()
rot = basis@basis[ref].swapaxes(-2,-1)
ids = np.asarray(worst['ids'])
slots = np.searchsorted(state['stable_ids'].numpy(),ids)
assert np.array_equal(state['stable_ids'].numpy()[slots],ids)
points,_ = native_graph(state['garment'][1].detach().numpy()[slots],graph,slots,(origin,rot,length/length[ref]),ref,True)
view = motion['frame_ids'].tolist().index(worst['frame'])
extrinsic = motion['camera_w2c'][view]
camera = points[view] @ extrinsic[:3,:3].T + extrinsic[:3,3]
clip = camera @ motion['camera_K'][view].T
xy = clip[:,:2]/clip[:,2,None]-.5
ij = np.round(xy).astype(int)
alphas = state['garment'][6].detach()[slots].sigmoid().numpy()[:,0]
positive = graph['row_node_weights'][slots] > 0
indices = [a[b] for a,b in zip(graph['row_node_ids'][slots],positive,strict=True)]
print(json.dumps(dict(ids=ids.tolist(),frame=worst['frame'],native_xy=xy.tolist(),
        separation_px=float(np.linalg.norm(xy[0]-xy[1])),alpha=alphas.tolist(),
        mask_8bit=mask[ij[:,1],ij[:,0]].tolist(),camera_depth_m=camera[:,2].tolist(),
        node_ids=[i.tolist() for i in indices],shared_positive_nodes=np.intersect1d(*indices).tolist()),indent=2))
