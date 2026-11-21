"""Project the two registered 016 cross-face distance extremes to native pixels."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_r23_garment_seams import frames_and_scales
from audit_r23_numerical import LAB, OUTPUT, garment_seed, read_state, _pin, _read


subject = '016'
seed, _, _ = garment_seed(subject)
root = LAB/f'shuolee-smpl-delivery-round23-{subject}'/'run/gaussian-run/raw-step-018000'
state = read_state(root/'main-state.pt')
motion = root/'drive/motion.npz'
_pin(motion, _read(root/'drive/drive.json')['files_sha256']['motion.npz'])
with np.load(motion, allow_pickle=False) as archive:
    data = dict(archive)
centroid, basis, face_scale = frames_and_scales(data['vertices_world'], data['faces'])
ref = data['reference_index'].item()
frames = data['frame_ids'].tolist()
sources = {row['frame']:row for row in _read(seed['paths']['training_inputs'])['records']}
measurements = []
extremes = _read(OUTPUT/f'{subject}-transport-near-pairs.json')['edge_adjacent_owner']['worst']
for arm, item in extremes.items():
    ids = np.asarray(item['ids'])
    i = np.searchsorted(seed['ids'],ids)
    j = np.searchsorted(state['stable_ids'].numpy(),ids)
    assert np.array_equal(seed['ids'][i], ids) and np.array_equal(state['stable_ids'].numpy()[j],ids)
    owners = seed['owner'][i]
    assert np.array_equal(owners,state['support_face_ids'].numpy()[j])
    slot = frames.index(item['frame'])
    source = sources[item['frame']]
    _pin(Path(source['mask']),source['mask_sha256'])
    with Image.open(source['mask']) as img:
        mask = np.array(img.convert('L'))
    values = {}
    for name, centers, logits in (('r22', seed['xyz'][i], seed['opacity'][i]),
                                  ('r23', state['garment'][1].detach().numpy()[j], state['garment'][6].detach().numpy()[j,0])):
        assert np.array_equal(logits, seed['opacity'][i])
        rotation = basis[slot,owners]@basis[ref,owners].swapaxes(-2,-1)
        scale = face_scale[slot,owners]/face_scale[ref,owners]
        world = np.einsum('nij,nj->ni',rotation,centers-centroid[ref,owners])*scale[:,None]+centroid[slot,owners]
        E,K = data['camera_w2c'][slot],data['camera_K'][slot]
        camera = world @ E[:3,:3].T + E[:3,3]
        projected = camera @ K.T
        pixels = projected[:,:2]/projected[:,2,None]-.5
        integer = np.rint(pixels).astype(int)
        assert ((integer>=0)&(integer<data['camera_wh'][slot])).all()
        values[name] = dict(source_pixel_centers=pixels.tolist(),depth_m=camera[:,2].tolist(),
            pixel_separation=float(np.linalg.norm(pixels[0]-pixels[1])),mask_values_8bit=mask[integer[:,1],integer[:,0]].tolist(),
            point_opacity=(1/(1+np.exp(-logits))).tolist())
    measurements.append(dict(worst_arm=arm,frame=item['frame'],ids=ids.tolist(),owners=owners.tolist(),values=values))
print(json.dumps(measurements,indent=2))
