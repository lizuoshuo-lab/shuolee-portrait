"""CPU-only post-update baseline export and train-only Creator pose archive."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACK = ROOT/'outputs/large-scene-gs-001'
sys.path.insert(0,str(TRACK/'hierarchy-build'))
from prepare_creator_ply import convert

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run',default='truck-baseline-001')
    p.add_argument('--output-run',default='truck-baseline-final-hierarchy-001')
    a = p.parse_args()
    assert all(s.replace('-','').replace('_','').isalnum() for s in (a.training_run,a.output_run))
    training = TRACK/'training'/a.training_run
    configuration = json.loads((training/'configuration.json').read_text(encoding='utf-8'))
    outcome = json.loads((training/'outcome.json').read_text(encoding='utf-8'))
    assert outcome['status'] == 'training_completed' and configuration['gsplat_commit'] == '937e29912570c372bed6747a5c9bf85fed877bae'
    assert configuration['config']['sh_degree'] == 3
    state_path = training/'final_state.pt'
    assert digest(state_path) == outcome['final_state_sha256']
    import torch
    state = torch.load(state_path,map_location='cpu',weights_only=True)
    assert state['state_timing'] == 'after final optimizer and densification updates'
    assert state['completed_updates'] == outcome['completed_updates'] == 30000
    splats = state['splats']
    n = len(splats['means'])
    assert n == outcome['gaussian_count']
    for name in ('means','scales','quats','opacities','sh0','shN'):
        assert splats[name].device.type == 'cpu' and splats[name].dtype == torch.float32 and torch.isfinite(splats[name]).all()
    assert splats['shN'].shape == (n,15,3)
    output = TRACK/'hierarchy-build/real'/a.output_run
    output.mkdir(parents=True,exist_ok=False)
    raw_ply = output/'post-update-gsplat-schema.ply'
    names = ['x','y','z'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)] + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    with raw_ply.open('wb') as f:
        header = ['ply','format binary_little_endian 1.0',f'element vertex {n}']+[f'property float {name}' for name in names]+['end_header']
        f.write(('\n'.join(header)+'\n').encode('ascii'))
        for start in range(0,n,65536):
            end = min(start+65536,n)
            def arr(name):
                return splats[name][start:end].numpy()
            values = np.concatenate([arr('means'),arr('sh0').reshape(end-start,3),arr('shN').transpose(0,2,1).reshape(end-start,45),
                                     arr('opacities')[:,None],arr('scales'),arr('quats')],axis=1)
            f.write(values.astype('<f4',copy=False).tobytes())
    creator_ply = output/'post-update-creator-degree3.ply'
    layout_receipt = convert(raw_ply,creator_ply)
    with np.load(training/'parser_geometry.npz',allow_pickle=False) as z:
        parsed = {key:z[key].copy() for key in ('image_names','camera_ids','camtoworlds','K','train_indices','test_indices','global_transform')}
    split = json.loads((training/'split.json').read_text(encoding='utf-8'))
    image_rows = {row['name']:row for row in json.loads((TRACK/'data-preparation/image-audit.json').read_text(encoding='utf-8'))}
    selected = parsed['train_indices'].astype(int)
    assert parsed['image_names'][selected].tolist() == split['train_names']
    assert parsed['image_names'][parsed['test_indices'].astype(int)].tolist() == split['test_names']
    manifest = json.loads((TRACK/'data-preparation/input-manifest.json').read_text(encoding='utf-8'))
    with np.load(manifest['colmap_audit']['global_geometry_arrays_path'],allow_pickle=False) as z:
        image_ids = z['image_ids'].copy()
        assert np.array_equal(z['image_names'],parsed['image_names'])
    reference_path = ROOT/'repos/hierarchical-3d-gaussians/preprocess/read_write_model.py'
    spec = importlib.util.spec_from_file_location('hierarchy_colmap_reference',reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cameras = {}
    images = []
    errors = []
    for index in selected:
        cid = int(parsed['camera_ids'][index])
        name = str(parsed['image_names'][index])
        row = image_rows[name]
        K = parsed['K'][index]
        assert np.isfinite(K).all() and K[0,0] > 0 and K[1,1] > 0
        params = np.array([K[0,0],K[1,1],K[0,2],K[1,2]],dtype=np.float64)
        camera = (row['resized_width'],row['resized_height'],params)
        if cid in cameras:
            old = cameras[cid]
            assert old[:2] == camera[:2] and np.array_equal(old[2],params)
        cameras[cid] = camera
        w2c = np.linalg.inv(parsed['camtoworlds'][index])
        qvec = reference.rotmat2qvec(w2c[:3,:3])
        R = reference.qvec2rotmat(qvec)
        translation = w2c[:3,3]
        errors.append(float(np.max(np.abs(R-w2c[:3,:3]))))
        errors.append(float(np.max(np.abs(-R.T@translation-parsed['camtoworlds'][index,:3,3]))))
        assert '\n' not in name and '\0' not in name
        images.append((int(image_ids[index]),cid,name,qvec,translation))
    assert max(errors) < 1e-7
    sparse = output/'creator-train-pose-archive/sparse/0'
    sparse.mkdir(parents=True)
    with (sparse/'cameras.bin').open('wb') as f, (sparse/'cameras.txt').open('w',encoding='utf-8',newline='\n') as text:
        f.write(struct.pack('<Q',len(cameras)))
        text.write('# Derived PINHOLE intrinsics at actual render dimensions; parser global frame.\n')
        for cid,(width,height,params) in sorted(cameras.items()):
            f.write(struct.pack('<iiQQ4d',cid,1,width,height,*params))
            text.write(f'{cid} PINHOLE {width} {height} '+' '.join(format(x,'.17g') for x in params)+'\n')
    with (sparse/'images.bin').open('wb') as f, (sparse/'images.txt').open('w',encoding='utf-8',newline='\n') as text:
        f.write(struct.pack('<Q',len(images)))
        text.write('# Training views only; wxyz world-to-camera poses; no feature observations in this Creator-only archive.\n')
        for iid,cid,name,qvec,tvec in images:
            f.write(struct.pack('<i7di',iid,*qvec,*tvec,cid))
            f.write(name.encode('utf-8')+b'\0'+struct.pack('<Q',0))
            text.write(str(iid)+' '+' '.join(format(x,'.17g') for x in (*qvec,*tvec))+' '+str(cid)+' '+name+'\n\n')
    (sparse/'points3D.bin').write_bytes(struct.pack('<Q',0))
    (sparse/'points3D.txt').write_text('# Pose-only Creator input, not a reconstructed COLMAP training model.\n',encoding='utf-8')
    hierarchy_out = output/'official-creator'
    hierarchy_out.mkdir()
    command = [str(ROOT/'repos/gaussian-hierarchy-large-gs/build-local/Release/GaussianHierarchyCreator.exe'),
               str(creator_ply),str(sparse.parents[1]),str(hierarchy_out)]
    result = dict(status='CPU_preparation_complete_GPU_Creator_not_run',training_run=a.training_run,final_state_sha256=outcome['final_state_sha256'],
                  state_timing=state['state_timing'],completed_updates=state['completed_updates'],gaussian_count=n,layout=layout_receipt,
                  parser_geometry_sha256=digest(training/'parser_geometry.npz'),global_transform=parsed['global_transform'].tolist(),
                  coordinate_scope='PLY unchanged in saved parser frame; poses from saved parser C2W, inverted to W2C. No extra normalization.',
                  camera_count=len(cameras),training_view_count=len(images),excluded_test_view_count=len(split['test_names']),
                  pose_rotation_center_roundtrip_max_abs_error=max(errors),creator_argv=command,
                  pose_archive=str(sparse.parents[1]),pose_archive_scope='Camera-only derivative for Creator filtering; empty observations/points; do not use as training SfM.',
                  files=[dict(path=str(f),sha256=digest(f),bytes=f.stat().st_size) for f in sorted(sparse.iterdir())],
                  reference_quaternion_conversion=dict(path=str(reference_path),sha256=digest(reference_path),commit='596dc7081ee74e57a66502f4746328911f763b34'),
                  source_ply_difference='Official trainer point_cloud_29999.ply was saved before final optimizer/strategy update and is deliberately not used here.')
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','gaussian_count','training_view_count','pose_rotation_center_roundtrip_max_abs_error','creator_argv')}))

if __name__ == '__main__':
    main()
