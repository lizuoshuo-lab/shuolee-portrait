"""Derive same-frame official Hierarchical3DGS inputs without touching originals.

The small Truck experiment uses normalized scene units, not calibrated metres.
Frozen heldout names map one-to-one from JPG metadata to existing PNG images_4.
All calibration is published full-scene SfM; only image-loss updates are held out.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'repos' / 'hierarchical-3d-gaussians'
sys.path.insert(0,str(REPO / 'preprocess'))
import read_write_model as colmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--chunk-size',type=float,default=1.4)
    a = parser.parse_args()
    if not a.run.replace('-','').isalnum() or a.chunk_size<=0:
        raise ValueError('Invalid derived run identifier or chunk size')
    derived = ROOT / 'outputs' / 'large-scene-gs-001' / 'official-inputs' / a.run
    derived.mkdir(parents=True,exist_ok=False)
    baseline = ROOT / 'outputs' / 'large-scene-gs-001' / 'training' / 'truck-baseline-001'
    geometry = np.load(baseline / 'parser_geometry.npz')
    split = json.loads((baseline / 'split.json').read_text(encoding='utf-8'))
    original = ROOT / 'data' / 'large-scene-gs' / 'tandt' / 'truck'
    cams,images,points = colmap.read_model(str(original / 'sparse' / '0'),ext='.bin')
    names = geometry['image_names'].tolist()
    indices = {name:i for i,name in enumerate(names)}
    mapped_names = {name:str(Path(name).with_suffix('.png')).replace('\\','/') for name in names}
    if len(set(mapped_names.values()))!=len(names):
        raise ValueError('Image name collision')
    for image in mapped_names.values():
        if not (original / 'images_4' / image).is_file():
            raise FileNotFoundError(image)
    outcams = {}
    for cid,cam in cams.items():
        k = geometry['K'][np.flatnonzero(geometry['camera_ids']==cid)[0]]
        outcams[cid] = colmap.Camera(cid,'PINHOLE',245,136,np.array([k[0,0],k[1,1],k[0,2],k[1,2]]))
    outimages = {}
    max_camera_error = 0.
    for iid,image in images.items():
        i = indices[image.name]
        w2c = np.linalg.inv(geometry['camtoworlds'][i])
        qvec = colmap.rotmat2qvec(w2c[:3,:3])
        max_camera_error = max(max_camera_error,float(np.max(np.abs(colmap.qvec2rotmat(qvec)-w2c[:3,:3]))))
        oldcam,newcam = cams[image.camera_id],outcams[image.camera_id]
        oldfx,oldfy,oldcx,oldcy = oldcam.params
        newfx,newfy,newcx,newcy = newcam.params
        # Match the audited official parser K, including its odd-size rounding.
        xy = image.xys.copy()
        xy[:,0] = (xy[:,0]-oldcx)*newfx/oldfx+newcx
        xy[:,1] = (xy[:,1]-oldcy)*newfy/oldfy+newcy
        outimages[iid] = colmap.Image(iid,qvec,w2c[:3,3],image.camera_id,mapped_names[image.name],xy,image.point3D_ids)
    transform = geometry['global_transform']
    outpoints = {}
    for pid,point in points.items():
        xyz = point.xyz.astype(np.float32).astype(np.float64) @ transform[:3,:3].T + transform[:3,3]
        outpoints[pid] = colmap.Point3D(pid,xyz,point.rgb,point.error,point.image_ids,point.point2D_idxs)
    sparse = derived / 'aligned' / 'sparse' / '0'
    sparse.mkdir(parents=True)
    colmap.write_model(outcams,outimages,outpoints,str(sparse),ext='.bin')
    test_names = [mapped_names[name] for name in split['test_names']]
    (sparse / 'test.txt').write_text('\n'.join(test_names)+'\n',encoding='utf-8')
    model_manifest = {
        'scope':'Bounded official four-chunk Truck workflow; no metric scale calibration or paper-scale reproduction',
        'baseline_geometry_sha256':hashlib.sha256((baseline / 'parser_geometry.npz').read_bytes()).hexdigest(),
        'coordinate_frame':'One common saved gsplat normalized global Sim(3); no independent chunk normalization',
        'images_absolute':str(original / 'images_4'),
        'image_size':[245,136],'camera_model':'PINHOLE',
        'intrinsics':'Identical to frozen baseline parser K, not independently corrected resize K',
        'camera_rotation_roundtrip_max':max_camera_error,
        'train_names':[mapped_names[n] for n in split['train_names']],
        'test_names':test_names,'original_to_png':mapped_names,
        'chunk_size_normalized_scene_units':a.chunk_size,
        'scaffold_settings':'Official frozen-position SH1 scaffold, default 100000 skybox Gaussians; depth unavailable in small-scene check',
        'no_depth_reason':'No aligned monocular maps acquired yet; no claim of full-paper regularization reproduction',
        'partition_settings':'Unmodified make_chunk.py, random seed0; min_padd0, lapla_thresh0, min_n_cams20, max_n_cams1500; default visibility/random far-camera selection',
    }
    (derived / 'manifest.json').write_text(json.dumps(model_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    command = [sys.executable,str(REPO / 'preprocess' / 'make_chunk.py'),
        '--base_dir',str(sparse),'--images_dir',str(original / 'images_4'),
        '--output_path',str(derived / 'chunks'),'--chunk_size',str(a.chunk_size),
        '--min_padd','0','--lapla_thresh','0','--min_n_cams','20']
    proc = subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    (derived / 'partition.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Official partition failed, see {derived / "partition.log"}')
    chunks = []
    for chunk in sorted((derived / 'chunks').iterdir()):
        cc,ii,pp = colmap.read_model(str(chunk / 'sparse' / '0'),ext='.bin')
        # Partition selection may include test views as calibration context;
        # each trainer still receives the global heldout file and --eval.
        (chunk / 'sparse' / '0' / 'test.txt').write_text('\n'.join(test_names)+'\n',encoding='utf-8')
        available = [i.name for i in ii.values()]
        train = sorted(set(available)-set(test_names))
        if len(train)<10:
            raise ValueError('Too few actual training views in a chunk')
        chunks.append({'name':chunk.name,'train_names':train,'test_names':sorted(set(available)&set(test_names)),
                       'initial_points':len(pp),'center':np.loadtxt(chunk / 'center.txt').tolist(),
                       'extent':np.loadtxt(chunk / 'extent.txt').tolist()})
    if len(chunks)<2:
        raise ValueError('Partition did not create at least two chunks')
    model_manifest['chunks'] = chunks
    model_manifest['train_pair_overlaps'] = [dict(a=x['name'],b=y['name'],views=len(set(x['train_names'])&set(y['train_names'])))
        for i,x in enumerate(chunks) for y in chunks[i+1:]]
    model_manifest['status']='ready_for_official_scaffold_and_chunk_training'
    (derived / 'manifest.json').write_text(json.dumps(model_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(derived),'chunks':[dict(name=c['name'],train=len(c['train_names']),test=len(c['test_names']),points=c['initial_points']) for c in chunks],'camera_roundtrip':max_camera_error}))

if __name__=='__main__':
    main()
