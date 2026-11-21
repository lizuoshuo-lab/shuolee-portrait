from pathlib import Path
import hashlib, json, shutil

WIN = Path('D:/00/工作/人体重建')
P = WIN if WIN.exists() else Path('/mnt/d/00/工作/人体重建')
DATA=P/'data/hover175'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def local(s):
    if P==WIN:return Path(s)
    return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/'))
if __name__=='__main__':
    import numpy as np
    plan=read(P/'provenance/route113-extract-plan.json')
    with np.load(local(plan['source'])) as z:
        np.savez_compressed(local(plan['output']),**{k:z[k] for k in plan['keys']})
    assets=local('D:/GS-SLAM-Data/shuolee-113-assets')
    old=local('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets')
    lab=local('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
    files={
      'smplx/SMPLX_NEUTRAL.npz':old/'smplx/SMPLX_NEUTRAL.npz',
      'gvhmr/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt':old/'gvhmr/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',
      'gvhmr/hmr4d/utils/body_model/smplx2smpl_sparse.pt':old/'gvhmr/hmr4d/utils/body_model/smplx2smpl_sparse.pt',
      'models/flame2023/flame2023.pkl':lab/'outputs/mini-splatting2-mip-depth-001/shuolee/current/novel-view-face/head-structure/private-runtime/models/flame2023/flame2023.pkl',
      'models/flame/landmark_embedding_with_eyes.npy':lab/'repos/GaussianAvatars_SMPLX-shuolee/flame_model/assets/flame/landmark_embedding_with_eyes.npy',
      'models/flame/head_template_mesh.obj':lab/'repos/GaussianAvatars_SMPLX-shuolee/flame_model/assets/flame/head_template_mesh.obj',
    }
    for f in (old/'vposer/V02_05').rglob('*'):
        if f.is_file() and f.suffix in ['.yaml','.ckpt']:files[f.relative_to(old).as_posix()]=f
    asset_manifest=[]
    for name,src in files.items():
        dest=assets/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(src)
        else:shutil.copy2(src,dest)
        asset_manifest.append(dict(path=name,bytes=dest.stat().st_size,sha256=sha(dest)))
    input_names=set(['inputs.json','training-inputs.json','observations.npz','hand-observations.npz',
        'inference/prediction-postprocessed.npz','inference/time-camera-inputs.npz','inference/vitpose-keypoints.pt',
        'inspection/floor-reference.json','inspection/surface-support.npz','inspection/route-comparison.json'])
    for r in read(DATA/'training-inputs.json')['records']:
        input_names.update([r['image'],r['mask']])
    input_names.update(f.relative_to(DATA).as_posix() for f in (DATA/'route113').rglob('*') if f.is_file())
    data_manifest=[dict(path=n,bytes=(DATA/n).stat().st_size,sha256=sha(DATA/n)) for n in sorted(input_names)]
    doc=dict(route='113',fixed_existing_dataset=True,raw_video_to_frozen_inputs_not_automated=True,
        camera_source='074 full-scene COLMAP estimates; not device calibration',
        shared_observation_keys=plan['keys'],data_files=data_manifest,asset_files=asset_manifest,
        trained_person_geometry_or_gaussian_checkpoint_is_not_an_input=True,
        model_license='User-provided authorized assets; do not redistribute in Git or reports.')
    (P/'configs/route113-inputs.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sources=read(P/'provenance/route113-sources.json')
    sources['portability_changes'].append('Run final conditional-volume preflight without requiring the archived failed plane trial; controls and support thresholds unchanged.')
    for e in sources['files']:
        e['portable_sha256']=sha(P/e['path']);e['modified_for_portability']=e['sha256']!=e['portable_sha256']
    (P/'provenance/route113-sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(data_files=len(data_manifest),data_bytes=sum(x['bytes'] for x in data_manifest),asset_files=len(asset_manifest),asset_bytes=sum(x['bytes'] for x in asset_manifest),assets=str(assets))))
