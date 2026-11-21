from pathlib import Path
import json, shutil, hashlib, subprocess, ast
from datetime import datetime, timezone, timedelta

P=Path('D:/00/工作/人体重建')
LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
SRC=LAB/'repos/GaussianAvatars_SMPLX-shuolee'
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
PAIR=BASE/'current/avatar-repository-pair-114'
DATA=P/'data/hover175'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def loc(p):
    s=str(p).replace('\\','/')
    return Path(s[5]+':'+s[6:]) if s.startswith('/mnt/') else Path(s)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
records=[]
def cp(a,b,kind):
    a=loc(a);b=Path(b);b.parent.mkdir(parents=True,exist_ok=True)
    digest=sha(a)
    if not b.exists() or b.stat().st_size!=a.stat().st_size or sha(b)!=digest:shutil.copy2(a,b)
    records.append(dict(path=b.relative_to(P).as_posix(),bytes=b.stat().st_size,sha256=digest,kind=kind))
def code_tree(a,b,kind):
    exts={'.py','.cu','.cuh','.h','.hpp','.c','.cpp','.inl','.cmake','.txt','.md','.toml','.yml','.yaml','.json','.sh'}
    for f in a.rglob('*'):
        rel=f.relative_to(a)
        if any(x in {'.git','__pycache__','build','dist','.venv','outputs'} or x.endswith('.egg-info') for x in rel.parts):continue
        if f.is_file() and (f.suffix.lower() in exts or f.name in ['LICENSE','COPYING','CMakeLists.txt']):cp(f,b/rel,kind)

for d in ['arguments','scene','utils','gaussian_renderer','flame_model','mesh_renderer','lpipsPyTorch','submodules']:
    code_tree(SRC/d,P/'src/avatar_core'/d,'upstream_or_existing_local_code')
for f in ['avatar_pair_114.py','fit_avatar_pair_static_114.py','shuolee_common.py','shuolee_vposer.py','floor_reference_cameras.py','export_viewer_upright.py','compatible_gaussian_export.py','LICENSE.md','LICENSE_GS.md']:
    cp(SRC/f,P/'src/avatar_core'/f,'existing_local_adapter' if f.endswith('.py') else 'license')
# Exporter only needs this image conversion helper; avoid importing an unrelated trainer.
export=P/'src/avatar_core/export_viewer_upright.py'
s=export.read_text('utf-8').replace('from train_smplx import pil','def pil(t):\n    return Image.fromarray((t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8))')
export.write_text(s,encoding='utf-8')
for folder in ['model','submodules/smplx']:
    code_tree(LAB/'repos/GaussianAvatar'/folder,P/'third_party/repos/GaussianAvatar'/folder,'official_GaussianAvatar_code')
for f in (LAB/'repos/GaussianAvatar').glob('*LICENSE*'):cp(f,P/'third_party/repos/GaussianAvatar'/f.name,'license')
code_tree(LAB/'repos/human_body_prior-shuolee/human_body_prior',P/'third_party/human_body_prior/human_body_prior','official_VPoser_code')
cp(LAB/'repos/human_body_prior-shuolee/LICENSE',P/'third_party/human_body_prior/LICENSE','license')

for manifest in ['inputs.json','training-inputs.json']:
    doc=read(PAIR/'smplx/data'/manifest)
    for i,row in enumerate(doc['records']):
        for key,folder in [('image','images'),('mask','masks')]:
            if row[key] is None:continue
            src=loc(row[key]);dst=DATA/folder/f'{int(row["source_frame_index"]):06d}{src.suffix.lower()}'
            cp(src,dst,'original_observation')
            row[key]=dst.relative_to(DATA).as_posix()
    save(DATA/manifest,doc)
cp(PAIR/'smplx/data/observations.npz',DATA/'observations.npz','original_detector_observation')
for name in ['prediction-postprocessed.npz','time-camera-inputs.npz','vitpose-keypoints.pt']:
    cp(PAIR/'smplx/inference'/name,DATA/'inference'/name,'original_pretrained_detector_output')
cp(BASE/'current/smplx-hand-depth-fresh-011/hand-observations.npz',DATA/'hand-observations.npz','original_pretrained_hand_observation')
for name in ['floor-reference.json','surface-support.npz']:
    cp(BASE/'current/smplx-observation-support-014'/name,DATA/'inspection'/name,'read_only_inspection_reference')
cp(BASE/'current/smplx-whole-body-prior-fresh-006/gaussianavatar-comparison/route-comparison.json',DATA/'inspection/route-comparison.json','read_only_inspection_reference')
cp(BASE/'current/novel-view-face/head-structure/private-runtime/models/smplx/SMPLX_NEUTRAL.npz',P/'assets/smplx/SMPLX_NEUTRAL.npz','licensed_template_not_git')
for f in Path('D:/GS-SLAM-Data/models/vposer/V02_05').rglob('*'):
    if f.is_file() and f.suffix.lower() in ['.yaml','.yml','.ckpt']:cp(f,P/'assets/vposer/V02_05'/f.relative_to('D:/GS-SLAM-Data/models/vposer/V02_05'),'licensed_prior_not_git')
for name in ['smplx2smpl_sparse.pt','smpl_coco17_J_regressor.pt']:
    cp(LAB/'repos/GVHMR-shuolee/hmr4d/utils/body_model'/name,P/'assets/gvhmr/hmr4d/utils/body_model'/name,'body_mapping_not_git')
cfg={'dataset':'data/hover175','smplx_model':'assets/smplx/SMPLX_NEUTRAL.npz','vposer':'assets/vposer/V02_05','gvhmr_maps':'assets/gvhmr','frames':175,'initial_points':200000,'iterations':18000,'fit_steps':1200,'seed':6060,'training_height':1536,'static_shared_person':True,'camera_source':'074 COLMAP estimated full-frame cameras; physical calibration accuracy not accepted','camera_optimization':False,'resume_from_previous_experiment':False,'quality_accepted':False,'note':'Current material snapshot; local static adaptations, not complete official benchmark reproduction'}
save(P/'configs/hover175.json',cfg)
lock={'GaussianAvatar':{'url':'https://github.com/aipixel/GaussianAvatar','commit':'d981c62238ef64e89dcc04719d2ebbb4758b080a'},'GaussianAvatars_SMPLX':{'url':'https://github.com/jmaraval/GaussianAvatars_SMPLX','commit':'e44c1a6c5fbb7f76089f14982eba98d94e553214','missing_upstream':'train_smplx.py and scene/smplx_gaussian_model.py; this package includes explicitly local adapters'},'human_body_prior':{'url':'https://github.com/nghorbani/human_body_prior','commit':'6b883ecc3000697409a8caa2744523afe041e48a'}}
save(P/'upstream.lock.json',lock)
unique={r['path']:r for r in records}
save(P/'configs/assets-manifest.json',{'files':[v for v in unique.values() if v['path'].startswith(('assets/','data/'))],'learned_person_fit_included':False,'old_gaussians_included':False})
save(P/'provenance/source-manifest.json',{'created':datetime.now(timezone(timedelta(hours=8))).isoformat(),'files':[v for v in unique.values() if v['path'].startswith(('src/','third_party/'))],'modified_exporter':'Only removed the dependency on an unrelated train_smplx image helper; no Gaussian transformation equations changed.'})
(P/'.gitignore').write_text('/assets/\n/data/\n/runs/\n/outputs/\n/models/\n/weights/\n/private/\n*.ply\n*.splat\n*.spz\n*.pt\n*.pth\n*.ckpt\n*.pkl\n*.npz\n*.mp4\n*.mov\n*.zip\n*.so\n*.pyd\n*.dll\n*.exe\n.env\n.env.*\n!.env.example\n__pycache__/\n.venv/\nbuild/\n*.egg-info/\n',encoding='utf-8')
(P/'requirements.txt').write_text('numpy==1.26.4\nscipy==1.14.1\npillow==11.1.0\nplyfile==1.1\nsmplx==0.1.28\nomegaconf==2.3.0\nopencv-python-headless==4.10.0.84\nroma\ntqdm==4.70.1\nninja\nchumpy\neinops\ntrimesh\n',encoding='utf-8')
state=read(Path('D:/00/整理状态.json'));state.update(status='preparing_materials',wps_sync='D:/00 configured by user; direct folder sync selected; separate round-trip waived',current_priority='Prepare the three coherent projects and per-route reports. Git private transport verified. WPS follows D:/00 directly per user instruction.');save(Path('D:/00/整理状态.json'),state)
transfer=read(Path('D:/00/传送验证/传送状态.json'));transfer.update(wps_configuration='User confirmed D:/00 is the selected sync folder',wps_download_requirement='waived_by_user; do not ask again',next_step='Prepare material in D:/00; WPS client handles directory synchronization',wps_roundtrip_claim=False);save(Path('D:/00/传送验证/传送状态.json'),transfer)
print(json.dumps({'project':str(P),'files':len(unique),'bytes':sum(r['bytes'] for r in unique.values()),'trained_subject_parameters_copied':False},ensure_ascii=False))
