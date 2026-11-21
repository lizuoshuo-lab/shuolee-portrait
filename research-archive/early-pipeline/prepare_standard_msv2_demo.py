"""Prepare full native demo frames using the saved dataset's 8-fps/PINHOLE/COLMAP recipe."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,subprocess,sys,traceback
import numpy as np
from PIL import Image

ROOT=Path('D:/GS-SLAM-Data/msv2-demo-fullframe-standard-20261104')
VIDEO=Path('D:/00/工作/数据集/demo.mp4')
FFMPEG=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
COLMAP=Path('C:/Users/Administrator/colmap-x64-windows-cuda/bin/colmap.exe')
PIPELINE=Path('D:/00/工作/minisplatting2/local_pipeline')
OFFICIAL_IO=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/mini-splatting2/scene/colmap_loader.py')
def now():return datetime.now(timezone.utc).isoformat()
def save(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(4*1024**2),b''):h.update(a)
    return h.hexdigest()
def main():
    ROOT.mkdir(exist_ok=False);images=ROOT/'dataset/images';images.mkdir(parents=True);raw=ROOT/'raw-sparse';raw.mkdir()
    state=dict(state='running',started=now(),video=str(VIDEO),video_sha256=sha(VIDEO),commands=[])
    save(ROOT/'state.json',state)
    def run(name,args):
        state.update(stage=name,updated=now());state['commands'].append(list(map(str,args)));save(ROOT/'state.json',state)
        with (ROOT/(name+'.log')).open('w',encoding='utf-8') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        run('extract',[FFMPEG,'-nostdin','-v','warning','-i',VIDEO,'-r','8','-pix_fmt','rgb24','-compression_level','3',images/'%06d.png'])
        paths=sorted(images.glob('*.png'));assert len(paths)>100
        for p in paths:
            with Image.open(p) as im:assert im.size==(2704,1520) and im.mode=='RGB'
        # Identify every exported frame against the untouched video decoder's RGB checksums.
        md5=subprocess.check_output([str(FFMPEG),'-nostdin','-v','error','-i',str(VIDEO),'-pix_fmt','rgb24','-f','framemd5','pipe:1'],text=True)
        (ROOT/'original-video-framemd5.txt').write_text(md5,encoding='utf-8')
        native={};index=0
        for line in md5.splitlines():
            if not line or line.startswith('#'):continue
            fields=[s.strip() for s in line.split(',')];native.setdefault(fields[-1],[]).append(dict(source_index=index,source_pts=int(fields[2])));index+=1
        lineage=[]
        for p in paths:
            rgb=np.asarray(Image.open(p).convert('RGB'));key=hashlib.md5(rgb.tobytes()).hexdigest();assert key in native,p
            lineage.append(dict(image=p.name,sha256=sha(p),rgb_md5=key,matching_original_frames=native[key],wh=[2704,1520]))
        save(ROOT/'pixel-lineage.json',dict(passed=True,output_frames=len(paths),source_frames=index,no_spatial_resizing=True,no_mask=True,records=lineage))
        db=ROOT/'database.db'
        run('features',[COLMAP,'feature_extractor','--database_path',db,'--image_path',images,'--ImageReader.single_camera','1','--ImageReader.camera_model','PINHOLE',
            '--SiftExtraction.use_gpu','0','--SiftExtraction.num_threads','4','--SiftExtraction.estimate_affine_shape','1','--SiftExtraction.domain_size_pooling','1'])
        run('matching',[COLMAP,'sequential_matcher','--database_path',db,'--SiftMatching.guided_matching','1','--SiftMatching.use_gpu','0','--SiftMatching.num_threads','4','--SequentialMatching.overlap','10'])
        run('mapping',[COLMAP,'mapper','--database_path',db,'--image_path',images,'--output_path',raw,'--Mapper.num_threads','4','--Mapper.ba_refine_principal_point','1'])
        finalize(state)
    except BaseException:
        state.update(state='failed',finished=now(),error=traceback.format_exc());save(ROOT/'state.json',state);raise

def finalize(state=None):
    """Read completed COLMAP outputs; never rerun extraction or structure recovery."""
    import shutil
    state=state or json.loads((ROOT/'state.json').read_text(encoding='utf-8'))
    images=ROOT/'dataset/images';paths=sorted(images.glob('*.png'));raw=ROOT/'raw-sparse'
    lineage=json.loads((ROOT/'pixel-lineage.json').read_text(encoding='utf-8'))['records']
    spec=importlib.util.spec_from_file_location('colmap_io',OFFICIAL_IO);io=importlib.util.module_from_spec(spec);spec.loader.exec_module(io)
    models=[]
    for p in sorted(raw.iterdir()):
        if (p/'images.bin').exists():models.append((len(io.read_extrinsics_binary(p/'images.bin')),p))
    assert models;registered,selected=max(models,key=lambda x:x[0]);assert registered>=int(len(paths)*2/3)
    dst=ROOT/'dataset/sparse/0';dst.mkdir(parents=True,exist_ok=True)
    for p in selected.iterdir():
        if p.is_file():shutil.copy2(p,dst/p.name)
    command=[str(COLMAP),'model_analyzer','--path',str(dst)]
    with (ROOT/'model-info.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    state['commands'].append(command)
    intrinsic=io.read_intrinsics_binary(dst/'cameras.bin');extrinsic=io.read_extrinsics_binary(dst/'images.bin')
    rows=[];byname={r['image']:r for r in lineage}
    for name,image in sorted(((v.name,v) for v in extrinsic.values())):
        camera=intrinsic[image.camera_id];assert camera.model=='PINHOLE';fx,fy,cx,cy=camera.params
        k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        w=np.eye(4);w[:3,:3]=image.qvec2rotmat();w[:3,3]=image.tvec
        rows.append(dict(frame=Path(name).stem,name=name,image=str(images/name),wh=[camera.width,camera.height],K=k.tolist(),w2c=w.tolist(),source_sha256=byname[name]['sha256'],source_frame_matches=byname[name]['matching_original_frames']))
    save(ROOT/'camera-inputs.json',dict(records=rows,coordinate_contract='COLMAP pixel-corner coordinates',registered=registered,extracted=len(paths),mask_used=False))
    save(ROOT/'recipe.json',dict(training='msv2 sparse; outdoor (explicit user choice); config/fast; r=1; sampling_factor=0.6',fps=8,camera_model='PINHOLE',sequential_overlap=10,
        estimate_affine_shape=True,domain_size_pooling=True,ba_refine_principal_point=True,
        data_script=str(PIPELINE/'test_preprocess_material.py'),data_script_sha256=sha(PIPELINE/'test_preprocess_material.py'),train_script_sha256=sha(PIPELINE/'test_gs.py'),
        camera_reader=str(OFFICIAL_IO),camera_reader_sha256=sha(OFFICIAL_IO),
        explicit_user_override='Preserve original spatial resolution and enlarge the original complete image; video_r=1, lossless RGB PNG',
        command_compatibility='Structured CLI arguments; correct missing -- prefixes and duplicate mapper argument; CPU SIFT/matching. Use official complete camera reader rather than the count-only dataset helper.',
        original_video_changed=False,masks_used=False,prior_fitted_camera_or_gaussians_loaded=False))
    if 'error' in state:state['reader_issue_before_finalization']=state.pop('error')
    state.update(state='completed',stage='camera_inputs',finished=now(),frames_extracted=len(paths),frames_registered=registered,selected_model=str(selected));save(ROOT/'state.json',state)
    print('STANDARD_DATASET_READY',registered,len(paths),flush=True)

if __name__=='__main__':
    if '--finalize' in sys.argv:finalize()
    else:main()
