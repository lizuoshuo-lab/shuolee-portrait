"""Fresh full-image SfM for every real training frame; never resumes an old map."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
import numpy as np

LAB = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE = LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
OUT = BASE/'current/full-scene-localization-074'
VIDEO = LAB/'inputs/computer00-local-pipeline/HOVER_VERSA_0102.mp4'
TOOL = Path('C:/Users/Administrator/colmap-x64-windows-cuda')
EXE = TOOL/'bin/colmap.exe'
FFMPEG = Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
sys.path.insert(0, str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
import read_write_model as rw


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, path)


def main():
    OUT.mkdir(exist_ok=False)
    for name in ['images', 'raw_sparse', 'refined_sparse', 'logs']:
        (OUT/name).mkdir()
    prior = json.loads((BASE.parent/'source-resolution-rectified-002/manifest.json').read_text(encoding='utf-8'))
    f = float(prior['original_raw_camera']['params'][0])*2
    protocol = dict(created=stamp(), owner='shuolee', request='把整个场景做定位试一下',
        purpose='Estimate all cameras and the complete visible sparse scene from fresh image features, then test independent foreground and background correspondences.',
        source=str(VIDEO), source_sha256=sha(VIDEO), frame_indices=list(range(1,176)),
        frames='Zero-based raw video indices1..175, full1520x2704 pixels, no sampling interpolation, cropping, person mask, or background compositing.',
        fresh=['Raw-frame decoding','SIFT feature extraction','All175x174/2 image-pair matching','Incremental initialization','All camera poses and scene points','Final global bundle adjustment'],
        reused='Only scalar focal initial guess from previous physical-camera estimate; no old rotation, translation, 3D point, feature, individual body fit, or Gaussian.',
        initial_camera=dict(model='SIMPLE_RADIAL', params=[f,760.,1352.,0.], shared_across_frames=True),
        optimization='Mapper refines shared focal/radial distortion and all poses/points; final global BA additionally releases the shared principal point.',
        intrinsic_prior_is_not_frozen=True,
        software='COLMAP3.11.1 WindowsCUDA commit682ea9a; CPU mapping/BA. GPU SIFT and matching, CPU fallback only on actual executable failure, logged.',
        features=dict(max_image_size=3200,max_num_features=8192,full_frame=True),
        matching=dict(strategy='exhaustive',pair_count=15225,guided_matching=True,ratio=.8,cross_check=True),
        budget=dict(extraction_seconds=300,feature_seconds=900,matching_seconds=1800,mapping_seconds=1800,final_ba_seconds=600),
        checks_before_any_training=['One connected reconstruction containing175/175 distinct raw frames','Finite valid camera matrices and intrinsics','Source-pixel and raw-to-rectified coordinate checks',
            'Same image-only KLT tracks, contiguous held middle thirds: face/background; new image-only body tracks',
            'Native sparse source reprojections and held image correspondences reviewed in the source image pixel system'],
        camera_promotion_screen=dict(face_held_median_reduction_fraction=.2,background_held_p90_max_increase_px=.2,
            body_held_p90_max_increase_px=.5,all_registered=175,positive_depth=True,visual_review_required=True),
        negative_result='Do not promote cameras if only sparse training reprojection improves or face improves while background/body degrades. Partial models do not count as whole-scene localization.',
        limits=['Fresh SfM changes features, point structure, correspondence graph, and poses together; it cannot by itself isolate a single extrinsic cause.',
            'The same monocular video has no metric ground-truth cameras; registration completeness and reprojection are not proof of correct geometry.',
            'A static scene model can still be inconsistent with real small motions, reflections, occlusions, or wrong image correspondences.'],
        new_gaussian_training=False, existing_training_inputs_modified=False,
        code_sha256=sha(__file__))
    save(OUT/'protocol.json',protocol)
    state=dict(created=stamp(),updated=stamp(),owner='shuolee',running=True,status='preparing',
        output=str(OUT),pid=os.getpid(),commands=[],stage_seconds={},training_steps=0,
        camera_estimate_accepted=False,root_cause_established=False)
    env=os.environ.copy()
    env['PATH']=str(TOOL/'bin')+os.pathsep+env.get('PATH','')
    env['QT_PLUGIN_PATH']=str(TOOL/'plugins')
    env['OMP_NUM_THREADS']='8'
    start=time.monotonic()

    def run(label,cmd,timeout,allow_failure=False):
        tick=time.monotonic()
        state.update(status=label,updated=stamp())
        state['commands'].append(dict(label=label,command=[str(x) for x in cmd]))
        save(OUT/'state.json',state)
        print(json.dumps(dict(stage=label,started=stamp())),flush=True)
        with (OUT/'logs'/f'{label}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([str(x) for x in cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid
            save(OUT/'state.json',state)
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
                raise TimeoutError(f'{label} exceeded declared{timeout}s budget')
        state['stage_seconds'][label]=time.monotonic()-tick
        state['commands'][-1]['returncode']=code
        state.pop('child_pid',None)
        save(OUT/'state.json',state)
        if code and not allow_failure:
            raise RuntimeError(f'{label} failed,code={code}; see its preserved log')
        return code

    try:
        run('decode_raw_frames',[FFMPEG,'-hide_banner','-loglevel','warning','-n','-threads','4','-i',VIDEO,
            '-vf','select=between(n\\,1\\,175)','-fps_mode','passthrough','-frames:v','175',
            '-start_number','1',OUT/'images/raw_%06d.png'],300)
        files=sorted((OUT/'images').glob('*.png'))
        assert [p.name for p in files]==[f'raw_{i:06d}.png' for i in range(1,176)]
        from PIL import Image
        images=[]
        for p in files:
            with Image.open(p) as im:
                assert im.size==(1520,2704)
            images.append(dict(name=p.name,source_frame_index=int(p.stem[4:]),sha256=sha(p),wh=[1520,2704]))
        save(OUT/'input-images.json',dict(records=images,video_sha256=protocol['source_sha256']))
        db=OUT/'database.db'
        extraction=[EXE,'feature_extractor','--database_path',db,'--image_path',OUT/'images',
            '--ImageReader.camera_model','SIMPLE_RADIAL','--ImageReader.single_camera','1',
            '--ImageReader.camera_params',f'{f},760,1352,0','--SiftExtraction.use_gpu','1',
            '--SiftExtraction.num_threads','8','--SiftExtraction.max_num_features','8192',
            '--SiftExtraction.max_image_size','3200']
        if run('features_gpu',extraction,900,True):
            extraction[extraction.index('--SiftExtraction.use_gpu')+1]='0'
            run('features_cpu_fallback',extraction,900)
        with sqlite3.connect(db) as conn:
            n=conn.execute('SELECT COUNT(*) FROM keypoints').fetchone()[0]
            assert n==175,f'Only{n} feature records'
            count=conn.execute('SELECT SUM(rows),MIN(rows),MAX(rows) FROM keypoints').fetchone()
            state['feature_count']=dict(total=count[0],min=count[1],max=count[2])
        matching=[EXE,'exhaustive_matcher','--database_path',db,'--SiftMatching.use_gpu','1',
            '--SiftMatching.num_threads','8','--SiftMatching.guided_matching','1',
            '--ExhaustiveMatching.block_size','25']
        if run('matching_gpu',matching,1800,True):
            matching[matching.index('--SiftMatching.use_gpu')+1]='0'
            run('matching_cpu_fallback',matching,1800)
        with sqlite3.connect(db) as conn:
            state['matched_pairs']=conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0]
            state['verified_pairs']=conn.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>=15').fetchone()[0]
        run('mapping',[EXE,'mapper','--database_path',db,'--image_path',OUT/'images','--output_path',OUT/'raw_sparse',
            '--Mapper.num_threads','8','--Mapper.ba_use_gpu','0','--Mapper.max_num_models','5',
            '--Mapper.ba_refine_focal_length','1','--Mapper.ba_refine_principal_point','0',
            '--Mapper.ba_refine_extra_params','1'],1800)
        candidates=[]
        for path in (OUT/'raw_sparse').iterdir():
            if not (path/'images.bin').is_file():continue
            cams,ims,pts=rw.read_model(str(path),ext='.bin')
            candidates.append(dict(path=str(path),registered=len(ims),points=len(pts)))
        assert candidates,'No reconstructed model'
        state['models']=candidates
        best=max(candidates,key=lambda x:x['registered'])
        state['selected_model']=best
        if best['registered']!=175:
            state['status']='partial_registration_not_accepted'
            raise RuntimeError(f'Connected model contains{best["registered"]}/175 frames; no interpolation fallback')
        run('global_bundle_adjustment',[EXE,'bundle_adjuster','--input_path',best['path'],
            '--output_path',OUT/'refined_sparse','--BundleAdjustment.refine_focal_length','1',
            '--BundleAdjustment.refine_principal_point','1','--BundleAdjustment.refine_extra_params','1',
            '--BundleAdjustment.refine_extrinsics','1','--BundleAdjustment.max_num_iterations','100',
            '--BundleAdjustment.use_gpu','0'],600)
        cams,ims,pts=rw.read_model(str(OUT/'refined_sparse'),ext='.bin')
        assert len(ims)==175 and len(cams)==1
        cam=next(iter(cams.values()))
        assert cam.model=='SIMPLE_RADIAL' and np.isfinite(cam.params).all()
        records=[]
        for im in sorted(ims.values(),key=lambda x:x.name):
            E=np.eye(4);E[:3,:3]=rw.qvec2rotmat(im.qvec);E[:3,3]=im.tvec
            assert np.isfinite(E).all() and abs(np.linalg.det(E[:3,:3])-1)<1e-8
            records.append(dict(name=im.name,source_frame_index=int(Path(im.name).stem[4:]),image_id=im.id,
                camera_id=im.camera_id,w2c=E.tolist(),image=str(OUT/'images'/im.name),
                image_sha256=images[int(Path(im.name).stem[4:])-1]['sha256'],
                observation_points=int((im.point3D_ids>=0).sum())))
        assert [r['source_frame_index'] for r in records]==list(range(1,176))
        save(OUT/'cameras.json',dict(model=cam.model,width=cam.width,height=cam.height,
            intrinsics=cam.params.tolist(),records=records,registered=175,interpolation_used=False,
            world_scale='arbitrary monocular scale',accepted_for_training=False,
            source='Fresh full-frame175-imageCOLMAP and global BA; no old extrinsics or 3D points'))
        run('export_scene_sparse',[EXE,'model_converter','--input_path',OUT/'refined_sparse',
            '--output_path',OUT/'whole_scene_sparse.ply','--output_type','PLY'],60)
        state.update(status='reconstruction_complete_pending_independent_evaluation',registered=175,
            points=len(pts),intrinsics=cam.params.tolist(),reprojection_mean_px=float(np.mean([p.error for p in pts.values()])),
            scene_cloud=str(OUT/'whole_scene_sparse.ply'),scene_cloud_type='colored SfM sparse points; not a Gaussian model',
            camera_estimate_accepted=False,quality_accepted=False,new_gaussian_training=False)
    except Exception as exc:
        state.update(status='failed_or_partial_not_accepted',error=repr(exc))
        raise
    finally:
        state.update(running=False,updated=stamp(),elapsed_seconds=time.monotonic()-start)
        save(OUT/'state.json',state)
        print(json.dumps({k:state[k] for k in ['status','running','elapsed_seconds','stage_seconds']}),flush=True)


if __name__=='__main__':
    main()
