"""Bounded CPU-only COLMAP reconstruction from 80 Truck train images.

Published calibration supplies only an explicitly disclosed intrinsic prior.
No input poses, sparse points, database or matches seed the new mapper.
Camera comparison uses filename correspondence and Sim(3), not point-ID reuse.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/large-scene-gs-001'
REPO=ROOT/'repos/hierarchical-3d-gaussians'
TOOL=Path('C:/Users/Administrator/colmap-x64-windows-cuda')
EXE=TOOL/'bin/colmap.exe'
DATA=ROOT/'data/large-scene-gs/tandt/truck'
sys.path.insert(0,str(REPO/'preprocess'))
import read_write_model as rw


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def similarity(source,target):
    mx,my=source.mean(axis=0),target.mean(axis=0)
    x,y=source-mx,target-my
    u,values,vt=np.linalg.svd(y.T@x/len(x))
    signs=np.ones(3);signs[-1]=np.linalg.det(u@vt)
    rotation=u@np.diag(signs)@vt
    scale=float(np.sum(values*signs)/np.mean(np.sum(x*x,axis=1)))
    if scale<=0:
        raise ValueError('Invalid Sim(3) alignment')
    translation=my-scale*rotation@mx
    return scale,rotation,translation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='truck-cpu-sfm-pilot-001')
    parser.add_argument('--reuse-features-from')
    parser.add_argument('--matching-threads',type=int,default=4)
    parser.add_argument('--matching-block-size',type=int,default=50)
    args=parser.parse_args()
    if Path(args.run).name!=args.run or args.matching_threads<1 or args.matching_block_size<2:
        raise ValueError('Invalid bounded pilot arguments')
    output=BASE/'sfm'/args.run
    output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env['PATH']=str(TOOL/'bin')+os.pathsep+env.get('PATH','')
    env['QT_PLUGIN_PATH']=str(TOOL/'plugins')
    version=subprocess.check_output([str(EXE),'-h'],env=env,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    (output/'binary-help.txt').write_text(version,encoding='utf-8')
    split=json.loads((BASE/'training/truck-baseline-001/split.json').read_text(encoding='utf-8'))
    names=np.array(sorted(split['train_names']))[np.linspace(0,len(split['train_names'])-1,80,dtype=int)].tolist()
    if len(set(names))!=80 or set(names)&set(split['test_names']):
        raise ValueError('Invalid selected train subset')
    image_list=output/'selected-images.txt'
    image_list.write_text('\n'.join(names)+'\n',encoding='utf-8')
    with Image.open(DATA/'images'/names[0]) as image:
        width,height=image.size
    cams=rw.read_cameras_binary(str(DATA/'sparse/0/cameras.bin'))
    if len(cams)!=1:
        raise ValueError('Pilot expects one shared calibrated camera')
    reference_camera=next(iter(cams.values()))
    if reference_camera.model!='PINHOLE':
        raise ValueError('Expected PINHOLE reference prior')
    params=reference_camera.params*np.array([width/reference_camera.width,height/reference_camera.height,width/reference_camera.width,height/reference_camera.height])
    database=output/'database.db'
    model=output/'sparse';model.mkdir()
    commands=[
        ['feature_extractor','--database_path',str(database),'--image_path',str(DATA/'images'),
         '--image_list_path',str(image_list),'--ImageReader.camera_model','PINHOLE','--ImageReader.single_camera','1',
         '--ImageReader.camera_params',','.join(format(v,'.17g') for v in params),
         '--SiftExtraction.use_gpu','0','--SiftExtraction.num_threads','4','--SiftExtraction.max_num_features','4096'],
        ['exhaustive_matcher','--database_path',str(database),'--SiftMatching.use_gpu','0','--SiftMatching.num_threads',str(args.matching_threads),
         '--ExhaustiveMatching.block_size',str(args.matching_block_size)],
        ['mapper','--database_path',str(database),'--image_path',str(DATA/'images'),'--output_path',str(model),
         '--Mapper.num_threads','4','--Mapper.ba_use_gpu','0','--Mapper.ba_refine_focal_length','1',
         '--Mapper.ba_refine_principal_point','0','--Mapper.ba_refine_extra_params','0'],
    ]
    feature_reuse=None
    if args.reuse_features_from:
        source=Path(args.reuse_features_from).resolve()
        source_config=json.loads((source/'configuration.json').read_text(encoding='utf-8'))
        expected_hashes={name:digest(DATA/'images'/name) for name in names}
        if source_config['selected_names']!=names or source_config['input_images_sha256']!=expected_hashes:
            raise ValueError('Feature source input membership/hash mismatch')
        if not np.array_equal(np.array(source_config['intrinsic_prior']),params):
            raise ValueError('Feature source intrinsic prior mismatch')
        source_hash=digest(source/'database.db')
        with sqlite3.connect('file:'+str(source/'database.db')+'?mode=ro',uri=True) as original:
            with sqlite3.connect(database) as copied:
                original.backup(copied)
                removed={table:copied.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['matches','two_view_geometries']}
                copied.execute('DELETE FROM matches')
                copied.execute('DELETE FROM two_view_geometries')
                copied.commit()
                features={table:copied.execute('SELECT COUNT(*),SUM(rows) FROM '+table).fetchone() for table in ['keypoints','descriptors']}
                if any(row[0]!=80 or row[1] is None for row in features.values()):
                    raise ValueError('Incomplete source features')
        if digest(source/'database.db')!=source_hash:
            raise ValueError('Source database unexpectedly changed')
        feature_reuse=dict(source=str(source),source_database_sha256=source_hash,
            source_configuration_sha256=digest(source/'configuration.json'),sqlite_backup=True,
            discarded_partial_matches=removed,retained_feature_counts=features,
            source_database_unchanged=True,reason='Retry after native matching access violation; no prior matches or geometries carried forward')
        commands=commands[1:]
    config=dict(started_utc=datetime.now(timezone.utc).isoformat(),script_sha256=digest(__file__),
        binary_sha256=digest(EXE),binary_reported_version=version.splitlines()[0],binary_origin='Existing locally installed COLMAP; package origin not independently attested',
        selected_names=names,image_size=[width,height],input_images_sha256={name:digest(DATA/'images'/name) for name in names},
        intrinsic_prior=params.tolist(),intrinsic_prior_source='Published full-scene COLMAP PINHOLE scaled to real input image dimensions',
        prior_limit='Reconstruction uses public intrinsic prior, not fully blind calibration; no source poses/points/matches/database supplied',
        CPU_flags=f'SIFT extraction GPU0/four CPU threads (or validated reuse), SIFT matching GPU0/{args.matching_threads} CPU threads/block {args.matching_block_size}, mapper BA GPU0/four CPU threads; COLMAP binary itself was built with CUDA',
        feature_reuse=feature_reuse,
        commands=[[str(EXE)]+command for command in commands],reference='Published SfM is a reconstruction reference, not ground truth',
        thread_contention='Runs in parallel with separate GPU GS training; elapsed CPU stage time is not an isolated throughput benchmark')
    save(output/'configuration.json',config)
    outcome=dict(status='running_CPU_SfM',stages=[])
    save(output/'outcome.json',outcome)
    try:
        for command in commands:
            start=time.perf_counter()
            stage=dict(name=command[0],status='running',argv=[str(EXE)]+command)
            outcome['stages'].append(stage);save(output/'outcome.json',outcome)
            with (output/(command[0]+'.log')).open('wb') as log:
                code=subprocess.call([str(EXE)]+command,env=env,stdout=log,stderr=subprocess.STDOUT)
            stage.update(exit_code=code,elapsed_seconds=time.perf_counter()-start,status='completed' if code==0 else 'failed')
            save(output/'outcome.json',outcome)
            print(json.dumps(stage),flush=True)
            if code:
                raise RuntimeError('COLMAP CPU stage failed')
        with sqlite3.connect(database) as connection:
            registered_names=[r[0] for r in connection.execute('SELECT name FROM images')]
            if set(registered_names)!=set(names):
                raise ValueError('Database image membership changed')
            database_audit=dict(images=len(registered_names),keypoint_rows=connection.execute('SELECT SUM(rows) FROM keypoints').fetchone()[0],
                verified_image_pairs=connection.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>0').fetchone()[0])
        models=[]
        for directory in sorted(model.iterdir()):
            if directory.is_dir() and (directory/'images.bin').exists():
                cc,ii,pp=rw.read_model(str(directory),ext='.bin')
                models.append((directory,cc,ii,pp))
        if not models:
            raise RuntimeError('Mapper emitted no registered model')
        chosen=max(models,key=lambda m:len(m[2]))
        directory,cc,ii,pp=chosen
        by_name={v.name:v for v in ii.values()}
        if not set(by_name)<=set(names):
            raise ValueError('Unexpected image reconstructed')
        geometry=np.load(BASE/'training/truck-baseline-001/parser_geometry.npz')
        references={name:c for name,c in zip(geometry['image_names'].tolist(),geometry['camtoworlds'])}
        ordered=sorted(by_name)
        c2w=[]
        for name in ordered:
            image=by_name[name];w2c=np.eye(4);w2c[:3,:3]=image.qvec2rotmat();w2c[:3,3]=image.tvec
            c2w.append(np.linalg.inv(w2c))
        c2w=np.array(c2w)
        ref=np.array([references[name] for name in ordered])
        scale,rotation,translation=similarity(c2w[:,:3,3],ref[:,:3,3])
        centers=scale*c2w[:,:3,3]@rotation.T+translation
        distances=np.linalg.norm(centers-ref[:,:3,3],axis=1)
        aligned_rotation=rotation@c2w[:,:3,:3]
        differences=aligned_rotation@ref[:,:3,:3].transpose(0,2,1)
        angles=np.arccos(np.clip((np.trace(differences,axis1=1,axis2=2)-1)/2,-1,1))
        residuals=[];negative=0
        for image in ii.values():
            camera=cc[image.camera_id]
            valid=[j for j,pid in enumerate(image.point3D_ids) if pid in pp]
            if not valid:continue
            xyz=np.stack([pp[image.point3D_ids[j]].xyz for j in valid])@image.qvec2rotmat().T+image.tvec
            negative+=int((xyz[:,2]<=0).sum())
            fx,fy,cx,cy=camera.params
            uv=xyz[:,:2]/xyz[:,2,None]*np.array([fx,fy])+np.array([cx,cy])
            residuals.extend(np.linalg.norm(uv-image.xys[valid],axis=1).tolist())
        residuals=np.array(residuals)
        if not np.isfinite(residuals).all():raise ValueError('Nonfinite fresh reprojection')
        metrics=dict(status='completed_actual_CPU_SfM_with_intrinsic_prior',database=database_audit,
            model_count=len(models),models=[dict(path=str(m[0]),registered_images=len(m[2]),points=len(m[3])) for m in models],
            selected_model=str(directory),selected_images=80,registered_images=len(ii),unregistered_names=sorted(set(names)-set(by_name)),points=len(pp),
            independently_projected_observations=len(residuals),nonpositive_depth_observations=negative,
            fresh_reprojection_pixel_error=dict(median=float(np.median(residuals)),p95=float(np.percentile(residuals,95)),mean=float(residuals.mean())),
            fitted_Sim3=dict(scale=scale,rotation=rotation.tolist(),translation=translation.tolist()),
            reference_camera_center_error_normalized_scene_units=dict(median=float(np.median(distances)),p95=float(np.percentile(distances,95)),max=float(distances.max())),
            reference_orientation_error_degrees=dict(median=float(np.median(np.degrees(angles))),p95=float(np.percentile(np.degrees(angles),95))),
            reference_limit='Public full-scene SfM is not GT. Sim(3) fitted on matched camera centers; normalized distances are not meters. Only this 80-image CPU pilot is newly reconstructed.',
            original_dataset_unchanged=True,source_model_hashes={name:digest(directory/name) for name in ['cameras.bin','images.bin','points3D.bin']})
        save(output/'metrics.json',metrics)
        np.savez_compressed(output/'aligned-camera-comparison.npz',names=ordered,c2w=c2w,reference_c2w=ref,aligned_centers=centers,
                            scale=scale,rotation=rotation,translation=translation)
        outcome.update(status=metrics['status'],selected_model=str(directory),metrics_sha256=digest(output/'metrics.json'))
        save(output/'outcome.json',outcome)
        print(json.dumps({k:metrics[k] for k in ['status','registered_images','points','fresh_reprojection_pixel_error']}),flush=True)
    except Exception as exc:
        outcome.update(status='failed_CPU_SfM_preserved',error_type=type(exc).__name__,error=str(exc))
        save(output/'outcome.json',outcome)
        raise


if __name__=='__main__':
    main()
