"""Training-only, fixed-camera sparse initialization for the public height benchmark.

No development/test RGB, masks, author point cloud, or learned prior is read.
COLMAP runs on CPU while the independent work-video GPU experiment continues.
"""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import cv2
import numpy as np
from PIL import Image
from shuolee_audit_diffuman_cameras import LAB, OUT
from shuolee_prepare_elevation_benchmark import digest, dump


def main():
    cv2.setNumThreads(2)
    benchmark=OUT/'elevation-benchmark'
    protocol=json.loads((benchmark/'protocol.json').read_text(encoding='utf-8'))
    qa=json.loads((benchmark/'input-qa.json').read_text(encoding='utf-8'))
    assert qa['test_images_opened'] is False
    dest=benchmark/'training-geometry';dest.mkdir(exist_ok=False)
    images=dest/'images';images.mkdir()
    masks=dest/'feature-masks';masks.mkdir()
    silhouettes=dest/'silhouettes';silhouettes.mkdir()
    validity=dest/'source-validity';validity.mkdir()
    empty=dest/'known-poses';empty.mkdir()
    sparse=dest/'sparse/0';sparse.mkdir(parents=True)
    rows=[]
    state=dict(owner='shuolee',status='rectifying',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        input_protocol_sha256=digest(benchmark/'protocol.json'),frame=protocol['frame'],
        training_cameras=protocol['roles']['train_horizontal'],reads_development_or_test_pixels=False,
        author_sparse_cloud_used=False,learned_prior_used=False,cuda_used=False,
        geometry_fixed='Author camera extrinsics; undistortion changes pixel coordinates/intrinsics only.',
        foreground='Author soft mask composited over black; uncertain silhouette band excluded from feature extraction.',
        commands=[],accepted=False)
    state_file=dest/'manifest.json';dump(state_file,state)
    try:
        for c in qa['rows']:
            name=c['camera'];w,h=c['resolution']
            rgbfile=benchmark/'train_horizontal/images'/name/(protocol['frame']+'.webp')
            maskfile=benchmark/'train_horizontal/fmasks'/name/(protocol['frame']+'.png')
            assert digest(rgbfile)==c['source_sha256'] and digest(maskfile)==c['mask_sha256']
            rgb=np.asarray(Image.open(rgbfile).convert('RGB'))
            alpha=np.asarray(Image.open(maskfile).convert('L'))
            K=np.array(c['K']);newK=K.copy();newK[0,2]=w/2;newK[1,2]=h/2
            mx,my=cv2.initUndistortRectifyMap(K,np.asarray(c['distortion']),None,newK,(w,h),cv2.CV_32FC1)
            valid=((mx>=0)&(my>=0)&(mx<w-1)&(my<h-1)).astype('uint8')*255
            rectified=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            a=cv2.remap(alpha,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            a[valid==0]=0
            composite=np.uint8(np.round(rectified.astype('float32')*(a[:,:,None]/255)))
            feature=cv2.erode((a>250).astype('uint8')*255,np.ones((7,7),np.uint8))
            feature=cv2.bitwise_and(feature,valid)
            # COLMAP's per-image mask name appends .png to the full image filename.
            Image.fromarray(composite).save(images/(name+'.png'))
            Image.fromarray(feature).save(masks/(name+'.png.png'))
            Image.fromarray(a).save(silhouettes/(name+'.png'))
            Image.fromarray(valid).save(validity/(name+'.png'))
            # Forward projection and inverse map must agree numerically.
            probes=np.array([[x,y] for x in np.linspace(80,w-81,7) for y in np.linspace(80,h-81,7)],dtype=np.float64)
            original=np.stack([cv2.remap(mx,probes[:,0,None].astype('float32'),probes[:,1,None].astype('float32'),cv2.INTER_LINEAR).ravel(),
                cv2.remap(my,probes[:,0,None].astype('float32'),probes[:,1,None].astype('float32'),cv2.INTER_LINEAR).ravel()],-1)
            restored=cv2.undistortPoints(original[:,None],K,np.asarray(c['distortion']),P=newK)[:,0]
            projection_error=float(np.max(np.linalg.norm(restored-probes,axis=-1)))
            assert projection_error<.2,(name,projection_error)
            rows.append(dict(camera=name,K=newK.tolist(),w2c=c['w2c'],width=w,height=h,
                source_valid_fraction=float((valid>0).mean()),foreground_fraction=float((a>127).mean()),
                projection_roundtrip_max_px=projection_error,image_sha256=digest(images/(name+'.png')),
                feature_mask_sha256=digest(masks/(name+'.png.png'))))
        assert sorted(p.stem for p in images.glob('*.png'))==sorted(state['training_cameras'])
        tool=Path('C:/Users/Administrator/colmap-x64-windows-cuda');colmap=tool/'bin/colmap.exe'
        env=os.environ.copy();env['PATH']=str(tool/'bin')+os.pathsep+env.get('PATH','');env['QT_PLUGIN_PATH']=str(tool/'plugins')
        db=dest/'database.db'
        def run(label,cmd):
            state.update(status=label,rows=rows);state['commands'].append([str(x) for x in cmd]);dump(state_file,state)
            start=time.time()
            with (dest/(label+'.log')).open('w',encoding='utf-8') as log:
                process=subprocess.Popen([str(x) for x in cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=process.pid;dump(state_file,state);code=process.wait()
            state.setdefault('seconds',{})[label]=time.time()-start
            assert code==0,f'{label} failed with code {code}'
        run('features',[colmap,'feature_extractor','--database_path',db,'--image_path',images,
            '--ImageReader.mask_path',masks,'--ImageReader.camera_model','PINHOLE',
            '--ImageReader.single_camera_per_image','1','--SiftExtraction.use_gpu','0',
            '--SiftExtraction.num_threads','2','--SiftExtraction.max_num_features','4096'])
        sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
        import read_write_model as rw
        cameras={};image_models={};feature_counts={}
        metadata={r['camera']:r for r in rows}
        with sqlite3.connect(db) as conn:
            dbimages=conn.execute('SELECT image_id,name,camera_id FROM images ORDER BY name').fetchall()
            assert len(dbimages)==16
            for image_id,name,camera_id in dbimages:
                cam=Path(name).stem;c=metadata[cam];K=np.array(c['K']);V=np.array(c['w2c'])
                params=np.array([K[0,0],K[1,1],K[0,2],K[1,2]],dtype=np.float64)
                conn.execute('UPDATE cameras SET model=?,width=?,height=?,params=?,prior_focal_length=1 WHERE camera_id=?',
                    (1,c['width'],c['height'],params.tobytes(),camera_id))
                cameras[camera_id]=rw.Camera(id=camera_id,model='PINHOLE',width=c['width'],height=c['height'],params=params)
                image_models[image_id]=rw.Image(id=image_id,qvec=rw.rotmat2qvec(V[:3,:3]),tvec=V[:3,3],
                    camera_id=camera_id,name=name,xys=np.empty((0,2)),point3D_ids=np.empty((0,),dtype=np.int64))
                num,cols,blob=conn.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?',(image_id,)).fetchone()
                xy=np.frombuffer(blob,dtype=np.float32).reshape(num,cols)[:,:2]
                fm=np.asarray(Image.open(masks/(name+'.png')).convert('L'))
                ix=np.clip(xy[:,0].astype(int),0,c['width']-1);iy=np.clip(xy[:,1].astype(int),0,c['height']-1)
                assert (fm[iy,ix]>0).all(),'Background features escaped mask'
                feature_counts[cam]=int(num)
            conn.commit()
        rw.write_model(cameras,image_models,{},str(empty),ext='.bin')
        state['feature_counts']=feature_counts;dump(state_file,state)
        run('matching',[colmap,'exhaustive_matcher','--database_path',db,'--SiftMatching.use_gpu','0',
            '--SiftMatching.num_threads','2','--SiftMatching.guided_matching','1'])
        run('triangulation',[colmap,'point_triangulator','--database_path',db,'--image_path',images,
            '--input_path',empty,'--output_path',sparse,'--refine_intrinsics','0','--Mapper.fix_existing_images','1',
            '--Mapper.num_threads','2','--Mapper.ba_use_gpu','0','--Mapper.filter_max_reproj_error','2',
            '--Mapper.filter_min_tri_angle','2','--Mapper.tri_ignore_two_view_tracks','1'])
        cc,ii,pp=rw.read_model(str(sparse),ext='.bin')
        assert set(ii)==set(image_models)
        for key,im in ii.items():
            original=image_models[key]
            assert np.allclose(im.qvec2rotmat(),original.qvec2rotmat(),atol=1e-7)
            assert np.allclose(im.tvec,original.tvec,atol=1e-7)
            assert np.allclose(cc[im.camera_id].params,cameras[im.camera_id].params,atol=1e-7)
        errors=[p.error for p in pp.values()]
        state.update(status='sparse_initialized_pending_quality_review',points=len(pp),fixed_poses_verified=True,
            feature_counts=feature_counts,point_track_median=float(np.median([len(set(p.image_ids)) for p in pp.values()])) if pp else None,
            reprojection_median_px=float(np.median(errors)) if errors else None,
            reprojection_p90_px=float(np.quantile(errors,.9)) if errors else None,
            training_started=False)
        dump(state_file,state)
        print(json.dumps({k:state[k] for k in ['status','points','feature_counts','point_track_median','reprojection_median_px','reprojection_p90_px']},indent=2),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));dump(state_file,state);raise


if __name__=='__main__':main()
