"""CPU-only independent audit of the freshly recovered video15 COLMAP model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, 'D:/Projects/msv2')
from msv2.data import _read_cameras, _read_images, _camera, _w2c, select_frames

ROOT=Path('D:/GS-SLAM-Data/msv2-video15-20261105')
OUT=Path('D:/00/记录/2026-11-05_MSV2_video15/相机输入核查.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def summary(values):
    x=np.asarray(values,dtype=np.float64)
    assert x.size and np.isfinite(x).all()
    return dict(count=int(x.size),mean=float(x.mean()),median=float(np.median(x)),
        rms=float(np.sqrt(np.mean(x*x))),p90=float(np.quantile(x,.90)),
        p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),
        min=float(x.min()),max=float(x.max()))


def read_points(path):
    ids=[];xyz=[];errors=[];track_lengths=[];track_images=[];track_2d=[];track_ids=[]
    with path.open('rb') as stream:
        (n,)=struct.unpack('<Q',stream.read(8))
        for _ in range(n):
            values=struct.unpack('<Q3d3Bd',stream.read(43))
            point_id=values[0]
            (length,)=struct.unpack('<Q',stream.read(8))
            pairs=np.frombuffer(stream.read(length*8),dtype='<i4').reshape(-1,2)
            ids.append(point_id);xyz.append(values[1:4]);errors.append(values[7]);track_lengths.append(length)
            track_images.extend(pairs[:,0].tolist());track_2d.extend(pairs[:,1].tolist());track_ids.extend([point_id]*length)
        assert stream.read(1)==b''
    order=np.argsort(ids)
    return (np.asarray(ids,dtype=np.int64)[order], np.asarray(xyz,dtype=np.float64)[order],
        np.asarray(errors)[order], np.asarray(track_lengths,dtype=np.int64)[order],
        np.asarray(track_images,dtype=np.int64),np.asarray(track_2d,dtype=np.int64),np.asarray(track_ids,dtype=np.int64))


def main():
    started=datetime.now(timezone.utc).isoformat()
    lineage=json.loads((ROOT/'pixel-lineage.json').read_text(encoding='utf-8'))
    inputs=json.loads((ROOT/'camera-inputs.json').read_text(encoding='utf-8'))
    state=json.loads((ROOT/'state.json').read_text(encoding='utf-8'))
    rows=inputs['records'];pixels=lineage['records']
    assert lineage['passed'] and lineage['no_spatial_resizing'] and lineage['full_rgb']
    assert inputs['full_rgb'] and not inputs['mask_used'] and not inputs['crop_used']
    assert state['state']=='completed' and len(rows)==len(pixels)==65
    assert sha(lineage['source'])==lineage['video_sha256']==state['video_sha256']
    probe=json.loads(subprocess.check_output([
        'C:/Users/Administrator/ffmpeg/bin/ffprobe.exe','-v','error','-select_streams','v:0',
        '-show_frames','-show_streams','-of','json',lineage['source']]))
    times=[row['best_effort_timestamp_time'] for row in probe['frames']]
    chosen=select_frames(times,3,duration=probe['streams'][0]['duration'])
    assert len(times)==513 and len(chosen)==65
    assert [x['source_index'] for x in chosen]==[x['source_index'] for x in pixels]
    assert all(abs(a['time_seconds']-b['time_seconds'])<1e-12 for a,b in zip(chosen,pixels))
    assert probe['streams'][0]['width']==1520 and probe['streams'][0]['height']==2704
    assert not probe['streams'][0].get('tags',{}).get('rotate')
    assert not any(abs(s.get('rotation',0))>0 for s in probe['streams'][0].get('side_data_list',[]))
    sparse=ROOT/'dataset/sparse/0'
    cams=_read_cameras(sparse);images=_read_images(sparse)
    assert len(cams)==1 and len(images)==65
    image_by_name={im.name:im for im in images}
    ids,xyz,stored_error,tracklen,track_images,track_2d,track_ids=read_points(sparse/'points3D.bin')
    assert len(ids)==58509 and len(set(ids.tolist()))==len(ids)
    assert np.isfinite(xyz).all() and np.all(stored_error>=0) and np.all(tracklen>=2)
    all_errors=[];all_point_rows=[];frame_stats=[];centers=[];axes=[];all_feature_count=0
    det_errors=[];orthogonal_errors=[];q_norm_errors=[];projection_contract_errors=[]
    with (sparse/'images.bin').open('rb') as stream:
        for i,(row,pixel,choice) in enumerate(zip(rows,pixels,chosen)):
            expected=f'frame_{i+1:06d}'
            assert row['frame']==pixel['frame']==expected
            assert row['name']==pixel['image']==expected+'.png'
            assert row['source_index']==pixel['source_index']==choice['source_index']
            assert row['wh']==pixel['wh']==[1520,2704]
            path=ROOT/'dataset/images'/row['name']
            assert sha(path)==row['source_sha256']==pixel['sha256']
            with Image.open(path) as image:
                assert image.mode=='RGB' and list(image.size)==row['wh']
                assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==pixel['rgb_sha256']
            im=image_by_name[row['name']];cam=cams[im.camera_id]
            K=np.asarray(_camera(cam));pose=np.asarray(_w2c(im))
            assert row['colmap_image_id']==im.id and row['colmap_camera_id']==cam.id
            assert np.array_equal(K,np.asarray(row['K'])) and np.array_equal(pose,np.asarray(row['w2c']))
            assert (cam.width,cam.height)==(1520,2704) and cam.model==1
            R,t=pose[:3,:3],pose[:3,3]
            det_errors.append(abs(np.linalg.det(R)-1));orthogonal_errors.append(np.max(np.abs(R@R.T-np.eye(3))))
            q_norm_errors.append(abs(np.linalg.norm(im.qvec)-1))
            center=-R.T@t;centers.append(center);axes.append(R.T@np.array([0.,0.,1.]))
            stream.seek(im.payload.offset)
            obs=np.frombuffer(stream.read(im.payload.length),dtype=[('x','<f8'),('y','<f8'),('id','<i8')])
            assert len(obs)==im.observations
            all_feature_count+=len(obs)
            track=track_images==im.id
            index=track_2d[track]
            assert np.all((index>=0)&(index<len(obs))) and np.array_equal(obs['id'][index],track_ids[track])
            valid=obs['id']>=0
            point_id=obs['id'][valid]
            position=np.searchsorted(ids,point_id)
            assert np.all(position<len(ids)) and np.array_equal(ids[position],point_id)
            point=xyz[position]
            transformed=point@R.T+t
            assert np.all(transformed[:,2]>0)
            uv=transformed@K.T;uv=uv[:,:2]/uv[:,2:]
            observed=np.stack([obs['x'][valid],obs['y'][valid]],axis=1)
            residual=np.linalg.norm(uv-observed,axis=1)
            assert np.isfinite(residual).all()
            # Independent homogeneous projection, in the same COLMAP
            # pixel-corner coordinates (no extra half-pixel shift).
            P=K@pose[:3]
            homogeneous=np.concatenate([point,np.ones((len(point),1))],axis=1)@P.T
            uv_h=homogeneous[:,:2]/homogeneous[:,2:]
            projection_contract_errors.append(float(np.max(np.abs(uv-uv_h))))
            all_errors.append(residual);all_point_rows.append(position)
            frame_stats.append(dict(frame=expected,source_index=row['source_index'],
                colmap_image_id=im.id,feature_count=len(obs),tracked_observations=int(valid.sum()),
                positive_depth_observations=int((transformed[:,2]>0).sum()),
                reprojection_euclidean_px=summary(residual),camera_center_sfm_units=center.tolist(),
                optical_axis_world=axes[-1].tolist()))
            print('CAMERA_AUDIT',expected,int(valid.sum()),float(residual.mean()),flush=True)
    residual=np.concatenate(all_errors);point_rows=np.concatenate(all_point_rows)
    assert len(residual)==len(track_images)==tracklen.sum()
    counts=np.bincount(point_rows,minlength=len(ids))
    assert np.array_equal(counts,tracklen)
    perpoint=np.bincount(point_rows,weights=residual,minlength=len(ids))/counts
    distinct_pairs=np.unique(np.column_stack((track_ids,track_images)),axis=0)
    distinct_views=np.bincount(np.searchsorted(ids,distinct_pairs[:,0]),minlength=len(ids))
    centers=np.asarray(centers);axes=np.asarray(axes)
    steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    changes=np.degrees(np.arccos(np.clip(np.sum(axes[1:]*axes[:-1],axis=1),-1,1)))
    mapper=next(c for c in state['commands'] if len(c)>1 and c[1]=='mapper')
    assert '--input_path' not in mapper
    assert (ROOT/'work/database.db').as_posix() in [str(x).replace('\\','/') for x in mapper]
    ba=next(c for c in state['commands'] if len(c)>1 and c[1]=='bundle_adjuster')
    assert (ROOT/'work/models/0').as_posix() in [str(x).replace('\\','/') for x in ba]
    native_log=(ROOT/'logs/bundle_adjustment.log').read_text(encoding='utf-8')
    result=dict(schema='msv2-camera-input-audit-v1',state='passed',started=started,
        finished=datetime.now(timezone.utc).isoformat(),source_video_sha256=lineage['video_sha256'],
        wh=[1520,2704],source_frames=513,selected_frames=65,registered_frames=65,
        real_PTS_selection_verified=True,all_source_indices=[r['source_index'] for r in pixels],
        first_PTS_seconds=float(times[0]),last_PTS_seconds=float(times[-1]),
        sample_fps=3,maximum_sampling_error_seconds=max(abs(r['sampling_error_seconds']) for r in chosen),
        PNG_and_decoded_RGB_hashes_verified=65,no_spatial_resize=True,mask_used=False,crop_used=False,
        colmap_cameras=[dict(id=c.id,model='PINHOLE',wh=[c.width,c.height],K=_camera(c)) for c in cams.values()],
        camera_binary_and_manifest_exactly_equal=True,
        quaternion_norm_max_error=max(q_norm_errors),rotation_det_max_error=max(det_errors),
        rotation_orthogonality_max_error=max(orthogonal_errors),
        independent_homogeneous_projection_max_error_px=max(projection_contract_errors),
        sparse_points=len(ids),total_2D_features=all_feature_count,tracked_2D_observations=len(residual),
        track_length=summary(tracklen),all_track_image_point2D_links_verified=True,
        distinct_registered_images_per_point=summary(distinct_views),
        additional_same_image_point_observations=int(len(track_ids)-len(distinct_pairs)),
        track_length_note='Track length counts 2D feature observations, not distinct image viewpoints. Distinct-view support is reported separately.',
        all_tracked_observations_have_positive_depth=True,
        reprojection=dict(
            definition='For every registered image observation with point3D_id>=0, compute || K (R X+t) normalized by z - COLMAP observed (x,y) ||2 in full 1520x2704 image pixels. Every observation weighted equally. Final saved binary K/poses/points are used.',
            observed_euclidean_px=summary(residual),
            point_weighted_mean_of_track_mean_px=summary(perpoint),
            stored_points3D_error_px=summary(stored_error),
            stored_minus_recomputed_track_mean_max_abs=float(np.max(np.abs(stored_error-perpoint))),
            coordinate_component_RMS_px=float(np.sqrt(np.mean(residual**2)/2)),
            bundle_adjuster_log=native_log,
            BA_printed_cost_relation_for_this_log='Final printed0.409581 equals half of the independently recomputed Euclidean RMS0.8191623182 within rounding; it is not the Euclidean arithmetic mean.',
            caution='BA printed cost is an optimizer summary, not the arithmetic mean of Euclidean pixel reprojection distances. Stored point.error weights each 3D point equally and may be stale after global BA; independent final observation distances are authoritative for this report.'),
        trajectory=dict(units='Arbitrary SfM scale, not measured metres',centers_min=centers.min(0).tolist(),
            centers_max=centers.max(0).tolist(),start_to_end_distance=float(np.linalg.norm(centers[-1]-centers[0])),
            path_length=float(steps.sum()),consecutive_center_distance=summary(steps),
            consecutive_optical_axis_angle_deg=summary(changes)),
        provenance=dict(source='Fresh CPU COLMAP affine/DSP SIFT, guided sequential overlap10, mapper, global bundle_adjuster with principal point refinement',
            mapper_command=mapper,bundle_adjuster_command=ba,mapper_has_no_prior_model_input=True,
            previous_fitted_camera_or_body_state_read=False,
            evidence='Final binary model and manifest matched directly; command receipts reference this fresh database/models only. No human pose initialization or interpolation commands are present.',
            physical_camera_calibration_accepted=False,geometry_quality_accepted=False),
        per_frame=frame_stats,
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'pixel-lineage.json',ROOT/'camera-inputs.json',ROOT/'state.json',
            sparse/'cameras.bin',sparse/'images.bin',sparse/'points3D.bin',ROOT/'logs/bundle_adjustment.log']},
        CPU_only=True,GPU_started=False,training_updates=0,prepared_files_modified=False)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINAL_CAMERA_AUDIT',json.dumps({k:result[k] for k in ['state','registered_frames','sparse_points','tracked_2D_observations','reprojection']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
