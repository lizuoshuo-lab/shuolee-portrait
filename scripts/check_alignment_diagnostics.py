"""Region and frozen-observation checks on synthetic data, never real subjects."""
import argparse
import copy
from contextlib import nullcontext,redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import pycolmap as pc

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt, video_sfm, cli
from shuolee_portrait.alignment_diagnostics import annotate_tracks, measure_regions, verify_camera_identity, diagnose, write_reviews, validate_settings
from check_sfm_holdout import fixture as native_fixture


def expect_error(call):
    try:call()
    except (ValueError,RuntimeError,FileExistsError) as exc:return str(exc)
    raise AssertionError('Invalid diagnostic input was accepted.')


def fixture(root, moving=False):
    root.mkdir();rows=[]
    K=np.array([[300.,0,240.5],[0,300.,180.5],[0,0,1.]])
    locations=[(220,90),(270,110),(215,230),(280,280),(60,180),(410,230),
               (140,180),(180,90),(365,225)]
    tracks=[dict(track_id=f'track_{i}',identity_basis='material_candidate',region='unknown',
                 triangulation_frames=['frame_000000','frame_000004'],observations=[]) for i in range(len(locations))]
    for i,center in enumerate([-.16,-.08,0.,.08,.16]):
        frame=f'frame_{i:06d}';dx=-300*center/3
        E=np.eye(4);E[0,3]=-center
        pixels=np.zeros((360,480,3),np.uint8);pixels[:]=[75,90,105]
        mask=np.zeros((360,480),np.uint8);mask[50:330,round(140+dx):round(340+dx)]=255
        pixels[mask>0]=[145,80,80]
        image=root/f'{frame}.png';mask_path=root/f'{frame}_mask.png'
        Image.fromarray(pixels).save(image);Image.fromarray(mask).save(mask_path)
        row=dict(frame=frame,name=image.name,source_frame_index=i,time_seconds=.1*i,
                 image=str(image),source_sha256=rt.sha(image),wh=[480,360],K=K.tolist(),w2c=E.tolist(),
                 mask=str(mask_path),mask_sha256=rt.sha(mask_path),face_box_xyxy=[180+dx,60,300+dx,140],
                 other_person_boxes_xyxy=[[350+dx,190,380+dx,280]])
        rows.append(row)
        for n,(u,v) in enumerate(locations):
            uv=[u+dx+(4 if moving and i==2 and n<4 else 0),v]
            tracks[n]['observations'].append(dict(frame=frame,observation_id=f'obs_{n}_{i}',
                image_sha256=row['source_sha256'],uv=uv))
    return rows,dict(schema='shuolee-fixed-tracks-v1',pixel_domain='undistorted_integer_centers',
                     camera_fit_observation_ids=None,tracks=tracks)


def strip_annotations(bank):
    data=copy.deepcopy(bank)
    for t in data['tracks']:
        t.pop('region_annotation',None);t['region']='unknown'
        for o in t['observations']:o.pop('region_annotation',None)
    return data


def run(evidence_dir):
    cases={}
    with tempfile.TemporaryDirectory(prefix='shuolee-region-diagnostic-') as tmp:
        base=Path(tmp);rows,bank=fixture(base/'static')
        annotated=annotate_tracks(rows,bank)
        assert strip_annotations(annotated)==bank
        summary,measurement=measure_regions(rows,annotated)
        expected=dict(face=2,body=2,static_scene_candidate=2,unknown=3)
        assert {k:v['tracks'] for k,v in summary['regions'].items()}==expected
        assert summary['total_tracks']==9 and summary['measured_tracks']==9
        assert summary['regions']['unknown']['held_error_px']['count']==9
        assert all(v['frames_with_observations']==5 for v in summary['regions'].values())
        cases['region_labels_and_full_membership'] = dict(expected_tracks=expected,
            original_membership_pixels_fit_views_unchanged=True,
            unknown_tracks_retained=summary['regions']['unknown']['tracks'])

        missing=copy.deepcopy(rows);missing[2]['face_box_xyxy']=None
        uncertain=annotate_tracks(missing,bank);uncertain_summary,_=measure_regions(missing,uncertain)
        assert uncertain_summary['region_disagreement_tracks']>=4
        assert uncertain_summary['total_tracks']==len(bank['tracks'])
        assert uncertain_summary['regions']['face']['held_error_px']['median'] is None
        cases['missing_face_region_is_unknown_not_zero_error']=dict(
            disagreement_tracks=uncertain_summary['region_disagreement_tracks'],
            face_error=uncertain_summary['regions']['face']['held_error_px'])

        moving_rows,moving_bank=fixture(base/'moving',moving=True)
        moving=annotate_tracks(moving_rows,moving_bank);motion_report,motion_measure=measure_regions(moving_rows,moving)
        face=motion_report['regions']['face']['frames'][2]['held_error_px']['median']
        body=motion_report['regions']['body']['frames'][2]['held_error_px']['median']
        background=motion_report['regions']['static_scene_candidate']['frames'][2]['held_error_px']['median']
        assert abs(face-4)<1e-7 and abs(body-4)<1e-7 and background<1e-7
        assert not motion_report['root_cause_resolved'] and not motion_report['subject_staticity_verified']
        cases['known_foreground_motion_preserved_per_frame']=dict(face_px=face,body_px=body,background_px=background,
            actual_motion_in_fixture_only=True,real_subject_staticity_verified=False)

        degenerate=copy.deepcopy(rows)
        for r in degenerate:r['w2c']=np.eye(4).tolist()
        report,_=measure_regions(degenerate,annotated)
        assert report['state']=='no_observable_reserved_tracks' and report['measured_tracks']==0
        assert report['regions']['face']['held_error_px']['median'] is None
        assert report['regions']['face']['unobservable_reasons']['coincident_camera_centers']==2
        cases['unobservable_points_retained']=dict(total=report['total_tracks'],measured=0,null_not_zero=True)

        altered=copy.deepcopy(rows);altered[0]['K'][0][0]+=1.
        errors=[expect_error(lambda:verify_camera_identity(rows,altered)),
                expect_error(lambda:verify_camera_identity(rows,list(reversed(rows))))]
        bad=copy.deepcopy(rows);bad[0]['face_box_xyxy']=[float('nan'),0,4,4]
        errors.append(expect_error(lambda:annotate_tracks(bad,bank)))
        mask_path=Path(rows[0]['mask']);original=mask_path.read_bytes()
        mask=np.array(Image.open(mask_path));mask[80,220]=128;Image.fromarray(mask).save(mask_path)
        bad=copy.deepcopy(rows);bad[0]['mask_sha256']=rt.sha(mask_path)
        errors.append(expect_error(lambda:annotate_tracks(bad,bank)))
        mask_path.write_bytes(original)
        cases['camera_order_roi_and_mask_contracts']=errors

        reviews=base/'unwarped-reviews';write_reviews(reviews,moving_rows,moving,motion_measure,validate_settings())
        review_index=rt.read(reviews/'index.json')['reviews']
        for r in review_index:
            source=next(x for x in moving_rows if x['frame']==r['frame'])
            expected=np.array(Image.open(source['image']).convert('RGB').crop(tuple(r['crop_xyxy'])))
            assert np.array_equal(expected,np.array(Image.open(reviews/r['original'])))
            assert not r['text_burned_into_images'] and not r['image_warped']
        if evidence_dir:
            if evidence_dir.exists():raise FileExistsError(evidence_dir)
            shutil.copytree(reviews,evidence_dir)
        partial=copy.deepcopy(motion_measure);partial['tracks'][0]['status']='not_observable'
        partial['tracks'][0]['reason']='non_positive_depth'
        invalid_reviews=base/'invalid-point-reviews'
        write_reviews(invalid_reviews,moving_rows,moving,partial,validate_settings())
        for card in rt.read(invalid_reviews/'index.json')['reviews']:
            for shown in card['displayed_tracks']:
                if shown['track_id']=='track_0':assert not shown['held_prediction_visible']
        cases['actual_png_reviews_keep_original_pixels_and_crop']=dict(pairs=len(review_index),
            text_burned_in=False,image_warped=False,unobservable_predictions_not_drawn_as_measurements=True)

        native=base/'native';source,decoded,_=native_fixture(native)
        config=rt.read(PROJECT/'configs/server.json');config['sfm_holdout']=dict(temporal_hops=[1,2],alias_radius_px=1.)
        rt.save(native/'config.json',config)
        with patch.object(pc,'extract_features',side_effect=lambda p,*a,**k:shutil.copy2(source,p)):
            video_sfm.sfm(native,config)
        training=rt.read(native/'data/sfm-inputs.json')
        for r in training['records']:
            w,h=r['wh'];mask=np.zeros((h,w),np.uint8);mask[45:h-20,180:450]=255
            path=native/'data'/f'{r["frame"]}_mask.png';Image.fromarray(mask).save(path)
            r.update(mask=str(path),mask_sha256=rt.sha(path),face_box_xyxy=[230,85,400,200],other_person_boxes_xyxy=[])
        rt.save(native/'data/training-inputs.json',training)
        before={p.relative_to(native).as_posix():rt.sha(p) for p in native.rglob('*') if p.is_file()}
        external=base/'actual-standalone'
        result=subprocess.run([sys.executable,str(PROJECT/'scripts/diagnose_alignment.py'),
            '--run',str(native),'--output',str(external)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert before=={p.relative_to(native).as_posix():rt.sha(p) for p in native.rglob('*') if p.is_file()}
        report=rt.read(external/'summary.json')
        assert report['measured_tracks']==report['total_tracks']==71
        support=rt.read(external/'scene-support.json')
        assert support['original_track_count']==71 and support['source_tracks_dropped']==0
        assert support['state']=='no_observable_scene_candidates'
        assert len(support['frames_without_scene_observations'])==len(training['records'])
        assert not support['physical_calibration_verified'] and not support['training_admitted']
        material=rt.read(external/'material-review/manifest.json')
        assert material['total_tracks']==71 and material['total_observations']==355
        assert not material['physical_material_identity_verified']
        assert (external/'material-review/index.html').is_file()
        cases['actual_native_sfm_to_region_cli_readonly']=dict(
            measured_tracks=report['measured_tracks'],by_region={k:v['tracks'] for k,v in report['regions'].items()},
            source_run_unchanged=True,training_admitted=report['training_admitted'],
            scene_support_state=support['state'],portable_material_tracks=material['total_tracks'],
            portable_material_observations=material['total_observations'])

        original=(native/'data/sfm-holdout-tracks.json').read_bytes()
        changed=json.loads(original);changed['tracks'].pop();rt.save(native/'data/sfm-holdout-tracks.json',changed)
        error=expect_error(lambda:diagnose(native,base/'silently-dropped'))
        assert 'dropped' in error
        (native/'data/sfm-holdout-tracks.json').write_bytes(original)
        expect_error(lambda:diagnose(native,external))
        cases['missing_reserved_track_and_overwrite_rejected']=True

        config['alignment_diagnostics']=dict(min_parallax_degrees=89.)
        rt.save(native/'config.json',config)
        process=subprocess.run([sys.executable,str(PROJECT/'run.py'),'_stage','--stage','alignment-diagnostics',
            '--output',str(native),'--assets',str(base/'missing-assets')],capture_output=True,text=True)
        assert process.returncode==3,process.stderr
        no_support=rt.read(native/'data/alignment-diagnostics/summary.json')
        assert no_support['measured_tracks']==0 and no_support['total_tracks']==71
        assert not (native/'fit').exists() and not (native/'gaussian-run').exists()
        cases['actual_stage_stops_before_fitting_without_observable_support']=dict(exit_code=process.returncode,
            source_tracks_retained=71,diagnostic_limit_deliberately_degenerate_for_test=True)
        parent=base/'parent';video=base/'fixture.mp4';video.write_bytes(b'preprocessing is simulated in this test')
        real_run=subprocess.run;calls=[]
        def dispatch(command,*args,**kwargs):
            name=command[command.index('--stage')+1];calls.append(name)
            if name=='alignment-diagnostics':return real_run(command,*args,**kwargs)
            if name not in ('extract','sfm','observations','material-tracks'):
                raise AssertionError('Parent continued to body fitting after an unobservable diagnostic.')
            if name=='observations':
                for folder in ('raw','sfm','rectified','data'):
                    shutil.copytree(native/folder,parent/folder,dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('alignment-diagnostics'))
            return subprocess.CompletedProcess(command,0)
        argv=['run.py','reconstruct','--video',str(video),'--output',str(parent),
              '--config',str(native/'config.json'),'--assets',str(base/'missing-assets')]
        with patch.object(sys,'argv',argv),patch.object(cli,'doctor',return_value={}),patch.object(cli,'gpu_lock',return_value=nullcontext()),patch.object(cli.shutil,'which',return_value='fixture'),patch.object(cli.subprocess,'run',side_effect=dispatch),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            try:cli.main()
            except SystemExit as exc:assert exc.code==3
            else:raise AssertionError('Parent should record the failed diagnostic.')
        parent_status=rt.read(parent/'status.json')
        assert parent_status['state']=='alignment_not_admitted' and parent_status['training_started'] is False
        assert parent_status['alignment_report']=='data/alignment-diagnostics/summary.json'
        assert calls==['extract','sfm','observations','material-tracks','alignment-diagnostics']
        cases['parent_status_records_actual_diagnostic_failure']=dict(
            previous_stages_simulated=True,diagnostic_stage_actually_executed=True,stopped_before_prior=True)
    return dict(passed=True,groups=len(cases),cases=cases,synthetic_only=True,
        real_person_tested=False,physical_camera_validated=False,ghosting_resolved=False,
        training_started=False,old_diagnosis_reopened=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--evidence-dir',type=Path)
    args=parser.parse_args();report=run(args.evidence_dir);rt.save(args.report,report)
    print(json.dumps(report,indent=2))
