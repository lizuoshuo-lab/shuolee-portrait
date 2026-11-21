"""Contract and call-boundary checks; fixtures never authorize real training."""
from pathlib import Path
from contextlib import ExitStack, nullcontext
from unittest.mock import patch
from types import ModuleType
import copy, json, hashlib
import numpy as np
from PIL import Image
import local_surface123_alignment_gate as gate
import local_surface123 as driver

REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'alignment-admission-engineering'


def save(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def rebind(root):
    receipt=gate.read(root/'reviews/alignment-admission.json')
    receipt['artifacts']={name:gate.sha(root/name) for name in gate.ARTIFACTS}
    review=gate.read(root/'reviews/review.json')
    review['artifacts']=receipt['artifacts']
    for key in ['roster','measurements']:
        review[key+'_sha256']=gate.sha(root/f'reviews/{key}.json')
    save(root/'reviews/review.json',review)
    for key in ['full_body','background']:
        r=gate.read(root/f'reviews/{key}.json');r['artifacts']=receipt['artifacts'];save(root/f'reviews/{key}.json',r)
    receipt['evidence']={key:dict(path=f'reviews/{key}.json',sha256=gate.sha(root/f'reviews/{key}.json'))
        for key in ['roster','measurements','review','full_body','background']}
    save(root/'reviews/alignment-admission.json',receipt)


def fixture(label):
    root=OUT/'fixtures-v4'/label;root.mkdir(parents=True,exist_ok=False)
    for path in ['fit','camera','data','raw','reviews','clothing-init','snapshot']:(root/path).mkdir()
    K=np.array([[100.,0,32.],[0,110.,32.],[0,0,1.]])
    KR=K.copy();KR[:2,2]+=.5
    E=np.repeat(np.eye(4)[None],3,axis=0);E[:,0,3]=[-.2,0,.2]
    np.savez(root/'camera/solver-output.npz',K=K,w2c=E,track_ids=np.array(['scene_fixture_001']))
    np.savez(root/'fit/fitted.npz',fixture=np.zeros((3,3)))
    np.savez(root/'clothing-init/points.npz',fixture=np.zeros((3,3)))
    (root/'snapshot/world.ply').write_text('synthetic call-order fixture; not a model',encoding='utf-8')
    save(root/'config.json',dict(synthetic=True))
    save(root/'fit/geometry-gate.json',dict(coarse_initialization_gate_passed=True,quality_accepted=False))
    rows=[];decoded=[];frames=[]
    for i in range(3):
        name=f'frame_{i+1:06d}';frames.append(name)
        image=root/'raw'/f'{name}.png';mask=root/'data'/f'{name}_mask.png'
        Image.new('RGB',(64,64),(128,128,128)).save(image);Image.new('L',(64,64),255).save(mask)
        dec=dict(frame=name,source_frame_index=i,source_sha256=gate.sha(image),wh=[64,64]);decoded.append(dec)
        rows.append(dict(**dec,image=str(image),mask=str(mask),mask_sha256=gate.sha(mask),
            K=KR.tolist(),K_integer_centers=K.tolist(),w2c=E[i].tolist(),camera_registration=dict(interpolation_used=False)))
    save(root/'decoded.json',dict(records=decoded));save(root/'data/training-inputs.json',dict(records=rows))
    families=[];points=[];checks=[]
    for part,xyz in [('eyes',[-.3,.2,4]),('nose',[0,0,3.8]),('mouth',[.25,-.2,4.2])]:
        family='synthetic_'+part
        families.append(dict(id=family,part=part,source_track_ids=[],alias_family_ids=[],
            physical_identity_reviewed=True,static_material_reviewed=True,used_for_camera_fit=False))
        points.append(dict(family_id=family,xyz_world=xyz,estimation='independent_physical_measurement',anchor_observation_ids=[]))
        for i in range(3):
            cam=E[i,:3,:3]@xyz+E[i,:3,3];uv=K@cam;uv=uv[:2]/uv[2]
            checks.append(dict(id=f'{family}_{i}',family_id=family,frame=frames[i],xy=(uv+[.2,0]).tolist(),
                localization_uncertainty_px=.05,uncertainty_method='controlled_physical_target'))
    roster=dict(pixel_domain='native_video_integer_centers',camera_model='rectified_pinhole',
        frozen_before_projection_evaluation=True,no_error_based_exclusions=True,all_registered_frame_ids=frames,
        families=families,checks=checks)
    measurements=dict(points=points,evaluated_observation_ids=[c['id'] for c in checks],
        point_estimation_used_check_observations=False,camera_artifact_sha256=gate.sha(root/'camera/solver-output.npz'),
        evaluation_started='2026-01-02T00:00:00+00:00')
    save(root/'reviews/roster.json',roster);save(root/'reviews/measurements.json',measurements)
    save(root/'alignment-preregistration.json',dict(subject=root.name,created='2026-01-01T00:00:00+00:00',
        frozen_before_error_evaluation=True,roster_sha256=gate.sha(root/'reviews/roster.json'),
        camera_sha256=gate.sha(root/'camera/solver-output.npz'),training_inputs_sha256=gate.sha(root/'data/training-inputs.json'),
        all_registered_frame_ids=frames))
    review=dict(actual_visual_review=True,identity_checked_before_errors=True,held_family_independence_reviewed=True,
        uncertainty_adequate_for_subpixel_claim=True,all_training_frames_registered_reviewed=True,decision='approved',
        camera_fit_family_aliases=[],face_coverage=dict(actual_review=True,adequate=True,observed_frame_ids=frames,
        unobserved_frame_ids=[],view_groups=dict(left=[frames[0]],front=[frames[1]],right=[frames[2]])))
    save(root/'reviews/review.json',review)
    for scope in ['full_body','background']:
        save(root/f'reviews/{scope}.json',dict(actual_visual_review=True,passed=True,independent_validation=True,
            no_error_based_exclusions=True,protocol_and_limits='synthetic_fixture_only',observations_or_report='Known synthetic geometry; not a realdata report',
            reviewed_frame_ids=frames,assessed_regions=['torso','hands','feet','occlusions','eyes','nose','mouth'],current_smplx_overlay_reviewed=True))
    save(root/'reviews/alignment-admission.json',dict(schema=gate.SCHEMA,subject=root.name,purpose='synthetic_contract_test'))
    rebind(root)
    return root


def edit(root,name,fn):
    p=root/f'reviews/{name}.json';j=gate.read(p);fn(j);save(p,j)
    # Each synthetic mutation constructs its test case before evaluation.
    # The dedicated stale-roster case below intentionally keeps the original hash.
    if name=='roster':
        p=root/'alignment-preregistration.json';j=gate.read(p);j['roster_sha256']=gate.sha(root/'reviews/roster.json');save(p,j)
    rebind(root)


def main():
    need=OUT/'test-result-v4.json'
    assert not need.exists(),'Tests already recorded; only rerun after a justified code change.'
    cases=[]
    def test(name,change,expected):
        root=fixture(name);change(root)
        result=gate.evaluate(root,allow_synthetic_contract_test=True)
        assert not result['training_allowed'],name
        assert (result['contract_passed'] if expected=='contract_passed' else
                any(expected in r for r in result['reasons'])),(name,result)
        cases.append(dict(name=name,expected=expected,result=result))
    test('valid_synthetic',lambda r:None,'contract_passed')
    positive=OUT/'fixtures-v4/valid_synthetic'
    prod=gate.evaluate(positive)
    assert not prod['training_allowed'] and 'synthetic_evidence_not_training_admission' in prod['reasons']
    cases.append(dict(name='synthetic_rejected_by_training_entry',result=prod))
    def stale_roster(r):
        p=r/'reviews/roster.json';j=gate.read(p);j['checks'].pop();save(p,j);rebind(r)
    test('roster_changed_after_registration',stale_roster,'preregistered_roster_changed')
    test('late_preregistration',lambda r:edit(r,'measurements',lambda j:j.update(evaluation_started='2025-12-31T00:00:00+00:00')),'preregistration_not_before_evaluation')
    test('one_error_exactly_one',lambda r:edit(r,'roster',lambda j:j['checks'][0]['xy'].__setitem__(0,j['checks'][0]['xy'][0]+.8)), 'face_observation_at_or_above_1px')
    test('omitted_held',lambda r:edit(r,'measurements',lambda j:j['evaluated_observation_ids'].pop()),'omitted_or_duplicate_held_observations')
    test('duplicate_held',lambda r:edit(r,'roster',lambda j:j['checks'].append(copy.deepcopy(j['checks'][0]))),'held_observation_roster_duplicate')
    test('used_held_in_point_fit',lambda r:edit(r,'measurements',lambda j:j.update(point_estimation_used_check_observations=True)),'held_used_for_point_estimation')
    test('anchor_overlap',lambda r:edit(r,'measurements',lambda j:j['points'][0].update(anchor_observation_ids=['synthetic_eyes_0'])),'check_observation_is_anchor')
    test('fit_family_alias',lambda r:edit(r,'roster',lambda j:j['families'][0].update(alias_family_ids=['scene_fixture_001'])),'fit_held_family_leakage')
    test('material_not_reviewed',lambda r:edit(r,'roster',lambda j:j['families'][0].update(physical_identity_reviewed=False)),'unverified_material')
    test('hessian_is_not_truth',lambda r:edit(r,'roster',lambda j:j['checks'][0].update(uncertainty_method='photometric_hessian')),'uncertainty_not_independent_physical_evidence')
    test('nan_uncertainty',lambda r:edit(r,'roster',lambda j:j['checks'][0].update(localization_uncertainty_px=float('nan'))),'unknown_localization_uncertainty')
    test('nonpositive_depth',lambda r:edit(r,'measurements',lambda j:j['points'][0].update(xyz_world=[0,0,-1])),'nonpositive_camera_depth')
    test('face_part_missing',lambda r:edit(r,'roster',lambda j:j['families'][0].update(part='mouth')),'eyes_nose_mouth_coverage_incomplete')
    test('coverage_not_reviewed',lambda r:edit(r,'review',lambda j:j['face_coverage'].update(adequate=False)),'face_view_coverage_not_accepted')
    test('background_protection_failed',lambda r:edit(r,'background',lambda j:j.update(passed=False)),'protection_failed:background')
    test('feet_not_assessed',lambda r:edit(r,'full_body',lambda j:j.update(assessed_regions=['torso','hands'])),'fullbody_region_coverage_incomplete')
    test('smplx_overlay_missing',lambda r:edit(r,'full_body',lambda j:j.update(current_smplx_overlay_reviewed=False)),'current_smplx_overlay_not_reviewed')
    test('stale_fit',lambda r:(r/'fit/fitted.npz').write_bytes(b'changed'), 'artifact_hash_mismatch:fit/fitted.npz')
    test('stale_camera',lambda r:(r/'camera/solver-output.npz').write_bytes(b'changed'), 'artifact_hash_mismatch:camera/solver-output.npz')
    test('changed_native_image',lambda r:(r/'raw/frame_000001.png').write_bytes(b'changed'), 'input_file_hash_mismatch')
    def rows_change(r,fn):
        p=r/'data/training-inputs.json';j=gate.read(p);fn(j['records']);save(p,j)
        p=r/'alignment-preregistration.json';j=gate.read(p);j['training_inputs_sha256']=gate.sha(r/'data/training-inputs.json');save(p,j);rebind(r)
    test('frame_order_changed',lambda r:rows_change(r,lambda rows:rows.reverse()),'training_frame_identity_or_order_mismatch')
    test('interpolated_camera',lambda r:rows_change(r,lambda rows:rows[0]['camera_registration'].update(interpolation_used=True)),'interpolated_or_unverified_camera')
    test('half_pixel_mismatch',lambda r:rows_change(r,lambda rows:rows[0]['K'][0].__setitem__(2,32.)), 'pixel_center_contract_mismatch')
    real=[]
    runbase=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
    for subject in ['grayT','whiteT','winterhat']:
        root=runbase/subject;before={n:gate.sha(root/n) for n in gate.ARTIFACTS if (root/n).is_file()}
        result=gate.evaluate(root);camera_result=gate.evaluate(root,phase='camera')
        assert not result['training_allowed'] and result['reasons']==['fine_alignment_receipt_missing']
        assert not camera_result['training_allowed'] and camera_result['reasons']==['camera_alignment_receipt_missing']
        assert all(gate.sha(root/n)==h for n,h in before.items())
        real.append(dict(subject=subject,result=result,camera_result=camera_result,current_artifacts_unchanged=True,
            historical_coarse_gate=gate.read(root/'fit/geometry-gate.json')))

    camera_root=fixture('camera_positive')
    receipt=gate.read(camera_root/'reviews/alignment-admission.json')
    receipt['schema']=gate.CAMERA_SCHEMA;receipt['artifacts'].pop('fit/fitted.npz')
    for name in ['review','full_body','background']:
        p=camera_root/f'reviews/{name}.json';j=gate.read(p);j['artifacts']=receipt['artifacts'];save(p,j)
    for name,link in receipt['evidence'].items():link['sha256']=gate.sha(camera_root/link['path'])
    save(camera_root/'reviews/camera-alignment-admission.json',receipt)
    cr=gate.evaluate(camera_root,phase='camera',allow_synthetic_contract_test=True)
    assert cr['contract_passed'] and not cr['training_allowed'] and not cr['fitting_allowed'],cr
    cases.append(dict(name='camera_contract_synthetic',result=cr))
    cr=gate.evaluate(camera_root,phase='camera')
    assert not cr['training_allowed'] and 'synthetic_evidence_not_training_admission' in cr['reasons']
    cases.append(dict(name='camera_synthetic_rejected_by_real_entry',result=cr))

    integration=[]
    for fail_call in [1,2,3,4]:
        root=fixture('entry_boundary_'+str(fail_call));events=[];counter=[0]
        runtime=ModuleType('shuolee_portrait.runtime');runtime.configure=lambda *a:events.append('configure')
        parent=ModuleType('shuolee_portrait');parent.runtime=runtime
        surf=ModuleType('shuolee_portrait.clothed_surface');surf.initialize=lambda *a:events.append('surface')
        train=ModuleType('shuolee_portrait.clothed_training')
        def fake_train(*args,**kw):
            events.append('synthetic_training_callback_only')
            assert kw['checkpoint_review'](500,root/'snapshot') is True
            kw['checkpoint_review'](1000,root/'snapshot')
            raise AssertionError('1000step guard should stop the simulated callback')
        train.train_clothed=fake_train
        def boundary(_):
            counter[0]+=1;events.append('fine_guard_'+str(counter[0]))
            if counter[0]==fail_call:raise RuntimeError('synthetic_denial')
        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules',{'shuolee_portrait':parent,'shuolee_portrait.runtime':runtime,
                'shuolee_portrait.clothed_surface':surf,'shuolee_portrait.clothed_training':train}))
            for key,val in dict(prepare=lambda s:root,stage=lambda r,n:events.append('stage_'+n),
                await_review=lambda *a:True,require_alignment_admission=boundary,require_camera_admission=boundary,gpu_slot=nullcontext,
                status=lambda *a,**k:None,read=lambda p:{'training_started':False},save=lambda *a:None).items():
                stack.enter_context(patch.object(driver,key,val))
            try:driver.run_subject({'id':root.name})
            except RuntimeError as exc:assert str(exc)=='synthetic_denial'
            else:raise AssertionError('Denial did not stop the driver')
        assert counter[0]==fail_call
        assert ('stage_fit' in events)==(fail_call>=2)
        assert ('surface' in events)==(fail_call>=3)
        assert ('synthetic_training_callback_only' in events)==(fail_call==4)
        integration.append(dict(denied_at_guard_call=fail_call,events=events,no_real_GPU_or_training=True))
    result=dict(contract_cases=len(cases),all_contract_cases_passed=True,real_artifact_rejections=real,
        entry_boundary_tests=integration,cases=cases,training_launched=False,physical_alignment_fixed=False,
        guard_source_sha256=gate.sha(Path(gate.__file__)),orchestrator_source_sha256=gate.sha(Path(driver.__file__)))
    save(need,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','real_artifact_rejections']},ensure_ascii=False))


if __name__=='__main__':main()
