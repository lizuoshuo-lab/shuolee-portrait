"""CPU-only source/measurement admission for the actual native training entry.

Hashes bind supplied evidence, not its physical truth. This consumer does not
estimate cameras, change pixels, discard failed observations, or certify an
unseen capture. Existing N3/N13 geometry remains the numerical authority.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, math
import numpy as np
from PIL import Image
from material_observations122 import Observation, camera_fit_permission
from material_geometry122 import checked_collection, checked_cameras, evaluate_material_geometry
from native_pixel_convention122 import to_native_pixels

SCHEMA='shuolee-native-training-input-v1'
FACE_REGIONS={'left_eye','right_eye','nose','mouth'}
BODY_REGIONS={'shoulder','sleeve','hem','leg','foot'}
SCENE_REGIONS={'near_head','near_body','far_scene'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def checked_asset(base,ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Asset requires path and SHA256')
    if not isinstance(ref['sha256'],str) or len(ref['sha256'])!=64 or any(x not in '0123456789abcdef' for x in ref['sha256']):
        raise ValueError('Invalid SHA256')
    p=Path(ref['path']);p=p if p.is_absolute() else base/p
    p=p.resolve(strict=True)
    if not p.is_file() or sha(p)!=ref['sha256']:raise ValueError('Missing or changed asset: '+str(p))
    return p

def observation(record):
    d=dict(record)
    if d.pop('schema',None)!='shuolee-material-observation-v1':raise ValueError('Use existing N3 observation schema')
    d['raw_uv']=tuple(d['raw_uv']);d['evidence_source']=tuple(d['evidence_source'])
    return Observation(**d)

def audit_input(path):
    """Return every discovered problem before any GPU or model import occurs.

    The frame plan and visibility/region plan are frozen separate assets. They
    cannot be inferred from whichever camera/image files happen to be present.
    Physical assertions need scoped, hash-bound evidence as well as independent
    target-excluded numeric verification; a bare `accepted:true` is ignored.
    """
    path=Path(path).resolve(strict=True);base=path.parent;manifest_sha=sha(path);m=read(path)
    issues=[];assets={};frame_rows=[];geometry=None;fit_fail=[];requirements=[]
    def refhash(name):return (m.get(name) or {}).get('sha256')
    def issue(kind,**detail):issues.append(dict(kind=kind,**detail))
    def asset(name,ref,json_data=True):
        try:
            p=checked_asset(base,ref);assets[str(p)]=sha(p)
            return read(p) if json_data else p
        except (OSError,ValueError,TypeError,KeyError) as e:issue('invalid_asset',asset=name,reason=str(e));return None
    if m.get('schema')!=SCHEMA:issue('unsupported_input_schema')
    if m.get('scope') not in {'person-reconstruction','synthetic-control-only'}:issue('explicit_scope_required')
    if m.get('input_domain')!='already_rectified_resized_cropped':issue('unsupported_input_domain_no_implicit_undistortion')
    if m.get('projection')!='pinhole' or m.get('radial_coefficients') is not None:issue('captured_export_requires_explicit_verified_output_projection')
    if m.get('pixel_convention')!='integer_centers':issue('native_integer_pixel_measurements_required')
    if m.get('world_frame')!='right_handed_world_y_up':issue('explicit_world_y_up_frame_required')
    if not isinstance(m.get('run_id'),str) or not m['run_id']:issue('fresh_run_identity_required')
    training=m.get('training',{})
    rates=training.get('learning_rates',{})
    if (type(training.get('steps'))is not int or not 1000<=training['steps']<=18000 or type(training.get('seed'))is not int
        or set(rates)!= {'means','quats','log_scales','opacity_logits','sh'}
        or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in rates.values())):
        issue('invalid_fixed_training_budget_or_rates')
    source=m.get('source',{});source_sha=source.get('asset',{}).get('sha256')
    source_path=asset('source',source.get('asset'),False)
    width,height,count=(source.get(k) for k in ['width','height','encoded_frame_count'])
    if any(type(v)is not int or v<=0 for v in [width,height,count]):issue('source_raster_and_encoded_frame_count_required')
    frame_plan=asset('frozen_training_frame_plan',m.get('frame_plan'))
    required=[]
    if frame_plan is not None:
        required=frame_plan.get('frame_ids',[])
        if frame_plan.get('kind')!='preregistered_training_frames' or frame_plan.get('source_sha256')!=source_sha:
            issue('frame_plan_source_or_purpose_mismatch')
        if not required or len(required)!=len(set(required)) or any(type(x)is not int or x<0 or (type(count)is int and x>=count) for x in required):
            issue('invalid_training_frame_identity_list');required=[]
    camera_data=asset('cameras',m.get('cameras'))
    cameras={};K=None;camera_records=[]
    if camera_data is not None:
        try:
            if camera_data['source_sha256']!=source_sha or camera_data['pixel_convention']!='integer_centers':raise ValueError('Camera/source pixel contract mismatch')
            camera_records=camera_data['cameras']
            if not camera_records or len({c['frame_id'] for c in camera_records})!=len(camera_records):raise ValueError('Duplicate or empty camera identities')
            K,cameras=checked_cameras(camera_data['K'],{c['frame_id']:c['w2c'] for c in camera_records})
            if any(c['registration']!='observed' for c in camera_records):issue('unobserved_or_interpolated_camera')
            if required and set(cameras)!=set(required):issue('camera_frame_set_mismatch',missing=sorted(set(required)-set(cameras)),extra=sorted(set(cameras)-set(required)))
        except (KeyError,TypeError,ValueError) as e:issue('invalid_camera_contract',reason=str(e));K=None;cameras={}
    decode=asset('native_decode_receipt',m.get('decode_receipt'))
    decoded={}
    if decode is not None:
        if decode.get('source_sha256')!=source_sha or decode.get('geometric_resampling') is not False or decode.get('rotation_applied') is not False:
            issue('decode_source_or_geometry_mismatch')
        values=decode.get('frames',[])
        if len({r.get('frame_id') for r in values})!=len(values):issue('duplicate_decode_frame_identity')
        decoded={r.get('frame_id'):r for r in values}
    frames=m.get('frames',[])
    ids=[f.get('frame_id') for f in frames]
    if any(type(f)is not int or f<0 for f in ids) or len(ids)!=len(set(ids)):issue('duplicate_or_invalid_image_frame_identity')
    if required and set(ids)!=set(required):issue('image_frame_set_mismatch',missing=sorted(set(required)-set(ids)),extra=sorted(set(ids)-set(required)))
    if not frames:issue('native_training_images_missing')
    for f in frames:
        frame=f.get('frame_id');loaded={};row=dict(frame_id=frame,valid=False)
        for key in ['rgb','alpha','rgb_valid','observation_valid']:
            p=asset(f'{frame}:{key}',f.get(key),False)
            if p is None:continue
            try:
                with Image.open(p) as im:
                    if im.format!='PNG' or im.size!=(width,height):raise ValueError('Exact native PNG raster required')
                    if (key=='rgb' and im.mode!='RGB') or (key!='rgb' and im.mode!='L'):raise ValueError('RGB image and L masks required without implicit conversion')
                    loaded[key]=np.array(im)
            except (OSError,ValueError) as e:issue('invalid_native_image',frame_id=frame,asset=key,reason=str(e))
        if 'rgb' in loaded:
            d=decoded.get(frame,{})
            if d.get('rgb_sha256')!=f['rgb']['sha256'] or d.get('width')!=width or d.get('height')!=height:
                issue('image_decode_identity_mismatch',frame_id=frame)
        if len(loaded)==4:
            mask=loaded['rgb_valid'];valid=loaded['observation_valid']
            if not np.isin(mask,[0,255]).all() or not np.isin(valid,[0,255]).all():issue('nonboolean_supervision_mask',frame_id=frame)
            else:
                mask=mask==255;valid=valid==255
                if not mask.any() or np.any(mask&~valid):issue('foreground_outside_observation_support',frame_id=frame)
                if np.any(mask&(loaded['alpha']==0)):issue('foreground_rgb_supervises_declared_empty_pixels',frame_id=frame)
                integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
                windows=integral[11:,11:]-integral[:-11,11:]-integral[11:,:-11]+integral[:-11,:-11]
                if not np.any(windows==121):issue('no_complete_foreground_SSIM_window',frame_id=frame)
                row.update(rgb_pixels=int(mask.sum()),observed_pixels=int(valid.sum()),valid=True)
        frame_rows.append(row)
    records=asset('material_observations',m.get('observations'))
    observations=[]
    if records is not None:
        try:
            observations=[observation(r) for r in records]
            checked_collection(observations)
            if any(o.source_sha256!=source_sha or o.width!=width or o.height!=height or type(o.frame_id)is not int or type(o.track_id)is not int for o in observations):raise ValueError('Observation/source/raster mismatch')
        except (KeyError,TypeError,ValueError) as e:issue('invalid_material_observations',reason=str(e));observations=[]
    fit_families=[] if camera_data is None else camera_data.get('fit_families',[])
    if not fit_families or len(set(fit_families))!=len(fit_families):issue('camera_fit_family_manifest_missing_or_duplicate')
    fitobs=[o for o in observations if o.frozen_split=='fit_background']
    if set(o.frozen_family_id for o in fitobs)!=set(fit_families):issue('camera_fit_material_support_incomplete')
    for o in fitobs:
        p=camera_fit_permission(o)
        if not p['allowed']:fit_fail.append(dict(key=list(o.key),reasons=p['reasons']))
    if fit_fail:issue('camera_fit_material_unverified',observations=len(fit_fail))
    held=[o for o in observations if o.frozen_split!='fit_background']
    if K is not None and cameras and observations:
        # Same immutable N13 computation, now used by the entry; not a new solve.
        geometry=evaluate_material_geometry(K,cameras,held,fit_families,
            m.get('cameras',{}).get('sha256'),required,source_sha)
        if not geometry['material_geometry_component_accepted']:issue('independent_material_geometry_failed',failed=sum(not r['material_geometry_component_accepted'] for r in geometry['observations']))
    else:issue('independent_material_geometry_unavailable')
    coverage=asset('frozen_visibility_region_plan',m.get('coverage_plan'))
    missing_cells=[];visibility_exclusions=[]
    if coverage is not None:
        if coverage.get('source_sha256')!=source_sha or coverage.get('frame_plan_sha256')!=refhash('frame_plan'):
            issue('coverage_source_or_frame_plan_mismatch')
        annotations=coverage.get('observations',[]);key_regions={};validkeys={o.key:o for o in observations}
        for ann in annotations:
            key=tuple(ann.get('key',[]))
            if key not in validkeys or key in key_regions:issue('invalid_or_duplicate_region_annotation');continue
            if asset('region_annotation_evidence',ann.get('evidence'),False) is None:continue
            region=ann.get('region');o=validkeys[key]
            expected_group='face_candidate' if region in FACE_REGIONS else 'body_candidate' if region in BODY_REGIONS else 'static_scene'
            if o.group!=expected_group or o.frozen_split!=('heldout_background' if region in SCENE_REGIONS else 'heldout_human'):
                issue('region_annotation_not_matching_heldout_material_group');continue
            key_regions[key]=ann.get('region')
        requirements=coverage.get('requirements',[])
        required_cells=set()
        for req in requirements:
            cell=(req.get('frame_id'),req.get('region'))
            if cell in required_cells or cell[0] not in required or cell[1] not in FACE_REGIONS|BODY_REGIONS|SCENE_REGIONS:
                issue('invalid_or_duplicate_coverage_requirement');continue
            required_cells.add(cell)
            if asset('visibility_requirement_evidence',req.get('evidence'),False) is None:continue
            if req.get('visibility')=='occluded_or_out_of_frame' and req.get('reason'):
                visibility_exclusions.append(dict(frame_id=cell[0],region=cell[1],reason=req['reason']));continue
            if req.get('visibility')!='visible':issue('unknown_visibility_requirement');continue
            good=[] if geometry is None else [r for r in geometry['observations'] if r['observation']['frame_id']==cell[0] and key_regions.get(tuple(r['prediction']['target_key']))==cell[1] and r['material_geometry_component_accepted']]
            if not good:missing_cells.append(list(cell))
        if not required_cells:issue('empty_region_coverage_plan')
        expected={(f,r) for f in required for r in FACE_REGIONS|BODY_REGIONS|SCENE_REGIONS}
        if required_cells!=expected:issue('visibility_plan_does_not_cover_all_frame_regions',missing_count=len(expected-required_cells))
        visible={r.get('region') for r in requirements if r.get('visibility')=='visible'}
        if not (FACE_REGIONS|SCENE_REGIONS)<=visible:issue('required_face_or_scene_regions_never_observed')
        if missing_cells:issue('independent_region_view_coverage_failed',cells=missing_cells)
    assessments=asset('physical_camera_and_mask_evidence',m.get('physical_evidence'))
    if assessments is not None:
        bindings=dict(source_sha256=source_sha,cameras_sha256=refhash('cameras'),
            observations_sha256=refhash('observations'),frame_plan_sha256=refhash('frame_plan'),
            decode_receipt_sha256=refhash('decode_receipt'),frames_sha256=hashlib.sha256(json.dumps(frames,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        if assessments.get('bindings')!=bindings:issue('physical_assessment_binding_mismatch')
        for name in ['output_projection','pose_depth_support','static_scene_control','subject_staticity','mask_occluder_support','world_y_up']:
            item=assessments.get(name,{})
            # Unknown remains unknown; a self-declared training bool is not consumed.
            if item.get('state')!='supported' or not item.get('method') or not item.get('limitations'):
                issue('physical_assessment_incomplete',assessment=name)
            else:asset('physical_evidence:'+name,item.get('evidence'),False)
    if sha(path)!=manifest_sha:issue('manifest_changed_during_audit')
    for p,h in assets.items():
        if sha(p)!=h:issue('input_changed_during_audit',path=p)
    error_rows=[] if geometry is None else geometry['observations']
    face=[r for r in error_rows if r['observation']['group']=='face_candidate']
    errors=[r['error_px'] for r in face if r['error_px'] is not None]
    report=dict(schema='shuolee-native-training-admission-v1',manifest_sha256=manifest_sha,
        scope=m.get('scope'),run_id=m.get('run_id'),training_admitted=not issues,
        issues=issues,issue_counts=dict(Counter(i['kind'] for i in issues)),
        encoded_source_frames=count,declared_training_frames=required,registered_camera_frames=sorted(cameras),
        native_frame_audits=frame_rows,fit_material_failures=fit_fail,material_geometry=geometry,
        source_observation_records=len(records) if isinstance(records,list) else 0,
        heldout_face=dict(observations=len(face),numerical_predictions=len(errors),maximum_px=max(errors) if errors else None,
                         accepted=sum(r['material_geometry_component_accepted'] for r in face)),
        missing_region_cells=missing_cells,visibility_exclusions=visibility_exclusions,
        asset_hashes=assets,images_resampled=False,K_optimized=False,cameras_modified=False,
        training_started=False,initializer_called=False,optimizer_constructed=False,
        inference='Evidence-backed input admission only. No claim of trained appearance, novel-view quality or cross-person generalization.')
    return report,m

def assert_snapshot(report,path):
    if not report['training_admitted'] or sha(path)!=report['manifest_sha256']:raise RuntimeError('Training input not admitted or manifest changed')
    if any(sha(p)!=h for p,h in report['asset_hashes'].items()):raise RuntimeError('Admitted input changed before training')

def native_camera(record,camera_document,width,height,image_name):
    K,_,receipt=to_native_pixels(camera_document['K'],np.empty((0,2)),intrinsic_convention='integer_centers',observation_convention='integer_centers')
    camera=dict(id=str(record['frame_id']),image_name=image_name,width=width,height=height,K=K.tolist(),w2c=record['w2c'])
    return camera,receipt
