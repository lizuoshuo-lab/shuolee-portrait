"""Create isolated R5 consumers; never edit frozen T3/N13/live modules."""
from pathlib import Path
H=Path(__file__).parent

header='''"""Fixed complete output cameras for support-only material prediction.

No fitting, pixel/image warping or physical-evidence promotion. The camera is
the exact corner-coordinate K/E/D record consumed by the native renderer.
"""
from collections import defaultdict
import copy, math
import numpy as np
from material_observations122 import point_reasons, camera_fit_permission, independent_face_admission
from material_geometry122 import checked_collection
from native_residual_camera122 import validate_camera, undistort_normalized, project_world
from native_residual_domain122 import sampled_domain_check

def checked_cameras(cameras):
    out={}
    for frame,c in cameras.items():
        if type(frame)is not int or frame<0 or c.get('id')!=str(frame):
            raise ValueError('Explicit integer frame and matching camera identity required')
        validate_camera(c)
        E=np.asarray(c['w2c'],dtype=float)
        if not np.allclose(E[:3,:3]@E[:3,:3].T,np.eye(3),atol=1e-8,rtol=0) or abs(np.linalg.det(E[:3,:3])-1)>1e-8:
            raise ValueError('Admission requires a proper world-to-camera rigid pose')
        sampled_domain_check(c)
        out[frame]=copy.deepcopy(c)
    return out

def canonical_triangulate(cameras,support):
    reference=min(o.frame_id for o in support)
    E=np.asarray(cameras[reference]['w2c'],dtype=float); inv=np.linalg.inv(E)
    local={o.frame_id:np.asarray(cameras[o.frame_id]['w2c'],dtype=float)@inv for o in support}
    baseline=max(float(np.linalg.norm(M[:3,3])) for M in local.values())
    if not math.isfinite(baseline) or baseline<=0:return np.full(3,np.nan)
    rows=[]
    for o in support:
        c=cameras[o.frame_id]; K=np.asarray(c['K'],dtype=float)
        uv=np.asarray(o.raw_uv,dtype=float)+.5
        xy=undistort_normalized(((uv-K[:2,2])/[K[0,0],K[1,1]])[None],c['residual_projection'])[0]
        M=local[o.frame_id].copy();M[:3,3]/=baseline
        rows.extend([K[0,0]*(xy[0]*M[2]-M[0]),K[1,1]*(xy[1]*M[2]-M[1])])
    _,_,vt=np.linalg.svd(np.asarray(rows));q=vt[-1]
    if abs(q[3])<=1e-12:return np.full(3,np.nan)
    point=q[:3]/q[3]*baseline
    return (point-E[:3,3])@E[:3,:3]

def predict_target(cameras,target,track,camera_fit_families,camera_evidence):
    support=[o for o in track if o.frame_id!=target.frame_id]
    row=dict(target_key=list(target.key),uv=None,
        triangulation_observation_keys=[list(o.key) for o in support],
        camera_fit_families=list(camera_fit_families),frozen_camera_evidence=camera_evidence,
        target_used_in_triangulation=False,numerical_reasons=[],
        output_projection='opencv_brown12_output',reported_pixels='integer_centers',
        explicit_pixel_bridge='observed+0.5 to exact rendererK/E/D; projected-0.5 to raw observation')
    missing=[o.frame_id for o in [target]+support if o.frame_id not in cameras]
    if missing:row['numerical_reasons'].append('missing_camera_frames:'+','.join(map(str,sorted(set(missing)))))
    if len(support)<2:row['numerical_reasons'].append('insufficient_other_views')
    if any(not o.tracker_numeric_validity or not np.isfinite(o.raw_uv).all() for o in [target]+support):
        row['numerical_reasons'].append('invalid_native_measurement')
    if row['numerical_reasons']:return row
    try:
        X=canonical_triangulate(cameras,support)
        if not np.isfinite(X).all():raise ValueError('invalid_triangulation')
        depths=[]
        for o in [target]+support:
            E=np.asarray(cameras[o.frame_id]['w2c']);depths.append(float((E[:3,:3]@X+E[:3,3])[2]))
        if any(z<=0 or not math.isfinite(z) for z in depths):raise ValueError('nonpositive_depth')
        uv=project_world(X[None],cameras[target.frame_id])[0]-.5
        if not np.isfinite(uv).all():raise ValueError('nonfinite_projection')
    except (ValueError,np.linalg.LinAlgError) as e:
        row['numerical_reasons'].append(str(e));return row
    row.update(uv=uv.tolist(),positive_depth=True,point_world_diagnostic=X.tolist(),
        triangulation='Inverse residual rays, fx/fy weighted algebraicDLT in earliest-support-camera/max-support-baseline gauge; not nonlinear geometric optimum')
    return row

'''
original=(H/'material_geometry122.py').read_text(encoding='utf-8')
tail=original[original.index('def evaluate_material_geometry('):]
tail=tail.replace('def evaluate_material_geometry(K,cameras,','def evaluate_material_geometry(cameras,')
tail=tail.replace('keys,tracks=checked_collection(observations);K,cameras=checked_cameras(K,cameras)',
    'keys,tracks=checked_collection(observations);cameras=checked_cameras(cameras)\n    if any(o.frame_id in cameras and (o.width,o.height)!=(cameras[o.frame_id]["width"],cameras[o.frame_id]["height"]) for o in observations):\n        raise ValueError("Observation and camera raster mismatch")')
tail=tail.replace('pred=predict_target(K,cameras,','pred=predict_target(cameras,')
out=H/'material_geometry122_v2.py';assert not out.exists();out.write_text(header+tail,encoding='utf-8')

s=(H/'native_training_admission122.py').read_text(encoding='utf-8')
s=s.replace('Existing N3/N13 geometry remains the numerical authority.', 'N3 material semantics remain unchanged; explicit per-frame residual rays replace pinhole-only geometry.')
s=s.replace('import hashlib, json, math','import copy, hashlib, json, math')
s=s.replace('from material_geometry122 import checked_collection, checked_cameras, evaluate_material_geometry',
 'from material_geometry122_v2 import checked_collection, checked_cameras, evaluate_material_geometry\nfrom native_residual_camera122 import ResidualRenderContract')
s=s.replace("SCHEMA='shuolee-native-training-input-v1'", "SCHEMA='shuolee-native-training-input-v2-output-residual'")
s=s.replace("if m.get('projection')!='pinhole' or m.get('radial_coefficients') is not None:issue('captured_export_requires_explicit_verified_output_projection')", "if m.get('projection')!='opencv_brown12_output' or 'radial_coefficients' in m:issue('captured_export_requires_explicit_verified_output_projection')")
s=s.replace('cameras={};K=None;camera_records=[]', 'cameras={};camera_records=[]')
s=s.replace("if camera_data['source_sha256']!=source_sha or camera_data['pixel_convention']!='integer_centers':raise ValueError('Camera/source pixel contract mismatch')", "if (camera_data.get('schema')!='shuolee-output-residual-cameras-v1' or camera_data['source_sha256']!=source_sha\n                or camera_data['measurement_pixel_convention']!='integer_centers' or camera_data['camera_pixel_convention']!='pixel_corners' or 'K' in camera_data):\n                raise ValueError('Explicit per-camera residual/source/pixel contract required')")
s=s.replace("K,cameras=checked_cameras(camera_data['K'],{c['frame_id']:c['w2c'] for c in camera_records})", "cameras=checked_cameras({c['frame_id']:c['camera'] for c in camera_records})\n            domain='synthetic_known_camera' if m['scope']=='synthetic-control-only' else 'processed_output_pixels'\n            if any(c['width']!=width or c['height']!=height or c['residual_projection']['input_domain']!=domain for c in cameras.values()):\n                raise ValueError('Native raster or captured/synthetic output domain mismatch')")
s=s.replace("issue('invalid_camera_contract',reason=str(e));K=None;cameras={}", "issue('invalid_camera_contract',reason=str(e));cameras={}")
s=s.replace('if K is not None and cameras and observations:', 'if cameras and observations:')
s=s.replace('# Same immutable N13 computation, now used by the entry; not a new solve.', '# Same N3 material semantics, full output K/E/D for every support ray and target. No new solve.')
s=s.replace('geometry=evaluate_material_geometry(K,cameras,held,fit_families,', 'geometry=evaluate_material_geometry(cameras,held,fit_families,')
s=s.replace("schema='shuolee-native-training-admission-v1'", "schema='shuolee-native-training-admission-v2-output-residual'")
s=s.replace("scope=m.get('scope'),run_id=m.get('run_id'),training_admitted=not issues,", "scope=m.get('scope'),run_id=m.get('run_id'),training_admitted=not issues,\n        camera_projection='opencv_brown12_output',camera_parameters_sha256=refhash('cameras'),\n        image_pixels='integer_centers',render_pixels='pixel_corners',physical_calibration_inferred=False,")
s=s[:s.index('def native_camera(')]+'''def native_camera(record,camera_document,width,height,image_name):
    """Return exactly the validated renderer camera; never reconstruct it from FOV."""
    if camera_document.get('schema')!='shuolee-output-residual-cameras-v1':raise ValueError('Explicit residual document required')
    if camera_document.get('measurement_pixel_convention')!='integer_centers' or camera_document.get('camera_pixel_convention')!='pixel_corners':
        raise ValueError('Explicit measurement/render pixel bridge required')
    c=checked_cameras({record['frame_id']:record['camera']})[record['frame_id']]
    if (c['width'],c['height'])!=(width,height):raise ValueError('No implicit raster resize')
    if c['image_name']!=image_name:raise ValueError('Native camera/image identity differs')
    receipt=dict(camera_sha256=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        renderer_camera_unchanged=True,extrinsics_modified=False,coefficients_modified=False,
        observations_to_renderer_pixels='+0.5',projection_to_observations='-0.5')
    return c,receipt

def main():
    import argparse,sys
    p=argparse.ArgumentParser(description='CPU prepared-input audit with explicit complete output-residual cameras; no training.')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Fresh audit output required; no overwrite')
    out.mkdir(parents=True)
    r,m=audit_input(args.input);save(out/'admission.json',r)
    status=dict(stage='residual_input_admission',input_admitted=r['training_admitted'],training_started=False,
        GPU_imported='torch' in sys.modules,initializer_loaded=False,optimizer_constructed=False)
    save(out/'status.json',status)
    print(json.dumps(dict(exit_code=0 if r['training_admitted'] else 3,heldout_face=r['heldout_face'])))
    return 0 if r['training_admitted'] else 3

if __name__=='__main__':
    import sys
    sys.exit(main())
'''
out=H/'native_training_admission122_v2.py';assert not out.exists();out.write_text(s,encoding='utf-8')
print('R5_VERSIONED_CONSUMERS_WRITTEN')
