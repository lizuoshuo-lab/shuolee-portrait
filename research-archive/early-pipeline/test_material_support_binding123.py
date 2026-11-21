from pathlib import Path
from datetime import datetime,timezone
import tempfile,json,copy,time,traceback
import local_surface123_frozen_observations as mod
from test_frozen_camera_entry123 import make_fixture,ref,save,read,sha

H=Path(__file__).resolve().parent
OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/camera-observation-entry')

def main():
    assert not (OUT/'result-completed.json').exists()
    previous=read(OUT/'result.json');assert previous['passed']==18
    results=[];start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='shuolee-support-binding-') as temp:
        s,d,m=make_fixture(Path(temp)/'SYNTHETIC_ONLY')
        frames=read(Path(d['feature_manifest']['path']))['frames']
        rp=Path(d['physical_review']['path']);review=read(rp)
        for r in review['rows']:
            p=Path(temp)/('support-'+r['segment_id']+'.json')
            support=dict(schema='shuolee-material-support/1',synthetic_control=True,
                source_video_sha256=s['video_sha256'],segment_id=r['segment_id'],family_id=r['family_id'],
                material_role='static_background',material_identity='confirmed',motion_state='bounded_static',
                basis='Synthetic schema exercise only. Not a physical calibration or actual material acceptance.',observations=[])
            for v in r['actual_saved_views_reviewed']:
                f=frames[v['slot']]
                support['observations'].append(dict(**v,image_sha256=f['source_sha256'],feature_sha256=f['sha256'],visibility='visible',measurement_uncertainty_px=.2))
            save(p,support);r['supporting_evidence']=[ref(p)]
        save(rp,review);d['physical_review']=ref(rp);save(m,d);s['frozen_observations']=ref(m)
        def load():return mod.load_subject_input(s)
        b=load();fit,held=mod.solver_tracks(b)
        assert len(fit)==2 and len(held)==2 and b.tracks[0].observations[1].native_xy==(11.,10.)
        results.append(dict(name='synthetic_exact_evidence_scope_transports_original_FIT_and_HELD',passed=True))
        p=Path(review['rows'][0]['supporting_evidence'][0]['path']);baseline=p.read_bytes()
        def reject(name,change,expected):
            e=read(p)
            try:
                change(e);save(p,e)
                r=copy.deepcopy(review);r['rows'][0]['supporting_evidence']=[ref(p)]
                save(rp,r);dd=copy.deepcopy(d);dd['physical_review']=ref(rp);save(m,dd);ss=dict(s,frozen_observations=ref(m))
                try:mod.load_subject_input(ss)
                except mod.ObservationContractError as err:assert expected in str(err),(expected,str(err))
                else:raise AssertionError('Wrong evidence scope accepted: '+name)
                results.append(dict(name=name,passed=True))
            finally:p.write_bytes(baseline);save(rp,review);save(m,d)
        reject('support_from_another_video_rejected',lambda e:e.__setitem__('source_video_sha256','0'*64),'material_support_source_mismatch')
        reject('support_from_another_track_rejected',lambda e:e.__setitem__('segment_id','wrong'),'material_support_track_mismatch')
        reject('support_wrong_source_frame_rejected',lambda e:e['observations'][0].__setitem__('source_frame',2),'material_support_frame_mismatch')
        reject('support_shifted_native_pixel_rejected',lambda e:e['observations'][0]['uv'].__setitem__(0,10.2),'material_support_pixel_mismatch')
        reject('support_occluded_point_rejected',lambda e:e['observations'][0].__setitem__('visibility','occluded'),'material_support_visibility_unknown')
        reject('support_missing_node_rejected',lambda e:e['observations'].pop(),'material_support_incomplete_coverage')
        reject('support_missing_uncertainty_rejected',lambda e:e['observations'][0].__setitem__('measurement_uncertainty_px',None),'material_support_uncertainty_unknown')
    # The real records carried no positive static-material claims. This patch
    # does not change their prior rejection, or rewrite/re-evaluate geometry.
    initial_manifest=read(OUT/'code-manifest.json')
    save(OUT/'code-manifest-initial-18-checks.json',initial_manifest)
    final_manifest=dict(created=datetime.now(timezone.utc).isoformat(),purpose=initial_manifest['purpose'],
        files=[dict(name=r['name'],sha256=sha(H/r['name'])) for r in initial_manifest['files']])
    save(OUT/'code-manifest.json',final_manifest)
    result=dict(previous,updated=datetime.now(timezone.utc).isoformat(),state='engineering_entry_and_support_binding_passed_no_camera_admitted',
        initial_result='result.json',initial_checks=18,support_binding_checks=results,total_checks=18+len(results),
        current_code_manifest_sha256=sha(OUT/'code-manifest.json'),support_test_seconds=time.monotonic()-start,
        evidence_scope_verified=True,physical_evidence_truth_proven_by_parser=False)
    save(OUT/'result-completed.json',result)
    print(json.dumps(dict(initial_checks=18,additional_checks=len(results),total_checks=result['total_checks'],quality_accepted=False,geometry_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'support-test-failure.json',dict(error=repr(e),traceback=traceback.format_exc(),training=False))
        raise
