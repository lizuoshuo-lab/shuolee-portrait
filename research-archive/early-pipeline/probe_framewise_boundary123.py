"""One registered CPU boundary repair control; no model, renderer or training."""
from pathlib import Path
import argparse, hashlib, importlib.util, io, json, os, sys, time, traceback, unittest
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def summary(values):
    if not len(values): return dict(count=0)
    return dict(count=len(values), minimum=float(np.min(values)), median=float(np.median(values)),
                p90=float(np.quantile(values,.9)), p99=float(np.quantile(values,.99)), maximum=float(np.max(values)))


def independent_recount(xyz, bundle):
    """Saved cube translations + direct K/E and exact foreground-square union.

    Does not call the initializer's field, distance, transport, voting or support
    functions. Ground-truth continuous cube rays are a separate diagnostic.
    """
    positive=np.zeros(len(xyz),np.uint32); valid=np.zeros(len(xyz),np.uint32)
    continuous=np.zeros(len(xyz),np.uint32); distances=[]; view_stats=[]
    for name in bundle.report()['frame_ids']:
        frame=bundle.frame(name)
        offset=frame.vertices_world-bundle.reference.vertices_world
        shift=offset[0]
        if not np.allclose(offset,shift,atol=1e-14): raise ValueError('Fixture is not the registered known translation.')
        E,K=frame.w2c,frame.K_integer_centers
        world=xyz+shift
        camera=world@E[:3,:3].T+E[:3,3]
        hom=camera@K.T;uv=hom[:,:2]/hom[:,2:]
        w,h=frame.wh
        seen=(camera[:,2]>0)&(uv[:,0]>=-.5)&(uv[:,0]<w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<h-.5)
        pixel=np.floor(uv[seen]+.5).astype(np.int64)
        with Image.open(frame.mask) as im: mask=np.asarray(im.convert('L'))==255
        hit=np.zeros(len(xyz),bool);hit[seen]=mask[pixel[:,1],pixel[:,0]]
        positive+=hit;valid+=seen
        failed_uv=uv[seen & ~hit]
        foreground=np.argwhere(mask)[:,::-1].astype(float)
        local=[]
        for begin in range(0,len(failed_uv),2048):
            delta=np.maximum(np.abs(failed_uv[begin:begin+2048,None]-foreground[None])-.5,0.)
            local.extend(np.sqrt((delta*delta).sum(-1)).min(-1).tolist())
        distances.extend(local)
        C=-E[:3,:3].T@E[:3,3]
        ray=world-C;origin=C-np.array([0.,0.,4.])-shift
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(-.45-origin)/ray;b=(.45-origin)/ray
        ray_hit=np.maximum(np.minimum(a,b).max(-1),0)<=np.maximum(a,b).min(-1)
        continuous+=ray_hit
        view_stats.append(dict(frame=name,background=int((seen & ~hit).sum()),unknown=int((~seen).sum()),
                               failed_native_square_distance_px=summary(local)))
    all_votes=(valid>=6)&(positive/np.maximum(valid,1)>.9)
    all_continuous=continuous==6
    d=np.array(distances)
    return dict(points=len(xyz),positive_histogram={str(i):int((positive==i).sum()) for i in np.unique(positive)},
        all_six_known=bool((valid==6).all()), original_rule_passed=int(all_votes.sum()),
        original_rule_failed=int((~all_votes).sum()), digital_fail_but_continuous_all_hit=int((~all_votes & all_continuous).sum()),
        digital_and_continuous_fail=int((~all_votes & ~all_continuous).sum()),
        failed_observation_distance_px=summary(d), boundary_buckets_observations=dict(
            within_half_native_pixel=int((d<=.5).sum()), above_half_to_one_pixel=int(((d>.5)&(d<=1)).sum()),
            above_one_pixel=int((d>1).sum())), per_frame=view_stats), positive,valid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    out=args.output_dir; base=out.parent
    if (out/'progress.json').exists() or (out/'result.json').exists(): raise SystemExit('Refuse repeat of registered run.')
    plan=json.loads((out/'plan.json').read_text(encoding='utf-8'))
    source=args.project/'src/shuolee_portrait/framewise_clothed_surface.py'
    test=args.project/'tests/test_framewise_surface_boundary.py'
    for path,expected in plan['run_source_sha256'].items():
        resolved={'source':source,'test':test,'probe':Path(__file__)}[path]
        if sha(resolved)!=expected: raise ValueError('Frozen source changed: '+path)
    for path,expected in plan['frozen_input_sha256'].items():
        if sha(base/path)!=expected: raise ValueError('Frozen synthetic input changed: '+path)
    if sha(base/'result.json')!=plan['baseline_result_sha256']: raise ValueError('Baseline changed.')
    sys.path.insert(0,str(args.project/'src'))
    from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle
    from shuolee_portrait.framewise_clothed_surface import ReferenceGrid,initialize_framewise_surface
    start=time.monotonic();save(out/'progress.json',dict(state='running',pid=os.getpid(),CPU_only=True))
    report=dict(schema='boundary-correction-result/1',passed=False,CPU_only=True,GPU_calls=0,
                body_forwards=0,fit_steps=0,training_steps=0,real_person_images=0,real_quality_admitted=False,
                all_200k_rule_pass_is_not_training_gate=True,source_sha256=plan['run_source_sha256'],
                baseline_preserved=True,plan_sha256=sha(out/'plan.json'))
    try:
        spec=importlib.util.spec_from_file_location('boundary_new_controls',test)
        controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(controls.PixelBoundaryControls)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['controls']=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),transcript=stream.getvalue())
        if not result.wasSuccessful(): raise RuntimeError('Named independent controls failed before cube run.')
        bundle=load_frame_pose_bundle(base/'cube-input/manifest.json')
        old=json.loads((base/'initializer-result.json').read_text(encoding='utf-8'))
        grid=ReferenceGrid.from_bounds([-.8,-.8,3.2],[.8,.8,4.8],spacing_m=.1,reference_frame_id='reference',max_voxels=100000)
        frames=bundle.report()['frame_ids']
        masks={f:np.ones((bundle.frame(f).wh[1],bundle.frame(f).wh[0]),bool) for f in frames}
        notes={f:'Known synthetic six-view cube; every native pixel has observed foreground/background.' for f in frames}
        length=dict(translation_unit='meters',meters_per_camera_unit=1.,basis='synthetic_known',evidence='Saved known synthetic cube in model units; not physical footage.')
        initialized=initialize_framewise_surface(bundle,grid,reference_frame_id='reference',valid_masks=masks,
            validity_notes=notes,camera_length_contract=length,min_valid_views=old['min_valid_views'],
            vote_fraction=old['vote_fraction'],seed=old['seed'],initial_points=old['initial_points'])
        if initialized.report()['grid_axes_sha256']!=old['grid_axes_sha256']: raise ValueError('Grid axes changed.')
        report['initializer']=initialized.report()
        save(out/'initializer-result.json',initialized.report())
        np.savez_compressed(out/'corrected-initializer.npz',xyz_ref=initialized.xyz_ref,
            surface_vertices_ref=initialized.surface_vertices_ref,surface_faces=initialized.surface_faces,
            positive=initialized.sample_positive_counts,valid=initialized.sample_valid_counts)
        for name,xyz in [('new_samples',initialized.xyz_ref),('new_vertices',initialized.surface_vertices_ref)]:
            report[name],p,v=independent_recount(xyz,bundle)
            if name=='new_samples' and (not np.array_equal(p,initialized.sample_positive_counts) or
                                       not np.array_equal(v,initialized.sample_valid_counts)):
                raise ValueError('Independent sample recount disagrees.')
        with np.load(base/'synthetic-initializer.npz',allow_pickle=False) as archive:
            report['baseline_samples'],_,_=independent_recount(archive['xyz_ref'],bundle)
            report['baseline_vertices'],_,_=independent_recount(archive['surface_vertices_ref'],bundle)
        old_max=report['baseline_samples']['failed_observation_distance_px']['maximum']
        new_max=report['new_samples']['failed_observation_distance_px'].get('maximum',0.)
        report['improvement']=dict(baseline_mask_rule_failures=report['baseline_samples']['original_rule_failed'],
            new_mask_rule_failures=report['new_samples']['original_rule_failed'],
            baseline_max_boundary_distance_px=old_max,new_max_boundary_distance_px=new_max,
            fewer_hard_vote_violations=report['new_samples']['original_rule_failed']<report['baseline_samples']['original_rule_failed'],
            smaller_max_boundary_distance=new_max<old_max,
            no_camera_quality_inference=True,no_all_points_pass_admission_gate=True)
        report['passed']=bool(report['improvement']['fewer_hard_vote_violations'] and
                              report['improvement']['smaller_max_boundary_distance'])
        report['elapsed_seconds']=time.monotonic()-start
        if report['elapsed_seconds']>plan['budget']['CPU_wall_seconds']: raise RuntimeError('Registered CPU budget exceeded.')
    except Exception as exc:
        report['passed']=False;report['error']=repr(exc);report['traceback']=traceback.format_exc()
    report['elapsed_seconds']=time.monotonic()-start
    save(out/'result.json',report)
    print(json.dumps({k:report.get(k) for k in ('passed','error','elapsed_seconds','improvement')}))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
