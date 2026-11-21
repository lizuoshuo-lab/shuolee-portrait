"""One exact-config replay solely to preserve the failed900-step state.

The first run's geometry assertion exited before its first scheduled snapshot.
Keep that failure untouched. Capture this replay before enforcing the same gate;
no thresholds, data, loss or optimizer changes and no further training budget.
"""
from pathlib import Path
import argparse,ast,copy,json,traceback
import train_smplify_eight123 as eight
from run_official_gaussians123 import read,save,sha,now

OUT=eight.OUT.parent/'body-gaussian-smplify-eight-failure-replay'


def source():
    _,_,s=eight.source()
    def replace(a,b):
        nonlocal s
        assert s.count(a)==1,a;s=s.replace(a,b,1)
    replace("            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0", """            if step==800:
                old=json.loads((ORIGINAL/'gaussian-run/loss.jsonl').read_text().splitlines()[-1])
                errors={k:abs(float(v)-old[k]) for k,v in losses.items()}
                save(run/'replay-check.json',dict(updated=now(),step=step,loss_abs_differences=errors,
                    same_configuration=True,bitwise_trajectory_claim=False))
                assert max(errors.values())<1e-6,errors
            if gate['new_relative_normal_reversals'] or step==900:
                from facial_geometry_boundary123 import clothing_only_offset
                d=clothing_only_offset(model.static_offset,model.surface_clothing,model.surface_protected)
                v=model.surface_base+(model.surface_linear@d[None,:,:,None]).squeeze(-1)
                p0=model.surface_base[:,model.faces];p1=v[:,model.faces]
                n0=torch.cross(p0[:,:,1]-p0[:,:,0],p0[:,:,2]-p0[:,:,0],dim=-1)
                n1=torch.cross(p1[:,:,1]-p1[:,:,0],p1[:,:,2]-p1[:,:,0],dim=-1)
                bad=torch.nonzero((n0*n1).sum(-1)<=0).cpu().tolist()
                details=[]
                for frame_slot,face in bad:
                    vids=model.faces[face]
                    edges=[float((p0[frame_slot,face,a]-p0[frame_slot,face,b]).norm()) for a,b in [(0,1),(1,2),(2,0)]]
                    details.append(dict(frame=rows[frame_slot]['frame'],face=face,vertices=vids.cpu().tolist(),
                        optimized_image=rows[frame_slot]['frame'] in selected,initial_edges_m=edges,
                        initial_area_m2=float(n0[frame_slot,face].norm()/2),current_area_m2=float(n1[frame_slot,face].norm()/2),
                        maximum_vertex_move_m=float((v[frame_slot,vids]-model.surface_base[frame_slot,vids]).norm(dim=-1).max())))
                save(run/'failure-location.json',dict(updated=now(),step=step,geometry=gate,triangles=details,
                    original_failure=read(ORIGINAL/'gaussian-run/geometry-progress.json'),
                    reason='Preserved same-config replay, not the lost original process memory; no quality promotion.'))
                scene.save(step)
                status('geometry_failure_replay_preserved',step,geometry=gate)
                raise ControlComplete()
            assert gate['protected_face_motion_m']==0 and gate['protected_face_gradient']==0""")
    replace('            if step == 1000:\n                suspended_baseline()', '            if step == 900:\n                suspended_baseline()')
    replace("state='bounded_control_completed'", "state='geometry_failure_replay_preserved'")
    replace("status('bounded_control_completed', step", "status('geometry_failure_replay_preserved', step")
    replace("stop_reason='Preregistered1000-step eight-image roster comparison'", "stop_reason='Failed900-step configuration replayed only to persist diagnostic snapshot; no continuation authorized by this replay'")
    ns=dict(eight.source()[0].__globals__);ns.update(OUT=OUT,ORIGINAL=eight.OUT,SNAPSHOTS=[900])
    exec(compile(s,'<failure-capture-only-same-eight-frame-configuration>','exec'),ns)
    return ns['train'],s


def prepare():
    assert not OUT.exists(),'At most one diagnostic replay.'
    assert read(eight.OUT/'gaussian-run/geometry-progress.json')['step']==900
    p=copy.deepcopy(read(eight.OUT/'plan.json'))
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    OUT.mkdir()
    for name in ['fit/fitted.npz','fit/surface-cache.npz','data/training-inputs.json','smplify-gaussian-interface-receipt.json','selection.json','selection-rule.json']:
        import shutil
        dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(eight.OUT/name,dest)
    _,s=source();ast.parse(s);(OUT/'capture-adapter.py').write_text(s)
    p.update(updated=now(),state='prepared',purpose='Single diagnostic replay because the original guard exited before the first snapshot. Not another candidate or a relaxed guard.',
        original_failed_run=str(eight.OUT),comparison_budget=900,snapshots=[900],auto_continue=False,quality_accepted=False)
    p['source_hashes'].update({str(Path(__file__).resolve()):sha(__file__)})
    save(OUT/'plan.json',p);save(OUT/'progress.json',dict(updated=now(),state='prepared',pid=None,quality_accepted=False))
    print('FAILURE_CAPTURE_PREPARED',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','train']);ap.add_argument('--pid',type=int);args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='run':eight.direct.control.OUT=OUT;eight.direct.control.__file__=__file__;eight.direct.control.run(args.pid)
        else:source()[0]()
    except BaseException as e:
        if OUT.exists():save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
