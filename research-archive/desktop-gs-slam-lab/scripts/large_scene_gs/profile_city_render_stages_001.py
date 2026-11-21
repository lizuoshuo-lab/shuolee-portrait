"""Read-only, fixed-input stage profiling of the official City post renderer.

Archived cuts and interpolation weights are replayed exactly. CUDA events
surround the unchanged rasterizer call; no camera, model or kernel is changed.
"""
from pathlib import Path
from types import SimpleNamespace
import json, sys, time, traceback
import numpy as np
import evaluate_official_hierarchy as core

LAB=core.ROOT
OLD=core.TRACK/'independent-evaluation/small-city006-heldout-1404-001'
OUT=core.TRACK/'research/city-render-stages-021'
SAVE=core.save
INDICES=(0,58,116)
SETTINGS=('leaves','tau3')

def main():
    OUT.mkdir(exist_ok=False)
    SAVE(OUT/'plan.json',dict(status='registered_before_measurement',camera_indices=INDICES,
        selection_rule='First, middle, last of the 117 frozen heldout views; fixed before timing.',
        settings=SETTINGS,warmups_per_case=3,repetitions_per_case=10,
        schedule='Interleave leaves/tau3 each repetition; reverse order on odd repetitions.',
        question='Does native rasterization or surrounding tensor work dominate render_post?',
        timing='CUDA stream events for full render_post and its rasterizer call; synchronized host wall time separately. Selection/weights are replayed outside timing.',
        limits='3 diagnostic views, no aggregate 117-view speed or streaming claim. Event span includes stream idle/dispatch gaps, not exclusive summed GPU kernel time.',
        no_training=True,source_script_sha256=core.digest(Path(__file__))))
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    core.source_audit(OUT)
    protocol=core.read_json(OLD/'protocol.json')
    inp=Path(protocol['independent_input_npz_path'])
    assert core.digest(inp)==protocol['independent_input_npz_sha256']
    inputs=np.load(inp);names=inputs['names'];H,W=inputs['rgb'].shape[1:3]
    desc=next(d for d in protocol['models'] if d['name']=='merged-post')
    assert core.digest(Path(desc['path']))==desc['source_model_sha256']
    s=core.read_json(OLD/'shared-skybox-audit.json')
    sky,_=core.skybox_PLY_CPU(Path(s['ply_path']),s['count'],OUT,s['source_metadata'])
    assert core.digest(Path(s['ply_path']))==s['ply_sha256']
    print('Source and input checks passed; loading CPU model.',flush=True)
    prepared=core.model_cpu(desc,{},sky,OUT)
    root,h,topology,gh,tensors,leaves,audit=prepared
    cases={};bindings=[]
    for setting in SETTINGS:
        rows=core.read_json(OLD/'merged-post'/setting/'per-frame-audit.json')
        for j in INDICES:
            row=rows[j];assert row['name']==names[j]
            sel=Path(row['selected']['path']);assert core.digest(sel)==row['selected']['file_sha256']
            selected=np.load(sel)
            ids=selected['indices'];pids=selected['parent_gaussian_indices']
            stem=f'{j:02d}-{Path(str(names[j])).stem}'
            weight=OLD/'merged-post'/setting/'selections'/f'{stem}-interpolation.npz'
            with np.load(weight) as a:weights=a['child_weights'].copy();kids=a['num_parent_children'].copy()
            assert core.array_digest(weights)==row['interpolation_weight_bytes_sha256']
            assert core.array_digest(kids)==row['sibling_count_bytes_sha256']
            assert len(ids)==len(weights)==len(kids)
            assert row['coverage']['every_leaf_exactly_once'] and row['weight_vs_CPU_reference_match']
            ref=OLD/'merged-post'/setting/'float-frames'/f'{stem}.npy'
            camera=core.make_official_camera_CPU(inputs['K'][j],inputs['c2w'][j],W,H,str(names[j]))
            cases[(setting,j)]=(ids.copy(),pids.copy(),weights,kids,camera,np.load(ref))
            bindings.append(dict(setting=setting,index=j,name=str(names[j]),selected_tree_G=len(ids),
                selection_sha256=core.digest(sel),interpolation_sha256=core.digest(weight),reference_rgb_sha256=core.digest(ref)))
    SAVE(OUT/'input-bindings.json',dict(input_archive_sha256=core.digest(inp),model_sha256=audit['source_sha256'],cases=bindings))
    import torch
    sys.path.insert(0,str(core.REPO));sys.path.insert(0,str(core.EXT))
    from scene.gaussian_model import GaussianModel
    import gaussian_renderer as renderer
    import diff_gaussian_rasterization as native
    assert Path(renderer.__file__).resolve()==(core.REPO/'gaussian_renderer/__init__.py').resolve()
    assert Path(native._C.__file__).resolve().is_relative_to(core.EXT.resolve())
    free,total=torch.cuda.mem_get_info()
    plan=core.memory_plan(audit['total_G'],len(h.nodes),len(leaves),max(len(c[0]) for c in cases.values()),audit['background_G'],3)
    plan.update(free_bytes_before_load=free,total_device_bytes=total)
    plan['accepted']=plan['conservative_Python_tensor_peak_upper_bytes']+1024**3<free
    SAVE(OUT/'memory-preflight.json',plan)
    assert plan['accepted'],'Not enough free memory for known tensor bound plus native workspace headroom'
    pc,gpu=core.cached_model_GPU(GaussianModel,tensors,audit['background_G'],root)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');N=len(pc._xyz)
    weights_gpu=torch.zeros(N,device='cuda');kids_gpu=torch.ones(N,device='cuda',dtype=torch.int32)
    # Keep only one case's gather inputs resident; model remains wholly resident.
    original=renderer.GaussianRasterizer
    native_events=[]
    class ObservedRasterizer:
        def __init__(self,*args,**kwargs):self.delegate=original(*args,**kwargs)
        def __call__(self,*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();result=self.delegate(*args,**kwargs);end.record()
            native_events.append((start,end))
            return result
    renderer.GaussianRasterizer=ObservedRasterizer
    all_rows=[];checks=[]
    @torch.no_grad()
    def measure(setting,j,repetition):
        ids,pids,w,k,c,reference=cases[(setting,j)]
        fields=vars(c).copy()
        for key in ('world_view_transform','projection_matrix','full_proj_transform','camera_center'):fields[key]=fields[key].cuda()
        view=SimpleNamespace(**fields);ii=torch.from_numpy(ids).cuda();pp=torch.from_numpy(pids).cuda()
        weights_gpu[:len(w)].copy_(torch.from_numpy(w));kids_gpu[:len(k)].copy_(torch.from_numpy(k))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();native_events.clear()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        began=time.perf_counter();a.record()
        result=renderer.render_post(view,pc,pipe,bg,render_indices=ii,parent_indices=pp,
            interpolation_weights=weights_gpu,num_node_kids=kids_gpu,interp_python=True,use_trained_exp=False)
        b.record();torch.cuda.synchronize();wall=(time.perf_counter()-began)*1000
        assert len(native_events)==1
        native_ms=native_events[0][0].elapsed_time(native_events[0][1]);full_ms=a.elapsed_time(b)
        image=result['render'].permute(1,2,0).cpu().numpy()
        exact=bool(np.array_equal(image,reference));diff=float(np.max(np.abs(image-reference)))
        assert np.isfinite(image).all() and exact,f'Output mismatch: {setting}, {j}, {diff}'
        row=dict(setting=setting,index=j,name=str(names[j]),repeat=repetition,selected_tree_G=len(ids),
            full_render_stream_ms=full_ms,native_rasterizer_stream_ms=native_ms,surrounding_stream_ms=full_ms-native_ms,
            synchronized_host_ms=wall,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            rgb_bitwise_equal=exact,max_abs_rgb_difference=diff,visible_G=len(result['radii']))
        if repetition==0:
            r=result['radii'].cpu().numpy()
            row['visible_projected_radius_px_quantiles']=np.quantile(r,[.5,.9,.99,1]).tolist()
        return row
    try:
        for j in INDICES:
            print(f'Profiling camera {j}: {names[j]}',flush=True)
            for setting in SETTINGS:
                for warm in range(3):checks.append(measure(setting,j,-warm-1))
            for repeat in range(10):
                for setting in SETTINGS if repeat%2==0 else SETTINGS[::-1]:
                    all_rows.append(measure(setting,j,repeat))
                SAVE(OUT/'measurements.json',all_rows)
    finally:renderer.GaussianRasterizer=original
    SAVE(OUT/'warmups.json',checks)
    summaries=[]
    for j in INDICES:
        for setting in SETTINGS:
            rows=[r for r in all_rows if r['index']==j and r['setting']==setting]
            summary={k:rows[0][k] for k in ('setting','index','name','selected_tree_G','visible_G','visible_projected_radius_px_quantiles')}
            for k in ('full_render_stream_ms','native_rasterizer_stream_ms','surrounding_stream_ms','synchronized_host_ms'):
                summary[k]={'median':float(np.median([r[k] for r in rows])),'min':min(r[k] for r in rows),'max':max(r[k] for r in rows)}
            summaries.append(summary)
    assert core.digest(Path(desc['path']))==audit['source_sha256']
    SAVE(OUT/'metrics.json',dict(status='completed_exact_RGB_stage_profile',measured_renders=len(all_rows),warmup_renders=len(checks),all_RGB_bitwise_equal=True,
        source_model_sha256=audit['source_sha256'],camera_indices=INDICES,summaries=summaries,
        limitations=['Three predetermined views only','Archived cut/weights excluded from timing','CUDA event spans include dispatch/idle gaps','No geometry or alpha modifications','Whole tree resident; no streaming']))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
