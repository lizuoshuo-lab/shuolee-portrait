"""Read-only, fixed-input stage profiling of the official City post renderer.

Official cuts and interpolation weights are recomputed and checked against the archived values. CUDA events
surround the unchanged rasterizer call; no camera, model or kernel is changed.
"""
from pathlib import Path
from types import SimpleNamespace
import json, sys, time, traceback
import numpy as np
import evaluate_official_hierarchy as core

LAB=core.ROOT
OLD=core.TRACK/'independent-evaluation/small-city006-heldout-1404-001'
OUT=core.TRACK/'research/city-render-stages-022'
SAVE=core.save
INDICES=tuple(range(117))
SETTINGS=('leaves','tau3')

def main():
    OUT.mkdir(exist_ok=False)
    SAVE(OUT/'plan.json',dict(status='registered_before_measurement',camera_indices=INDICES,
        selection_rule='All 117 frozen heldout views; no timing-based selection.',
        settings=SETTINGS,warmups_per_case=1,repetitions_per_case=3,
        schedule='Interleave leaves/tau3 each repetition; reverse order on odd repetitions.',
        question='Does native rasterization or surrounding tensor work dominate render_post?',
        timing='CUDA stream events for full render_post and its rasterizer call; synchronized host wall time separately. Selection/weights recomputed with unchanged official kernels and synchronized separately. Transfers, audits and saves excluded.',
        limits='117 paired views; repeated within this session, not independent devices. No streaming claim. Event span includes stream idle/dispatch gaps, not exclusive summed GPU kernel time.',
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
            # Validate all inputs, but do not retain 117 copies of leaf arrays in RAM.
            del camera
            bindings.append(dict(setting=setting,index=j,name=str(names[j]),selected_tree_G=len(ids),
                selection_sha256=core.digest(sel),interpolation_sha256=core.digest(weight),reference_rgb_sha256=core.digest(ref)))
    SAVE(OUT/'input-bindings.json',dict(input_archive_sha256=core.digest(inp),model_sha256=audit['source_sha256'],cases=bindings))
    def get_case(setting,j):
        key=(setting,j)
        if key not in cases:
            row=core.read_json(OLD/'merged-post'/setting/'per-frame-audit.json')[j]
            selected=np.load(row['selected']['path'])
            stem=f'{j:02d}-{Path(str(names[j])).stem}'
            with np.load(OLD/'merged-post'/setting/'selections'/f'{stem}-interpolation.npz') as a:
                w=a['child_weights'].copy();k=a['num_parent_children'].copy()
            camera=core.make_official_camera_CPU(inputs['K'][j],inputs['c2w'][j],W,H,str(names[j]))
            cases[key]=(selected['indices'],selected['parent_gaussian_indices'],w,k,camera,
                np.load(OLD/'merged-post'/setting/'float-frames'/f'{stem}.npy'))
        return cases[key]
    import torch
    sys.path.insert(0,str(core.REPO));sys.path.insert(0,str(core.EXT))
    from scene.gaussian_model import GaussianModel
    import gaussian_renderer as renderer
    import diff_gaussian_rasterization as native
    assert Path(renderer.__file__).resolve()==(core.REPO/'gaussian_renderer/__init__.py').resolve()
    assert Path(native._C.__file__).resolve().is_relative_to(core.EXT.resolve())
    free,total=torch.cuda.mem_get_info()
    plan=core.memory_plan(audit['total_G'],len(h.nodes),len(leaves),max(b['selected_tree_G'] for b in bindings),audit['background_G'],3)
    plan.update(free_bytes_before_load=free,total_device_bytes=total)
    plan['accepted']=plan['conservative_Python_tensor_peak_upper_bytes']+1024**3<free
    SAVE(OUT/'memory-preflight.json',plan)
    assert plan['accepted'],'Not enough free memory for known tensor bound plus native workspace headroom'
    pc,gpu=core.cached_model_GPU(GaussianModel,tensors,audit['background_G'],root)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');N=len(pc._xyz)
    weights_gpu=torch.zeros(N,device='cuda');kids_gpu=torch.ones(N,device='cuda',dtype=torch.int32)
    cut_ids=torch.empty(N,device='cuda',dtype=torch.int32);cut_parents=torch.empty_like(cut_ids);cut_nodes=torch.empty_like(cut_ids)
    viewdir=torch.zeros(3,dtype=torch.float32)
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
        ids,pids,w,k,c,reference=get_case(setting,j)
        fields=vars(c).copy()
        for key in ('world_view_transform','projection_matrix','full_proj_transform','camera_center'):fields[key]=fields[key].cuda()
        view=SimpleNamespace(**fields);ii=torch.from_numpy(ids).cuda();pp=torch.from_numpy(pids).cuda()
        torch.cuda.synchronize();start_cut=time.perf_counter()
        if setting=='tau3':
            theta=core.tau_to_threshold(3.,W/(2*np.tan(view.FoVx*.5)))
            count=gh.expand_to_size(pc.nodes,pc.boxes,theta,view.camera_center.contiguous(),viewdir,cut_ids,cut_parents,cut_nodes)
            ii=cut_ids[:count];pp=cut_parents[:count]
        torch.cuda.synchronize();cut_ms=(time.perf_counter()-start_cut)*1000 if setting=='tau3' else 0.
        start_weights=time.perf_counter()
        if setting=='tau3':
            gh.get_interpolation_weights(cut_nodes[:count].contiguous(),theta,pc.nodes,pc.boxes,c.camera_center.contiguous(),viewdir,weights_gpu,kids_gpu)
        else:
            weights_gpu[:len(ids)+pc.skybox_points]=1;kids_gpu[:len(ids)+pc.skybox_points]=1
        torch.cuda.synchronize();weight_ms=(time.perf_counter()-start_weights)*1000
        if repetition<=0:
            assert np.array_equal(ii.cpu().numpy(),ids) and np.array_equal(pp.cpu().numpy(),pids)
            assert np.array_equal(weights_gpu[:len(w)].cpu().numpy(),w)
            assert np.array_equal(kids_gpu[:len(k)].cpu().numpy(),k)
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
            cut_host_ms=cut_ms,weights_host_ms=weight_ms,synchronized_stage_sum_ms=cut_ms+weight_ms+wall,
            full_render_stream_ms=full_ms,native_rasterizer_stream_ms=native_ms,surrounding_stream_ms=full_ms-native_ms,
            synchronized_host_ms=wall,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            rgb_bitwise_equal=exact,max_abs_rgb_difference=diff,visible_G=len(result['radii']))
        if repetition==0:
            r=result['radii'].cpu().numpy()
            row['visible_projected_radius_px_quantiles']=np.quantile(r,[.5,.9,.99,1]).tolist()
        return row
    try:
        for j in INDICES:
            cases.clear()
            print(f'Profiling camera {j}: {names[j]}',flush=True)
            for setting in SETTINGS:
                for warm in range(1):checks.append(measure(setting,j,-warm-1))
            for repeat in range(3):
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
            for k in ('cut_host_ms','weights_host_ms','synchronized_stage_sum_ms','full_render_stream_ms','native_rasterizer_stream_ms','surrounding_stream_ms','synchronized_host_ms'):
                summary[k]={'median':float(np.median([r[k] for r in rows])),'min':min(r[k] for r in rows),'max':max(r[k] for r in rows)}
            summaries.append(summary)
    assert core.digest(Path(desc['path']))==audit['source_sha256']
    SAVE(OUT/'metrics.json',dict(status='completed_exact_RGB_stage_profile',measured_renders=len(all_rows),warmup_renders=len(checks),all_RGB_bitwise_equal=True,
        source_model_sha256=audit['source_sha256'],camera_indices=INDICES,summaries=summaries,
        limitations=['117 fixed views; 3 repeats each; one desktop session','Per-frame synchronized stage sums, not continuous throughput','CUDA event spans include dispatch/idle gaps','No geometry or alpha modifications','Whole tree resident; no streaming']))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
