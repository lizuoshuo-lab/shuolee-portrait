"""Create a new, fully inspectable profiler namespace; preserve the pilot."""
from pathlib import Path
p=Path(__file__).with_name('profile_city_render_stages_001.py')
s=p.read_text(encoding='utf-8')
s=s.replace("city-render-stages-021'","city-render-stages-022'")
s=s.replace('INDICES=(0,58,116)','INDICES=tuple(range(117))')
s=s.replace("First, middle, last of the 117 frozen heldout views; fixed before timing.","All 117 frozen heldout views; no timing-based selection.")
s=s.replace('warmups_per_case=3,repetitions_per_case=10','warmups_per_case=1,repetitions_per_case=3')
s=s.replace('Selection/weights are replayed outside timing.','Selection/weights recomputed with unchanged official kernels and synchronized separately. Transfers, audits and saves excluded.')
s=s.replace('3 diagnostic views, no aggregate 117-view speed or streaming claim.','117 paired views; repeated within this session, not independent devices. No streaming claim.')
s=s.replace("cases[(setting,j)]=(ids.copy(),pids.copy(),weights,kids,camera,np.load(ref))","# Validate all inputs, but do not retain 117 copies of leaf arrays in RAM.\n            del camera")
marker="    import torch\n    sys.path.insert"
replacement="""    def get_case(setting,j):
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
    sys.path.insert"""
assert marker in s;s=s.replace(marker,replacement)
s=s.replace("max(len(c[0]) for c in cases.values())","max(b['selected_tree_G'] for b in bindings)")
s=s.replace("weights_gpu=torch.zeros(N,device='cuda');kids_gpu=torch.ones(N,device='cuda',dtype=torch.int32)","""weights_gpu=torch.zeros(N,device='cuda');kids_gpu=torch.ones(N,device='cuda',dtype=torch.int32)
    cut_ids=torch.empty(N,device='cuda',dtype=torch.int32);cut_parents=torch.empty_like(cut_ids);cut_nodes=torch.empty_like(cut_ids)
    viewdir=torch.zeros(3,dtype=torch.float32)""")
s=s.replace('ids,pids,w,k,c,reference=cases[(setting,j)]','ids,pids,w,k,c,reference=get_case(setting,j)')
marker="        weights_gpu[:len(w)].copy_(torch.from_numpy(w));kids_gpu[:len(k)].copy_(torch.from_numpy(k))"
replacement="""        torch.cuda.synchronize();start_cut=time.perf_counter()
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
            assert np.array_equal(kids_gpu[:len(k)].cpu().numpy(),k)"""
assert marker in s;s=s.replace(marker,replacement)
s=s.replace("full_render_stream_ms=full_ms,native_rasterizer_stream_ms=native_ms", "cut_host_ms=cut_ms,weights_host_ms=weight_ms,synchronized_stage_sum_ms=cut_ms+weight_ms+wall,\n            full_render_stream_ms=full_ms,native_rasterizer_stream_ms=native_ms")
s=s.replace("print(f'Profiling camera {j}: {names[j]}',flush=True)","cases.clear()\n            print(f'Profiling camera {j}: {names[j]}',flush=True)")
s=s.replace('for warm in range(3)','for warm in range(1)').replace('for repeat in range(10)','for repeat in range(3)')
s=s.replace("for k in ('full_render_stream_ms'", "for k in ('cut_host_ms','weights_host_ms','synchronized_stage_sum_ms','full_render_stream_ms'")
s=s.replace("limitations=['Three predetermined views only','Archived cut/weights excluded from timing'", "limitations=['117 fixed views; 3 repeats each; one desktop session','Per-frame synchronized stage sums, not continuous throughput'")
s=s.replace('3 diagnostic views','117 fixed views')
s=s.replace('Archived cuts and interpolation weights are replayed exactly.', 'Official cuts and interpolation weights are recomputed and checked against the archived values.')
target=p.with_name('profile_city_render_stages_002.py')
assert not target.exists();target.write_text(s,encoding='utf-8')
compile(s,str(target),'exec')
print(target)
