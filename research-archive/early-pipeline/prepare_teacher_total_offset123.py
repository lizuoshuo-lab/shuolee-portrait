"""Create an isolated, reference-correct control without editing prior runs."""
from pathlib import Path
import ast
import difflib


def main():
    folder = Path(__file__).parent
    source = folder / 'fit_gaussian_teacher_visible123.py'
    destination = folder / 'fit_gaussian_teacher_total_offset123.py'
    assert not destination.exists()
    old = source.read_text(encoding='utf-8')
    new = old
    edits = [
        ('One registered control: original foreground alpha contribution replaces opacity proxy.\n\nSame geometry, Laplacian, initialization, optimizer and protection as prior trial.',
         'One reference correction: regularize total offset as original VHAP does.\n\nSame actual contribution weights, geometry, optimizer and guards as the prior trial.'),
        ("OUT=AUDIT/'teacher-visible-surface-100'", "OUT=AUDIT/'teacher-total-offset-100'"),
        ("PREVIOUS=AUDIT/'teacher-surface-laplacian-100'", "PREVIOUS=AUDIT/'teacher-visible-surface-100'\nBASE_AUDIT=AUDIT/'teacher-base-mesh-audit'"),
        ("    OUT.mkdir(exist_ok=False)\n", "    assert io.read(BASE_AUDIT/'result.json')['state']=='source_mesh_and_regularizer_reference_audited'\n    OUT.mkdir(exist_ok=False)\n    reference_contract()\n"),
        ("NATIVE/'native-contribution-weights.npz',NATIVE/'result.json']", "NATIVE/'native-contribution-weights.npz',NATIVE/'result.json',BASE_AUDIT/'result.json',\n        OUT/'reference-contract.json']"),
        ('original uniform Laplacian-coordinate delta squared, summed XYZ and averaged all10475 vertices',
         'original uniform Laplacian of total canonical(initial_offset+delta), squared, summed XYZ and averaged all10475 vertices'),
        ("changed_factor='Only replace opacity proxy by measured foreground T*alpha contribution. Same cache geometry, Gaussian targets/binding, zero delta, Adam, LR,1e6Laplacian,59frames, masks, stopping and admission. Use raw contribution and recompute the original support so the CPU support rounding difference is not imported.'",
         "changed_factor='Only change regularizer reference from additional delta to initial_offset+additional delta. Original VHAP regularizes total offset. Same actual contribution weights, cache geometry, targets, zero additional delta, Adam, LR,1e6 coefficient,59frames,masks and all existing guards.'"),
        ("unit_audit='The strong original1e6Laplacian is deliberately unchanged to isolate visibility weighting. It may still overconstrain; no coefficient sweep or relaxing the existing geometry gate.'",
         "unit_audit='Original1e6 coefficient is unchanged. Total offset has nonzero initial energy, so this may straighten old folds or shrink learned shape; existing gates remain unchanged, including the conservative initial-normal gate. This is a reference correction, not a coefficient sweep.'"),
        ("no_alpha_or_RGB_training_term=True,no_SH_change=True,actual_foreground_alpha_contribution_weights=True",
         "no_alpha_or_RGB_training_term=True,no_SH_change=True,total_offset_reference_corrected=True,actual_foreground_alpha_contribution_weights=True"),
        ("    assert np.array_equal(lap_cache['faces'],cache['faces'])\n", "    assert np.array_equal(lap_cache['faces'],cache['faces'])\n    initial_offset=torch.tensor(cache['initial_offset'],device='cuda')\n"),
        ("lap_delta=torch.sparse.mm(L,delta*learn[:,None])", "lap_delta=torch.sparse.mm(L,initial_offset+delta*learn[:,None])"),
        ("SMPL-X_高斯实际贡献引导_{completed}步_", "SMPL-X_总位移平滑对照_{completed}步_"),
        ("regularizer_source='Exact original PyTorch3D uniform Laplacian and VHAP coordinate-change squared loss;1e6default. Shared additional static-delta use is a local adapter.'",
         "regularizer_source='Exact original VHAP total-offset reference and PyTorch3D uniform Laplacian;1e6default. Shared fixed-skeleton surface adapter remains local, not full official GaussianAvatars static-offset reproduction.'"),
        ("TEACHER_VISIBLE_CONTROL_REGISTERED", "TEACHER_TOTAL_OFFSET_CONTROL_REGISTERED"),
        ("TEACHER_VISIBLE_CONTROL_FINISHED", "TEACHER_TOTAL_OFFSET_CONTROL_FINISHED"),
    ]
    for before, after in edits:
        assert new.count(before) == 1, before
        new = new.replace(before, after)
    contract = '''
def reference_contract():
    # Execute the existing VHAP method itself as the loss oracle, on CPU.
    import torch
    from types import SimpleNamespace
    from probe_upstream_surface_laplacian123 import original_method
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    with np.load(CONTRACT/'frozen-lbs-cache.npz',allow_pickle=False) as z:
        base=torch.tensor(z['base_vertices'][0]);initial=torch.tensor(z['initial_offset'])
    with np.load(LAP/'original-laplacian.npz',allow_pickle=False) as z:
        L=torch.sparse_coo_tensor(torch.tensor(z['indices']),torch.tensor(z['values']),tuple(z['shape'])).coalesce()
    with np.load(PREVIOUS/'candidate-diagnostic-only.npz',allow_pickle=False) as z:
        learn=torch.tensor(z['learnable_vertices'])
    d=torch.zeros_like(initial,requires_grad=True)
    offset=initial+d*learn[:,None]
    original=original_method(VHAP/'vhap/model/tracker.py','compute_laplacian_smoothing_loss')
    proxy=SimpleNamespace(flame=SimpleNamespace(laplacian_matrix=L.to_dense()))
    upstream=original(proxy,base[None],(base+offset)[None]).mean()
    sparse=torch.sparse.mm(L,offset).square().sum(-1).mean()
    gu=torch.autograd.grad(upstream,d,retain_graph=True)[0]
    gs=torch.autograd.grad(sparse,d)[0]
    relative=float((gu-gs).norm()/gu.norm().clamp_min(1e-20))
    assert relative<1e-3 and abs(float(upstream-sparse))<1e-9
    assert not torch.any(gs[~learn]) and torch.isfinite(gs).all()
    L64=L.double();dd=torch.zeros_like(initial,dtype=torch.float64,requires_grad=True)
    fn=lambda x:torch.sparse.mm(L64,initial.double()+x*learn[:,None]).square().sum(-1).mean()
    grad=torch.autograd.grad(fn(dd),dd)[0]
    records=[]
    for index in torch.topk(grad.abs().flatten(),3).indices:
        i,j=divmod(int(index),3);plus=dd.detach().clone();minus=plus.clone()
        plus[i,j]+=1e-7;minus[i,j]-=1e-7
        numeric=float((fn(plus)-fn(minus))/2e-7);analytic=float(grad[i,j])
        assert abs(numeric-analytic)<1e-12
        records.append(dict(vertex=i,coordinate=j,numeric=numeric,analytic=analytic))
    assert not torch.cuda.is_initialized()
    io.save(OUT/'reference-contract.json',dict(state='passed_original_total_offset_reference',
        upstream_loss=float(upstream),sparse_loss=float(sparse),gradient_relative_L2=relative,
        finite_differences=records,protected_gradient_zero=True,cuda_initialized=False,
        original_method_executed=True,optimizer_updates=0,
        scope='VHAP total static/dynamic offset reference. Original GA FLAME method applies its own dynamic-offset term; this is not evidence that original GA automatically regularizes this shared static offset.'))

'''
    new = new.replace('\ndef prepare():\n', '\n' + contract + '\ndef prepare():\n')
    ast.parse(new)
    destination.write_text(new, encoding='utf-8', newline='\n')
    diff = ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=source.name,tofile=destination.name))
    (folder/'teacher-total-offset-reference.diff').write_text(diff,encoding='utf-8',newline='\n')
    print('PREPARED',destination,'changed lines',len(diff.splitlines()))


if __name__ == '__main__':
    main()
