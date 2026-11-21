"""Version a bounded import-mesh teacher pilot; preserve previous experiments."""
from pathlib import Path
import ast
import difflib


def main():
    folder=Path(__file__).parent;source=folder/'fit_gaussian_teacher_visible123.py'
    dest=folder/'fit_gaussian_teacher_import123.py';assert not dest.exists()
    old=source.read_text();new=old
    def replace(a,b):
        nonlocal new
        assert new.count(a)==1,a
        new=new.replace(a,b)
    replace('One registered control: original foreground alpha contribution replaces opacity proxy.\n\nSame geometry, Laplacian, initialization, optimizer and protection as prior trial.',
        'One bounded pilot on original VHAP import geometry, with fixed120k teacher.\n\nNo Gaussian training, residual-based teacher filtering or new camera/pose solver.')
    replace("CONTRACT=AUDIT/'frozen-skeleton-contract'", "CONTRACT=AUDIT/'original-import-teacher-contract'")
    replace("OUT=AUDIT/'teacher-visible-surface-100'", "OUT=AUDIT/'teacher-import-surface-1000'")
    replace("PREVIOUS=AUDIT/'teacher-surface-laplacian-100'", "PREVIOUS=AUDIT/'teacher-visible-surface-100'")
    replace("    verify();result=io.read(CONTRACT/'result.json');assert result['frames']==59 and result['skeletal_joints_frozen']\n    assert io.sha(CONTRACT/'frozen-lbs-cache.npz')==result['cache_sha256']",
        "    verify();result=io.read(CONTRACT/'result.json');assert result['frames']==59 and result['skeletal_joints_frozen']\n    assert result['numerical_contract_passed'] and result['coverage']['all59_nonempty_support']")
    new=new.replace("CONTRACT/'frozen-lbs-cache.npz'", "CONTRACT/'original-import-lbs-cache.npz'")
    replace("NATIVE/'native-contribution-weights.npz',NATIVE/'result.json']", "NATIVE/'native-contribution-weights.npz',NATIVE/'result.json',CONTRACT/'teacher-support-diagnostic-only.npz']")
    replace("dict(created=io.now(),frames=59,max_steps=100,max_seconds=300,", "dict(created=io.now(),frames=59,max_steps=1000,max_seconds=300,")
    replace("stage='Shared clothing-surface distillation pilot from an immutable Gaussian teacher; NOT Gaussian continuation or pose optimization.'",
        "stage='Exploratory shared clothing-surface distillation from original VHAP import and immutable120k teacher; not Gaussian training or a physical-surface acceptance.'")
    replace("changed_factor='Only replace opacity proxy by measured foreground T*alpha contribution. Same cache geometry, Gaussian targets/binding, zero delta, Adam, LR,1e6Laplacian,59frames, masks, stopping and admission. Use raw contribution and recompute the original support so the CPU support rounding difference is not imported.'",
        "changed_factor='Original VHAP import mesh and its original frozen LBS replace the degraded120k mesh. Same immutable120k target positions and Gaussian IDs; inside-bound-triangle support and footpoints are consistently recomputed on this reference. This coupled initialization/association change is explicit. Same Adam,LR,1e6 additional-delta Laplacian,59frames and geometric gates.1000step ceiling, not a claim of matched-step causal quality improvement.'")
    replace("unit_audit='The strong original1e6Laplacian is deliberately unchanged to isolate visibility weighting. It may still overconstrain; no coefficient sweep or relaxing the existing geometry gate.'",
        "unit_audit='1e6 additional-delta Laplacian preserves the imported learned clothing shape. The failed total-offset shrinkage control is not imported. No coefficient/LR sweep.1000step ceiling allows slow coherent motion; all20step guards and2cm displacement stop remain.'")
    replace("        trainable_initialization='zero additional shared canonical10475x3 displacement; empty optimizer',",
        "        teacher_uncertainty='Only2604/6921clothing faces supported; canonical descriptor dispersion median1.588mm. This admits a guarded research pilot only, not physical correspondence or production training.',\n        trainable_initialization='originalVHAP import plus zero additional shared canonical10475x3 displacement; empty optimizer',")
    replace("    with np.load(NATIVE/'native-contribution-weights.npz',allow_pickle=False) as z:contribution_cache=dict(z)\n    contribution_slots={str(x):i for i,x in enumerate(contribution_cache['frame_ids'])}",
        "    with np.load(CONTRACT/'teacher-support-diagnostic-only.npz',allow_pickle=False) as z:teacher=dict(z)")
    begin=new.index('    model=SMPLXGaussianModel(3);model.load_ply')
    end=new.index('    order=torch.tensor([slots[r[\'frame\']] for r in rows],device=\'cuda\')',begin)
    new=new[:begin]+'''    assert np.array_equal(teacher['frame_ids'],[r['frame'] for r in rows])
    binding=torch.tensor(teacher['binding'],device='cuda',dtype=torch.long)
    targets=torch.tensor(teacher['targets'],device='cuda')
    barys=torch.tensor(teacher['barycentric_per_frame'],device='cuda')
    weights=torch.tensor(teacher['weights'],device='cuda')
    contract_result=io.read(CONTRACT/'result.json')
    support_records=[dict(frame=r['frame'],points=r['points'],triangles=r['faces']) for r in contract_result['frames_result']]
    normal_support=[dict(frame=r['frame'],positive_normal_weight=r['positive_normal_weight'],
        caveat='Original-import mesh normal sign; not anatomical outward or physical clothing truth.') for r in contract_result['frames_result']]
    assert torch.allclose(weights.sum(1),torch.ones(59,device='cuda'),atol=1e-6)
    assert torch.isfinite(targets).all() and torch.isfinite(barys).all()
''' +new[end:]
    replace('tri_vid=faces[model.binding]', 'tri_vid=faces[binding]')
    replace("            'Same positive depth/native mask/original radius visibility admission; detached native foreground T*alpha contribution replaces detached opacity weight.',",
        "            'Same positive depth/native center mask and measured foreground T*alpha visibility, recomputed inside-bound-triangle support on original import.',")
    replace("regularizer_source='Exact original PyTorch3D uniform Laplacian and VHAP coordinate-change squared loss;1e6default. Shared additional static-delta use is a local adapter.'",
        "regularizer_source='Original PyTorch3D uniform Laplacian applied to new shared delta relative to frozen VHAP import,1e6weight; this is a local fine-tuning prior, not a claim of original VHAP total-offset semantics.'")
    replace('SMPL-X_高斯实际贡献引导_{completed}步_', 'SMPL-X_原VHAP网格高斯引导_{completed}步_')
    replace('TEACHER_VISIBLE_CONTROL_REGISTERED','IMPORT_TEACHER_PILOT_REGISTERED')
    replace('TEACHER_VISIBLE_CONTROL_FINISHED','IMPORT_TEACHER_PILOT_FINISHED')
    ast.parse(new);dest.write_text(new,encoding='utf-8',newline='\n')
    diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=source.name,tofile=dest.name))
    (folder/'teacher-import-surface-pilot.diff').write_text(diff,encoding='utf-8',newline='\n')
    print('PREPARED_IMPORT_PILOT',dest,flush=True)


if __name__=='__main__':main()
