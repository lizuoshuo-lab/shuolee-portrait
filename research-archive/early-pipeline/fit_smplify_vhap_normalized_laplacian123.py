"""Restore the original VHAP/FLAME smoothing operator on the SMPL-X mesh."""
import inspect
import torch
from pytorch3d.structures import Meshes
import fit_smplify_vhap_mask123 as mask
from run_official_gaussians123 import read,save,sha,now

mask.joint.OUT=mask.joint.ROOT/'smplify-vhap-normalized-laplacian'
mask.joint.PLAN=mask.joint.ROOT/'smplify-vhap-normalized-laplacian-plan.json'


def make_tracker(out):
    t,rows=mask.make_tracker(out)
    raw=t.smplx_laplacian.coalesce()
    # Call the same upstream method used by vhap/model/flame.py.
    normalized=Meshes(verts=[t.smplx.v_template],faces=[t.smplx.faces_tensor.long()]).laplacian_packed().coalesce().detach()
    idx=raw.indices();isdiag=idx[0]==idx[1];degree=torch.zeros(raw.shape[0],device=raw.device)
    degree[idx[0,isdiag]]=raw.values()[isdiag]
    assert torch.all(degree>0)
    check=torch.sparse_coo_tensor(idx,-raw.values()/degree[idx[0]],raw.shape,device=raw.device).coalesce()
    diff=(check-normalized).coalesce();error=float(diff.values().abs().max()) if diff._nnz() else 0
    assert error<1e-7 and t.cfg.w.reg_offset_lap==1e6 and t.cfg.w.reg_offset==300
    t.smplx_laplacian=normalized
    save(out/'operator-contract.json',dict(updated=now(),source='pytorch3d.structures.Meshes.laplacian_packed',
        source_sha256=sha(inspect.getsourcefile(Meshes)),original='D-A',candidate='A/degree-I',
        equivalence_max_abs=error,coefficient_unchanged=1e6,L1_coefficient_unchanged=300,
        optimizer_updates=0,quality_accepted=False))
    return t,rows


def main():
    plan=read(mask.joint.PLAN)
    assert plan['restore_original_offset_initialization'] and plan['operator']=='original_vhap_normalized_laplacian'
    mask.joint.make_tracker=make_tracker;mask.joint.main()
    for name in ('result.json','progress.json'):
        d=read(mask.joint.OUT/name)
        d.update(updated=now(),original_optimizer_loss_scheduler=False,original_optimizer_and_scheduler=True,
            original_energy_terms_preserved=True,added_loss='photo30 * sum(abs(pred_alpha-target_alpha)) / sum(target_alpha)',
            restored_stage='rgb_init_offset500 at scale1 + global59x30 at scale0.1',
            restored_operator='Exact original VHAP/FLAME PyTorch3D normalized Laplacian on the same SMPL-X topology',
            scope='Single operator change relative to offset-init. Same weights, input, mask, initial geometry and optimizer schedule.')
        save(mask.joint.OUT/name,d)


if __name__=='__main__':main()
