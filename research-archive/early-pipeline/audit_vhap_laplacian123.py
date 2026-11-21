"""CPU comparison of upstream FLAME and SMPL-X fork smoothing contracts."""
import os,inspect,time,json
from pathlib import Path
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops.laplacian_matrices import laplacian
from vhap_demo_adapter123 import ROOT,REPO
from run_official_gaussians123 import read,save,sha,now

RUN=ROOT/'smplify-vhap-offset-init'
OUT=RUN/'regional-gradient-audit'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (OUT/'operator-result.json').exists()
    start=time.monotonic();torch.set_num_threads(4)
    saved=RUN/'epoch-30/tracked-smplx-params.npz'
    refs=[REPO/'vhap/model/flame.py',REPO/'vhap/model/tracker.py',REPO/'vhap/model/tracker_smplx.py',
        Path(inspect.getsourcefile(Meshes)),Path(inspect.getsourcefile(laplacian)),saved]
    save(OUT/'operator-plan.json',dict(created=now(),device='cpu',optimizer_updates=0,
        hypothesis='SMPL-X fork combinatorial D-A differs from original FLAME PyTorch3D normalized A/degree-I, magnifying the squared penalty under the unchanged coefficient.',
        input_hashes={str(p):sha(p) for p in refs},source_sha256=sha(__file__),max_seconds=60,
        stop='One operator comparison on the unchanged mesh and offset; no fitting or coefficient search.'))
    with np.load(saved,allow_pickle=False) as z:
        v=torch.tensor(z['vertices_camera'][0],dtype=torch.float32)
        faces=torch.tensor(z['faces'],dtype=torch.long)
        offset=torch.tensor(z['static_offset'][0],dtype=torch.float32,requires_grad=True)
    # Execute the exact public Meshes method used by flame.py, on this topology.
    mesh=Meshes(verts=[v],faces=[faces]);normalized=mesh.laplacian_packed().coalesce()
    edges=mesh.edges_packed();idx=torch.cat([edges,edges[:,[1,0]]],dim=0).T
    deg=torch.bincount(idx[0],minlength=len(v)).to(v)
    diag=torch.arange(len(v));raw=torch.sparse_coo_tensor(torch.cat([idx,torch.stack([diag,diag])],dim=1),
        torch.cat([-torch.ones(idx.shape[1]),deg]),(len(v),len(v))).coalesce()
    # L_fork = -D L_original; squared loss removes the overall sign.
    converted=torch.sparse_coo_tensor(raw.indices(),-raw.values()/deg[raw.indices()[0]],raw.shape).coalesce()
    difference=(converted-normalized).coalesce()
    max_difference=0 if difference._nnz()==0 else float(difference.values().abs().max())
    assert max_difference<1e-7 and not normalized.is_cuda
    raw_lap=torch.sparse.mm(raw,offset);normal_lap=torch.sparse.mm(normalized,offset)
    raw_rows=raw_lap.square().sum(-1);normal_rows=normal_lap.square().sum(-1)
    raw_loss=1e6*raw_rows.mean();normal_loss=1e6*normal_rows.mean()
    raw_grad,=torch.autograd.grad(raw_loss,offset,retain_graph=True)
    normal_grad,=torch.autograd.grad(normal_loss,offset)
    model=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
    with np.load(model,allow_pickle=False) as z:dominant=np.argmax(z['weights'],axis=-1)
    regions={'torso':[0,3,6,9],'arms':[13,14,16,17,18,19],'legs_and_feet':[1,2,4,5,7,8,10,11]}
    results={}
    for name,ids in regions.items():
        selected=torch.from_numpy(np.isin(dominant,ids))
        results[name]=dict(vertices=int(selected.sum()),median_degree=float(deg[selected].median()),
            same_offset_penalty_ratio=float(raw_rows[selected].sum()/normal_rows[selected].sum()),
            raw_gradient_l2=float(raw_grad[selected].norm()),normalized_gradient_l2=float(normal_grad[selected].norm()))
    source=read(OUT/'result.json')
    actual=np.mean([r['terms']['offset_laplacian']['loss'] for r in source['records']])
    assert abs(float(raw_loss)-actual)<1e-4
    assert time.monotonic()-start<60 and all(sha(p)==h for p,h in read(OUT/'operator-plan.json')['input_hashes'].items())
    result=dict(updated=now(),state='complete_read_only',device='cpu',optimizer_updates=0,source_checkpoint_unchanged=True,
        normalized_reference='Actual pytorch3d.structures.Meshes.laplacian_packed, as used by VHAP flame.py',
        fork_operator='D-A',original_operator='A/degree-I',normalized_equivalence_max_abs=max_difference,
        raw_loss=float(raw_loss),logged_raw_loss=actual,normalized_loss=float(normal_loss),penalty_ratio=float(raw_loss/normal_loss),
        raw_gradient_l2=float(raw_grad.norm()),normalized_gradient_l2=float(normal_grad.norm()),regions=results,
        source_paths={str(p):sha(p) for p in refs},seconds=time.monotonic()-start,quality_accepted=False,
        interpretation='Numeric coefficient 1e6 is unchanged but operator normalization differs. This is an inherited SMPL-X fork semantic difference, not proof that normalization alone yields clothing quality.',
        risk='Restoring normalization loosens smoothing substantially; bumps, noisy offsets, mask leakage or hand/face regressions require a complete frozen-input comparison before adoption.')
    save(OUT/'operator-result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
