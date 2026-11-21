"""Real GPU smoke proof of the new fixed-pose garment guide; no Gaussian training."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

import launch_unified_cold_016_006 as cold
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-head-support-20261111')
OUT = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round21/input-audit'
sys.path.insert(0,str(PROJECT/'src'))

from shuolee_portrait import runtime as rt
from shuolee_portrait.clothing_surface_guide import fit_shared_clothing
from shuolee_portrait.training import _frozen_smpl_surface, named_pose_layout
from shuolee_portrait.stable_selection import source_bin_training_order


class Frozen:
    def posed_vertices(self, slot):
        return self.base[slot] + (self.linear[slot] @ self.static_offset[..., None]).squeeze(-1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('subjects',nargs='+')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    rt.configure(cold.ASSETS)
    for subject in args.subjects:
        destination=OUT/f'cuda-clothing-guide-{subject}.json'
        if destination.exists():
            raise FileExistsError(destination)
        root=cold.LAB/f'shuolee-smpl-delivery-round20-{subject}/run'
        frozen=root/'gaussian-run/step-010000/reference-mesh.npz'
        with np.load(root/'fit/fitted.npz',allow_pickle=False) as pack:
            fit=dict(pack)
        with np.load(frozen,allow_pickle=False) as pack:
            initial_offset=pack['static_offset'].copy()
        rows=rt.read(root/'data/training-inputs.json')['records']
        layout=named_pose_layout(fit, rows)
        reference=layout['reference_pose_index']
        slots=layout['pose_indices']
        selection=rt.read(root/'fit/official-fit-result.json')['selection']
        bins,_=source_bin_training_order(rows, selection, seed=0)
        base,linear,clothing,protected,error=_frozen_smpl_surface(fit,rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
        model=Frozen()
        model.base,model.linear=base.cuda(),linear.cuda()
        model.faces=torch.as_tensor(fit['faces'],device='cuda')
        model.clothing=torch.as_tensor(clothing,device='cuda')
        model.static_offset=torch.as_tensor(initial_offset,device='cuda')
        print('Checking shared clothing guide on '+subject+' with '+str(len(rows))+' native views.',flush=True)
        chosen,supported,report=fit_shared_clothing(model,rows,slots,bins,reference)
        assert torch.equal(chosen[~model.clothing],model.static_offset[~model.clothing])
        assert report['inputs']['frames']==[row['frame'] for row in rows]
        report.update(subject=subject, baseline_mesh_sha256=rt.sha(frozen),
                      implementation_sha256=rt.sha(PROJECT/'src/shuolee_portrait/clothing_surface_guide.py'),
                      replay_error_m=error,supported_faces=int(supported.sum()),optimizer_scope='shared garment offsets only')
        rt.save(destination,report)
        print(report['state'],report['support'],str(destination),rt.sha(destination),flush=True)

if __name__=='__main__':
    main()
