"""Quantify actual frozen-path CUDA repeat differences, not bitwise claims."""
from pathlib import Path
import json,torch
torch.set_num_threads(8)
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-support-030'
source=LAB/'outputs/dynamic-human-enerf-local-time-029/offset-index/motion'
checks=[]
for frame in [34,35]:
    old=torch.load(source/f'frame-{frame:03d}.pt',weights_only=False,map_location='cpu')
    new=torch.load(out/'frozen-reproduction/motion'/f'frame-{frame:03d}.pt',weights_only=False,map_location='cpu')
    entries={}
    for k in old:
        assert old[k].shape==new[k].shape and torch.isfinite(new[k]).all()
        diff=(old[k].double()-new[k].double()).abs();entries[k]=dict(exact=torch.equal(old[k],new[k]),max_abs=float(diff.max()),mean_abs=float(diff.mean()))
        if frame==34 or k in ['scales','opacities']:assert entries[k]['exact']
    checks.append(dict(common_frame=frame,tensors=entries))
result=dict(status='identical_initial_state_and_frozen_parameters_small_repeat_differences_recorded',checks=checks,decision='Run a fresh complete frozen control with the same current trainer, not substitute old model results. CUDA repeated gradients are not claimed bitwise deterministic.')
(out/'frozen-reproduction-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
