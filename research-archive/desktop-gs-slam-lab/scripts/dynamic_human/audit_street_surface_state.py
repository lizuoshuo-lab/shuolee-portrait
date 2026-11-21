"""Independent CPU verification of target-only surface/appearance interventions."""
from pathlib import Path
import argparse
import json
import math
import torch
from resume_street_deformable import equal_tree,sha

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-street-surface-refit-018'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
COLOR=('_features_dc','_features_rest')
SURFACE=('_means','_scales','_quats','_opacities')


def stats(values):
    values=values.float().flatten()
    return dict(min=float(values.min()),median=float(values.median()),p95=float(torch.quantile(values,.95)),max=float(values.max()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=['appearance_only','surface_and_appearance']);args=parser.parse_args()
    torch.set_num_threads(3)
    protocol=json.loads((OUT/'protocol.json').read_text())
    checkpoint=SOURCE/'checkpoint-refined.pt'
    assert sha(checkpoint)==protocol['source_checkpoint_sha256']
    source=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    initial=json.loads((LAB/'outputs/dynamic-human-street-smpl-joint-011/articulated/initialization.json').read_text())
    index=next(r['model_index'] for r in initial['instances']['SMPLNodes'] if r['author_instance']==79)
    before=source['models']['SMPLNodes'];mask=before['points_ids'][:,0]==index
    assert int(mask.sum())==6890
    for branch in [args.branch] if args.branch else ['appearance_only','surface_and_appearance']:
        run=OUT/branch if branch=='appearance_only' else OUT/'surface-retry-v2'/branch
        if branch=='surface_and_appearance':
            repaired=json.loads((OUT/'surface-retry-v2/protocol.json').read_text())
            for key in ('source_checkpoint_sha256','target','optimizer','parameterization','objective','selection','frozen','evaluation','renderer_mode'):
                assert protocol[key]==repaired[key],key
            assert repaired['branches']['surface_and_appearance']==list(COLOR+SURFACE)
        assert (run/'completion.json').exists()
        assert not (run/'independent-state-audit.json').exists(),'Preserve completed audit'
        complete=json.loads((run/'completion.json').read_text())
        selection=json.loads((run/'training-selection.json').read_text())
        assert complete['additional_fullbatch_updates']==selection['optimizer_updates']==100
        assert complete['region_backward_evaluations']==selection['region_backward_evaluations']==10100
        assert selection['evaluated_training_states']==101 and not selection['heldout_selection_used']
        best=min(selection['history'],key=lambda r:r['total'])
        assert best['iteration']==selection['selected_iteration'] and abs(best['total']-selection['selected_total'])<1e-12
        final_path=run/'checkpoint-refined.pt'
        actual=torch.load(final_path,map_location='cpu',mmap=True,weights_only=False)
        allowed=COLOR if branch=='appearance_only' else COLOR+SURFACE
        count=0;changes={}
        for name,state in source['models'].items():
            other=actual['models'][name];assert state.keys()==other.keys()
            for field,value in state.items():
                if name=='SMPLNodes' and field in allowed:
                    assert value.shape==other[field].shape and value.dtype==other[field].dtype
                    assert torch.equal(value[~mask],other[field][~mask]),field
                    assert torch.isfinite(other[field]).all()
                    changes[field+'_absolute_component_change']=stats((other[field][mask]-value[mask]).abs())
                else:equal_tree(value,other[field],name+'.'+field);count+=int(isinstance(value,torch.Tensor))
        for key,value in source.items():
            if key!='models':equal_tree(value,actual[key],'trainer.'+key)
        after=actual['models']['SMPLNodes']
        displacement=(after['_means'][mask]-before['_means'][mask]).norm(dim=-1)
        ratios=(after['_scales'][mask]-before['_scales'][mask]).exp()
        a=torch.nn.functional.normalize(before['_quats'][mask].double(),dim=-1)
        b=torch.nn.functional.normalize(after['_quats'][mask].double(),dim=-1)
        degrees=torch.acos((a*b).sum(-1).abs().clamp(max=1))*360/math.pi
        degrees[torch.all(before['_quats'][mask]==after['_quats'][mask],dim=-1)]=0
        assert displacement.max()<=.030001 and ratios.min()>=.49999 and ratios.max()<=2.00001
        assert degrees.max()<=20.01
        report=dict(status='independent_state_budget_training_selection_and_physical_bounds_passed',branch=branch,
            source_checkpoint_sha256=sha(checkpoint),refined_checkpoint_sha256=sha(final_path),target_points=6890,
            allowed_fields=list(allowed),frozen_tensor_entries=count,all_poses_voxel_background_and_other_people_exactly_unchanged=True,
            updates=100,region_backward_evaluations=10100,selected_iteration=selection['selected_iteration'],
            changes=changes,center_displacement_m=stats(displacement),axis_scale_ratio=stats(ratios),rotation_change_degrees=stats(degrees),
            quality_accepted=False,boundaries=['Parameter changes are relative to source016, not ground-truth surface errors.',
                'No target test-time attributes were independently fitted; the same canonical attributes apply across time.'])
        (run/'independent-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
