"""CPU check of actual local-capacity checkpoints and optimization records."""
from pathlib import Path
import json
import math
import torch
from resume_street_deformable import equal_tree,sha

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-street-local-capacity-020'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
FIELDS=('_features_dc','_features_rest','_means','_scales','_quats','_opacities')

def describe(v):
    v=v.double().flatten()
    return dict(min=float(v.min()),median=float(v.median()),p95=float(torch.quantile(v,.95)),max=float(v.max()))

def main():
    torch.set_num_threads(3)
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'fit_street_local_capacity.py')==protocol['script_sha256']
    assert sha(SOURCE/'checkpoint-refined.pt')==protocol['source_checkpoint_sha256']
    original=torch.load(SOURCE/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
    initialization=json.loads((LAB/'outputs/dynamic-human-street-smpl-joint-011/articulated/initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    source=original['models']['SMPLNodes'];mask=source['points_ids'][:,0]==model_index
    assert mask.sum()==6890 and set(protocol['fields'])==set(FIELDS)
    expected={'single_view':[(129,4)],'adjacent_pair':[(129,4),(131,4)]}
    summaries=[]
    for branch,pairs in expected.items():
        run=OUT/branch;destination=run/'independent-state-audit.json'
        assert (run/'completion.json').exists() and not destination.exists()
        complete=json.loads((run/'completion.json').read_text())
        selection=json.loads((run/'training-selection.json').read_text())
        rois=json.loads((run/'optimized-rois.json').read_text())['regions']
        assert [(r['frame'],r['camera']) for r in rois]==pairs
        assert selection['optimized_regions']==rois and complete['optimized_regions']==rois
        assert not selection['heldout_selection_used']
        assert selection['optimizer_updates']==complete['additional_updates']==1000
        assert selection['region_backward_evaluations']==complete['region_backward_evaluations']==1000*len(rois)
        history=selection['history'];assert len(history)==selection['evaluated_training_states']==1001
        assert [r['iteration'] for r in history]==list(range(1001))
        for row in history:
            assert [(r['frame'],r['camera']) for r in row['training_views']]==pairs
            photo=sum(.8*r['l1']+.2*r['ssim_loss'] for r in row['training_views'])/len(pairs)
            assert abs(photo-row['photo'])<2e-7 and abs(row['photo']+row['prior']-row['total'])<1e-12
            assert all(math.isfinite(row[k]) for k in ('total','photo','prior'))
            if row['iteration']<1000:assert set(row['gradient_norms'])==set(FIELDS)
        best=min(history,key=lambda r:r['total'])
        assert best['iteration']==selection['selected_iteration']==complete['selected_iteration']
        assert best['total']==selection['selected_total']
        current=torch.load(run/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
        changed={};frozen=0
        for name,state in original['models'].items():
            actual=current['models'][name];assert state.keys()==actual.keys()
            for field,value in state.items():
                if name=='SMPLNodes' and field in FIELDS:
                    assert value.shape==actual[field].shape and value.dtype==actual[field].dtype
                    assert torch.equal(value[~mask],actual[field][~mask]) and torch.isfinite(actual[field]).all()
                    changed[field]=describe((actual[field][mask]-value[mask]).abs())
                else:
                    equal_tree(value,actual[field],name+'.'+field)
                    frozen+=int(isinstance(value,torch.Tensor))
        for key,value in original.items():
            if key!='models':equal_tree(value,current[key],'trainer.'+key)
        actual=current['models']['SMPLNodes']
        center=(actual['_means'][mask]-source['_means'][mask]).norm(dim=-1)
        scale=(actual['_scales'][mask]-source['_scales'][mask]).exp()
        before=torch.nn.functional.normalize(source['_quats'][mask].double(),dim=-1)
        after=torch.nn.functional.normalize(actual['_quats'][mask].double(),dim=-1)
        rotation=(before*after).sum(-1).abs().clamp(max=1).acos()*360/math.pi
        rotation[torch.all(source['_quats'][mask]==actual['_quats'][mask],dim=-1)]=0
        assert center.max()<=.030001 and scale.min()>=.49999 and scale.max()<=2.00001 and rotation.max()<=20.01
        report=dict(status='actual_state_training_scope_budget_selection_and_bounds_verified',branch=branch,
            source_checkpoint_sha256=sha(SOURCE/'checkpoint-refined.pt'),checkpoint_sha256=sha(run/'checkpoint-refined.pt'),
            optimized_regions=rois,updates=1000,region_backward_evaluations=1000*len(rois),
            selected_iteration=best['iteration'],frozen_tensor_entries=frozen,
            all_poses_voxel_background_other_people_unchanged=True,changes=changed,
            center_displacement_m=describe(center),axis_scale_ratio=describe(scale),rotation_change_degrees=describe(rotation),
            quality_accepted=False,boundary='One/two training-view fitting diagnostic; even perfect training RGB fit would not prove accurate3D or new-view reconstruction.')
        destination.write_text(json.dumps(report,indent=2)+'\n')
        summaries.append({k:report[k] for k in ('branch','selected_iteration','frozen_tensor_entries','region_backward_evaluations')})
        del current
    print(json.dumps(dict(status='both_state_audits_complete',branches=summaries),indent=2))

if __name__=='__main__':main()
