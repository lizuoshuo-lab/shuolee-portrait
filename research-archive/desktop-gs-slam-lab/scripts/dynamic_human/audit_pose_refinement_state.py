"""CPU-only tensor-level verification of the pose-only street intervention."""
from pathlib import Path
import argparse
import json
import math
import torch
from resume_street_deformable import equal_tree,sha

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
SOURCE=LAB/'outputs/dynamic-human-street-smpl-joint-011/articulated'
RUN=LAB/'outputs/dynamic-human-street-pose-refinement-014'
NAMES=('instances_trans','instances_quats','smpl_qauts')


def summary(values):
    values=values.flatten()
    return dict(mean=float(values.mean()),median=float(values.median()),p95=float(torch.quantile(values,.95)),max=float(values.max()))


def main():
    global RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--long',action='store_true')
    args=parser.parse_args()
    if args.batch:RUN=LAB/'outputs/dynamic-human-street-pose-batch-015'
    if args.long:RUN=LAB/'outputs/dynamic-human-street-pose-convergence-016'
    assert (RUN/'completion.json').exists()
    assert not (RUN/'independent-state-audit.json').exists(),'Preserve completed state audit'
    torch.set_num_threads(3)
    protocol=json.loads((RUN/'protocol.json').read_text())
    source_path=SOURCE/'checkpoint-30000.pt'
    assert sha(source_path)==protocol['source_checkpoint_sha256']
    before=torch.load(source_path,map_location='cpu',mmap=True,weights_only=False)
    after=torch.load(RUN/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
    initialization=json.loads((SOURCE/'initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    bs,as_=before['models']['SMPLNodes'],after['models']['SMPLNodes']
    active=bs['instances_fv'][:,model_index].clone()
    active[torch.arange(10,151,10)]=False
    assert int(active.sum())==90
    frozen_tensors=0
    mutable_mask=torch.zeros(bs['instances_trans'].shape[:2],dtype=torch.bool)
    mutable_mask[active,model_index]=True
    for name,state in before['models'].items():
        actual=after['models'][name]
        assert state.keys()==actual.keys()
        for field,value in state.items():
            if name=='SMPLNodes' and field in NAMES:
                assert value.shape==actual[field].shape and value.dtype==actual[field].dtype
                assert torch.equal(value[~mutable_mask],actual[field][~mutable_mask]),field
                assert torch.isfinite(actual[field]).all()
            else:
                equal_tree(value,actual[field],name+'.'+field)
                frozen_tensors+=int(isinstance(value,torch.Tensor))
    for key,value in before.items():
        if key!='models':equal_tree(value,after[key],'trainer.'+key)
    translation=(as_['instances_trans'][active,model_index]-bs['instances_trans'][active,model_index]).norm(dim=-1)
    assert translation.max()<=.50001
    changes=dict(translation_m=summary(translation))
    for name,limit in [('instances_quats',25),('smpl_qauts',45)]:
        a=torch.nn.functional.normalize(as_[name][active,model_index],dim=-1)
        b=torch.nn.functional.normalize(bs[name][active,model_index],dim=-1)
        angle=torch.acos((a*b).sum(-1).abs().clamp(max=1))*360/math.pi
        assert angle.max()<=limit+.002
        changes[name+'_degrees']=summary(angle)
    report=dict(status='independent_checkpoint_comparison_confirms_only_target_training_poses_changed',
        model_index=model_index,person=79,training_times=90,frozen_tensor_entries=frozen_tensors,
        heldout_poses_and_other_people_exactly_unchanged=True,
        source_checkpoint_sha256=sha(source_path),refined_checkpoint_sha256=sha(RUN/'checkpoint-refined.pt'),
        changes=changes,quality_accepted=False,
        boundaries=['Parameter change magnitudes are not ground-truth pose errors.',
            'Additional optimization changes only target-person training-time poses; existing Gaussian shape and appearance remain exactly fixed.',
            'Heldout outputs may change via the unchanged neighboring-pose interpolation.'])
    (RUN/'independent-state-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
