"""CPU algebra/reference checks for a future spatial-weighting ablation only."""
import hashlib
import json
from pathlib import Path
import time
import torch
from head_region_l1 import TrainingHeadL1

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    torch.set_num_threads(2);torch.manual_seed(2001)
    out=ROOT/'head-attention-preparation-001';out.mkdir(exist_ok=False)
    path=ROOT/'training-head-regions-003';control=TrainingHeadL1(path,1.);weighted=TrainingHeadL1(path,2.)
    rows=[]
    for width,height in [(379,674),(757,1347),(1514,2694)]:
        for name in ['000001','000014','000058']:
            source=torch.rand((3,height,width));reference=torch.rand_like(source)
            valid=(torch.rand((1,height,width))>.1).float()
            old=source.clone().requires_grad_();new=source.clone().requires_grad_()
            a=((old-reference).abs()*valid).sum()/(3*valid.sum().clamp_min(1));b=control(name,new,reference,valid)
            a.backward();b.backward();assert torch.equal(a,b) and torch.equal(old.grad,new.grad)
            im=source.clone().requires_grad_();loss=weighted(name,im,reference,valid);loss.backward()
            mask=weighted.region(name,height,width,im.device,im.dtype)
            accepted=valid.expand_as(im)>0
            inside=(mask.expand_as(im)>0)&accepted;outside=(mask.expand_as(im)==0)&accepted
            inside_value=im.grad[inside].abs().mean();outside_value=im.grad[outside].abs().mean()
            ratio=float(inside_value/outside_value);assert abs(ratio-2)<1e-5
            assert torch.all(im.grad[~accepted]==0)
            assert torch.isfinite(loss) and torch.isfinite(im.grad).all()
            rows.append(dict(frame=name,size=[width,height],neutral_loss_and_pixel_gradient_bit_identical=True,
                weighted_inside_outside_gradient_ratio=ratio,invalid_pixels_zero_gradient=True,
                image_fraction=float(mask.mean()),gpu_used=False))
            del old,new,im,source,reference,valid
    for name in weighted.boxes:
        weighted.region(name,674,379,'cpu',torch.float32);assert len(weighted.cache)<=4
    excluded=False
    try:weighted.region('000008',674,379,'cpu',torch.float32)
    except AssertionError:excluded=True
    assert excluded
    empty=torch.zeros((1,20,20));x=torch.ones((3,20,20),requires_grad=True)
    loss=weighted('000001',x,torch.zeros_like(x),empty);loss.backward();assert float(loss)==0 and torch.all(x.grad==0)
    protocol=dict(status='cpu_validated_not_integrated_or_trained',created=time.time(),rows=rows,heldout_rejected=excluded,
        all51cache_bound_checked=True,zero_valid_pixels_finite_and_zero_gradient=True,
        candidate_weight=2.,neutral_weight=1.,
        scientific_question='If resolution alone leaves head detail poor, does allocating more RGB-L1 weight to training head regions improve real heldout detail?',
        not_used_by_current_resolution_experiment=True,not_approved_as_improvement=True,
        pending=['Actual native Gaussian parameter-gradient validation','Isolated trainer integration','Decide after resolution comparison and repeats'],
        constraint='Keep depth, SSIM, camera poses, reference split and joint backward unchanged; weighting may alter native densification via RGB gradients',
        annotation_scope='Training-only manually reviewed rectangles; include neck/background; not precise masks',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('head_region_l1.py'),path/'manifest.json',path/'visual-review.json']})
    (out/'validation.json').write_text(json.dumps(protocol,indent=2));print(json.dumps({k:v for k,v in protocol.items() if k not in ['rows','source_sha256']}))


if __name__=='__main__':main()
