"""Exact loss-interface invariants on native renderings, with gradient noise reported.

This does not relabel the earlier parameter-gradient equivalence failures. The
head weighting module changes RGB L1 only. Its neutral loss and dL/dRGB must be
bit identical; its active dL1/dRGB must match the analytic weighted L1 derivative.
Native Gaussian gradients are exercised and reported without a bitwise claim.
"""
import hashlib
import json
from pathlib import Path
import time
import validate_native_head_region as fixture

torch=fixture.torch;ROOT=fixture.ROOT


def main():
    out=ROOT/'native-head-interface-validation-003';out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model=ROOT/'native-resolution-cache-full-18000-s0/model';raw=model/'point_cloud/iteration_18000/point_cloud.ply'
    before=hashlib.sha256(raw.read_bytes()).hexdigest()
    protocol=dict(created=time.time(),status='frozen_before_execution',fixtures=[[2,0],[1,0],[1,21],[1,49]],
        neutral_invariant='Loss and dL/dRGB bit identical on the same native rendering and joint RGB+depth graph',
        active_invariant='d weighted L1/dRGB equals sign(I-G)*valid*(1+H)/(3*sum(valid*(1+H)))',
        derivative_tolerance=dict(rtol=2e-6,atol=2e-13),head_coefficient=2.,ssim_weight=.2,depth_weight=.003,
        rationale='Previous strict Gaussian-gradient gates also failed for unchanged native repeats. Here the changed loss interface is tested exactly; native parameter variability remains explicitly unresolved.',
        preserved_failures=['native-head-gradient-validation-001','native-head-repeatability-validation-002'],
        native_noise_audit='native-gradient-path-diagnosis-001/interpretation.json',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=before,
        no_kernel_or_training_change=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    p=fixture.argparse.ArgumentParser();mp=fixture.ModelParams(p);pp=fixture.PipelineParams(p)
    args=p.parse_args(['-s',str(ROOT/'source-resolution-rectified-002/dataset'),'-m',str(model),'-r','1','--eval','--data_device','cpu'])
    pc=fixture.GaussianModel(3);scene=fixture.Scene(mp.extract(args),pc,load_iteration=18000,shuffle=False,resolution_scales=[1,2])
    pc._mip_cameras=scene.getTrainCameras();pc._mip_iteration=18000
    prior=fixture.DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),out,.003)
    neutral=fixture.TrainingHeadL1(ROOT/'training-head-regions-003',1.)
    weighted=fixture.TrainingHeadL1(ROOT/'training-head-regions-003',2.)
    annotation=json.loads((ROOT/'training-head-regions-003/manifest.json').read_text())
    boxes={row['frame']:row['box'] for row in annotation['rows']}
    parameters=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
    names=['xyz','features_dc','features_rest','opacity','scaling','rotation','viewspace']
    background=torch.zeros(3,device='cuda');culled=torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool);rows=[]
    for scale,index in protocol['fixtures']:
        camera=scene.getTrainCameras(scale)[index];assert int(camera.image_name)%8!=0
        gt=camera.original_image.cuda();valid=fixture.trainer.motion_valid(camera,gt)
        package=fixture.render_imp(camera,pc,pp.extract(args),background,culling=culled);image=package['render']
        l1=((image-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
        n1=neutral(camera.image_name,image,gt,valid);w1=weighted(camera.image_name,image,gt,valid)
        ssim=1-fixture.fused_ssim((image*valid+gt*(1-valid))[None],gt[None])
        depth=prior(camera,pc,18001,valid,culled)
        losses=[.8*x+.2*ssim+depth for x in [l1,n1,w1]]
        gi=[torch.autograd.grad(loss,image,retain_graph=True)[0] for loss in losses]
        exact=bool(torch.equal(losses[0],losses[1]) and torch.equal(gi[0],gi[1]))
        weights=valid.detach().clone()
        x0,y0,x1,y1=[round(v*s) for v,s in zip(boxes[camera.image_name],
            [camera.image_width/757,camera.image_height/1347,camera.image_width/757,camera.image_height/1347])]
        weights[:,y0:y1,x0:x1]*=2
        analytic=torch.sign(image.detach()-gt)*weights/(3*weights.sum().clamp_min(1))
        active=torch.autograd.grad(w1,image,retain_graph=True)[0]
        analytic_ok=bool(torch.allclose(active,analytic,rtol=2e-6,atol=2e-13))
        invalid_zero=bool(torch.count_nonzero(active*(1-valid))==0)
        gradient_sets=[torch.autograd.grad(loss,parameters+[package['viewspace_points']],retain_graph=True) for loss in losses]
        finite=all(torch.isfinite(g).all() for group in gradient_sets for g in group)
        variation={name:dict(max_absolute=float((a-b).abs().max()),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-10)))
            for name,a,b in zip(names,gradient_sets[0],gradient_sets[1])}
        changed=any(float((a-b).norm())>0 for a,b in zip(gradient_sets[0],gradient_sets[2]))
        passed=exact and analytic_ok and invalid_zero and bool(finite) and changed
        row=dict(frame=camera.image_name,size=[camera.image_width,camera.image_height],neutral_loss_and_image_gradient_bit_identical=exact,
            active_image_gradient_matches_formula=analytic_ok,max_analytic_error=float((active-analytic).abs().max()),
            invalid_pixels_zero_gradient=invalid_zero,native_gradients_finite=bool(finite),active_parameter_gradients_changed=changed,
            neutral_native_gradient_variation=variation,interface_passed=passed)
        rows.append(row);(out/'partial.json').write_text(json.dumps(dict(status='running' if passed else 'failed',rows=rows),indent=2))
        print(json.dumps(dict(frame=camera.image_name,size=row['size'],interface_passed=passed)),flush=True)
        assert passed,'Do not integrate if the changed loss interface fails'
        del package,image,l1,n1,w1,ssim,depth,losses,gi,gradient_sets,gt,valid,weights,active,analytic
        torch.cuda.empty_cache()
    assert before==hashlib.sha256(raw.read_bytes()).hexdigest()
    result=dict(status='interface_validated',created=time.time(),protocol=protocol,rows=rows,model_unchanged=True,
        original_parameter_equivalence_gate_passed=False,exact_gaussian_gradient_equivalence=False,
        native_noise_not_fixed=True,head_weighting_integrated=False,quality_improvement_proven=False,
        authorized_next_scope='Isolated smoke with original kernels/backward/topology schedule; full quality comparison and repetitions still required.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],fixtures=len(rows))),flush=True)


if __name__=='__main__':main()
