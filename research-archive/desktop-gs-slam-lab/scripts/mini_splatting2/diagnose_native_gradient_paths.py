"""Read-only native gradient path isolation; detached compensation is diagnostic only."""
import hashlib
import json
from pathlib import Path
import time
import validate_native_head_region as fixture
import native_mip_filter as mip

torch=fixture.torch;ROOT=fixture.ROOT


def main():
    out=ROOT/'native-gradient-path-diagnosis-001';out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model=ROOT/'native-resolution-cache-full-18000-s0/model';raw=model/'point_cloud/iteration_18000/point_cloud.ply'
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    modes=['rgb','depth','joint','rgb_detached_2d_compensation','joint_detached_2d_compensation']
    protocol=dict(created=time.time(),modes=modes,repeats=5,training_camera_index=21,resolution=[1514,2694],
        model_sha256=digest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_optimizer_steps=True,no_head_weighting=True,
        intervention='Only two diagnostic modes detach derivatives of2D determinant compensation; keep forward formula unchanged.',
        acceptance='Diagnostic only. No modified gradient path may be used for training without separate derivation and validation.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    p=fixture.argparse.ArgumentParser();mp=fixture.ModelParams(p);pp=fixture.PipelineParams(p)
    args=p.parse_args(['-s',str(ROOT/'source-resolution-rectified-002/dataset'),'-m',str(model),'-r','1','--eval','--data_device','cpu'])
    pc=fixture.GaussianModel(3);scene=fixture.Scene(mp.extract(args),pc,load_iteration=18000,shuffle=False)
    pc._mip_cameras=scene.getTrainCameras();pc._mip_iteration=18000
    camera=scene.getTrainCameras()[21];gt=camera.original_image.cuda();valid=fixture.trainer.motion_valid(camera,gt)
    prior=fixture.DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),out,.003)
    parameters=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
    names=['xyz','features_dc','features_rest','opacity','scaling','rotation']
    background=torch.zeros(3,device='cuda');culled=torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool)
    original=mip.compensate_2d
    def detached(means,scales,quats,opacity,cam,scaling_modifier=1.):
        coefficient=original(means,scales,quats,torch.ones_like(opacity),cam,scaling_modifier).detach()
        return opacity*coefficient
    def compare(a,b):
        return {n:dict(max_absolute=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-10)))
            for n,x,y in zip(names,a,b)}
    results={}
    try:
        for mode in modes:
            mip.compensate_2d=detached if 'detached' in mode else original
            base=None;first=None;records=[];worst_value=-1;worst_indices=None
            for index in range(5):
                for parameter in parameters:parameter.grad=None
                image=None;rgb=pc.get_xyz.new_zeros(());depth=pc.get_xyz.new_zeros(())
                if mode!='depth':
                    package=fixture.render_imp(camera,pc,pp.extract(args),background,culling=culled)
                    image=package['render'];image.retain_grad()
                    l1=((image-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
                    rgb=.8*l1+.2*(1-fixture.fused_ssim((image*valid+gt*(1-valid))[None],gt[None]))
                if mode=='depth' or mode.startswith('joint'):depth=prior(camera,pc,18001,valid,culled)
                loss=rgb+depth;loss.backward();torch.cuda.synchronize()
                gradients=[torch.zeros_like(x).cpu() if x.grad is None else x.grad.detach().cpu().clone() for x in parameters]
                assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
                metadata=dict(index=index,loss=float(loss),rgb=float(rgb),depth=float(depth))
                if image is not None:
                    metadata.update(image_sha256=hashlib.sha256(image.detach().cpu().numpy().tobytes()).hexdigest(),
                        image_gradient_sha256=hashlib.sha256(image.grad.detach().cpu().numpy().tobytes()).hexdigest())
                if base is None:base=gradients;first=metadata
                else:
                    error=compare(base,gradients);metadata['gradient_error']=error
                    if error['scaling']['relative_l2']>worst_value:
                        worst_value=error['scaling']['relative_l2']
                        worst_indices=(base[4]-gradients[4]).norm(dim=-1).topk(50).indices.tolist()
                records.append(metadata)
                del gradients
            envelope={n:{k:max(r['gradient_error'][n][k] for r in records[1:]) for k in ['max_absolute','relative_l2']} for n in names}
            with torch.no_grad():
                indices=torch.tensor(worst_indices,device='cuda')
                scales,opacity=mip.filtered_3d(pc.get_scaling,pc.get_opacity,mip.refresh_filter(pc))
                cov=mip.projected_covariance(pc.get_xyz[indices],scales[indices],pc.get_rotation[indices],camera)
                eigen=torch.linalg.eigvalsh(cov.double()).cpu()
                covariance_probe=dict(indices=worst_indices,eigenvalues=eigen.tolist(),
                    condition=(eigen[:,1]/eigen[:,0].clamp_min(1e-20)).tolist(),
                    opacity=opacity[indices,0].cpu().tolist(),scales=scales[indices].cpu().tolist())
            results[mode]=dict(records=records,envelope=envelope,top_scale_error_gaussians=covariance_probe)
            (out/'partial.json').write_text(json.dumps(dict(status='running',protocol=protocol,results=results),indent=2))
            print(json.dumps(dict(mode=mode,scale=envelope['scaling'],rotation=envelope['rotation'])),flush=True)
            del base
    finally:mip.compensate_2d=original
    assert digest==hashlib.sha256(raw.read_bytes()).hexdigest()
    summary=dict(status='complete_pending_interpretation',finished=time.time(),protocol=protocol,results=results,
        model_unchanged=True,training_modified=False,detached_gradient_approved_for_training=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
