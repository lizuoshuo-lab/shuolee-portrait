"""Separate native baseline noise from the neutral head-loss path.

The first strict test is retained as failed: unchanged native repeats themselves
exceeded its relative-gradient tolerance. This diagnostic freezes a baseline-
calibrated envelope before neutral samples, while requiring identical RGB and
image-gradient hashes. It never updates model parameters.
"""
import hashlib
import json
import time
from pathlib import Path
import validate_native_head_region as fixture

torch=fixture.torch;ROOT=fixture.ROOT


def main():
    for name in ['native-late-queue.json','native-late-publication-queue.json','native-training-fit-queue.json']:
        assert json.loads((ROOT/name).read_text())['status'].startswith('complete')
    out=ROOT/'native-head-repeatability-validation-002';out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model=ROOT/'native-resolution-cache-full-18000-s0/model'
    raw=model/'point_cloud/iteration_18000/point_cloud.ply'
    before=hashlib.sha256(raw.read_bytes()).hexdigest()
    protocol=dict(created=time.time(),status='frozen_before_execution',prior='native-head-gradient-validation-001/partial.json',
        original_repeats=5,neutral_repeats=3,candidate_repeats=1,fixtures=[[2,0],[1,0],[1,21],[1,49]],
        rgb_and_rgb_input_gradient='bit identical between original and neutral',
        neutral_parameter_gate='per group: max(2e-4,4*baseline relative envelope), max(1e-5,4*baseline absolute envelope)',
        baseline_sanity_gate='all groups: relative L2<.02 and absolute<1e-4',
        interpretation='No exact Gaussian-gradient equivalence claim; empirical native repeatability gate plus exact neutral image-gradient invariant',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture_sha256=before,training_updated=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    p=fixture.argparse.ArgumentParser();mp=fixture.ModelParams(p);pp=fixture.PipelineParams(p)
    args=p.parse_args(['-s',str(ROOT/'source-resolution-rectified-002/dataset'),'-m',str(model),'-r','1','--eval','--data_device','cpu'])
    pc=fixture.GaussianModel(3);scene=fixture.Scene(mp.extract(args),pc,load_iteration=18000,shuffle=False,resolution_scales=[1,2])
    pc._mip_cameras=scene.getTrainCameras();pc._mip_iteration=18000
    prior=fixture.DepthGuidance(ROOT/'depth-priors',scene.getTrainCameras(),out,.003)
    heads={k:fixture.TrainingHeadL1(ROOT/'training-head-regions-003',v) for k,v in [('neutral',1.),('weighted',2.)]}
    parameters=[pc._xyz,pc._features_dc,pc._features_rest,pc._opacity,pc._scaling,pc._rotation]
    names=['xyz','features_dc','features_rest','opacity','scaling','rotation','viewspace']
    background=torch.zeros(3,device='cuda');culled=torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool)
    rows=[]
    def errors(a,b):
        return {name:dict(max_absolute=float((x-y).abs().max()),relative_l2=float((x-y).norm()/x.norm().clamp_min(1e-10)))
            for name,x,y in zip(names,a,b)}
    for scale,index in protocol['fixtures']:
        camera=scene.getTrainCameras(scale)[index]
        gt=camera.original_image.cuda();valid=fixture.trainer.motion_valid(camera,gt)
        def calculate(mode):
            for parameter in parameters:parameter.grad=None
            package=fixture.render_imp(camera,pc,pp.extract(args),background,culling=culled)
            image=package['render'];image.retain_grad()
            if mode=='original':l1=((image-gt).abs()*valid).sum()/(3*valid.sum().clamp_min(1))
            else:l1=heads[mode](camera.image_name,image,gt,valid)
            rgb=.8*l1+.2*(1-fixture.fused_ssim((image*valid+gt*(1-valid))[None],gt[None]))
            depth=prior(camera,pc,18001,valid,culled)
            loss=rgb+depth;loss.backward();torch.cuda.synchronize()
            gradients=[x.grad.detach().cpu().clone() for x in parameters]
            gradients.append(package['viewspace_points'].grad.detach().cpu().clone())
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
            meta=dict(loss=float(loss),l1=float(l1),depth=float(depth),
                image_sha256=hashlib.sha256(image.detach().cpu().numpy().tobytes()).hexdigest(),
                image_gradient_sha256=hashlib.sha256(image.grad.detach().cpu().numpy().tobytes()).hexdigest())
            return meta,gradients
        reference,base=calculate('original');baseline=[]
        for _ in range(protocol['original_repeats']-1):
            metadata,g=calculate('original');baseline.append(dict(metadata=metadata,error=errors(base,g)));del g
        envelope={name:{key:max(row['error'][name][key] for row in baseline) for key in ['max_absolute','relative_l2']} for name in names}
        sane=all(v['relative_l2']<.02 and v['max_absolute']<1e-4 for v in envelope.values())
        # Freeze the envelope before any neutral samples in this fixture.
        (out/f'{scale}-{index}-baseline.json').write_text(json.dumps(dict(reference=reference,repeats=baseline,envelope=envelope,sane=sane),indent=2))
        assert sane,'Baseline backward itself needs investigation before any new training'
        neutral=[]
        for _ in range(protocol['neutral_repeats']):
            metadata,g=calculate('neutral');e=errors(base,g)
            exact=all(metadata[k]==reference[k] for k in ['loss','image_sha256','image_gradient_sha256'])
            equivalent=all(e[n]['relative_l2']<=max(2e-4,4*envelope[n]['relative_l2']) and
                e[n]['max_absolute']<=max(1e-5,4*envelope[n]['max_absolute']) for n in names)
            neutral.append(dict(metadata=metadata,error=e,exact_image_gradient=exact,within_repeatability_envelope=equivalent));del g
        candidate,g=calculate('weighted');change=errors(base,g);del g
        candidate_valid=candidate['image_sha256']==reference['image_sha256'] and candidate['image_gradient_sha256']!=reference['image_gradient_sha256'] and any(e['relative_l2']>.01 for e in change.values())
        passed=all(x['exact_image_gradient'] and x['within_repeatability_envelope'] for x in neutral) and candidate_valid
        rows.append(dict(frame=camera.image_name,size=[camera.image_width,camera.image_height],reference=reference,
            baseline_envelope=envelope,neutral=neutral,candidate=candidate,weighted_gradient_change=change,passed=passed))
        (out/'partial.json').write_text(json.dumps(dict(status='running' if passed else 'failed',rows=rows),indent=2))
        print(json.dumps(dict(frame=camera.image_name,size=rows[-1]['size'],passed=passed)),flush=True)
        assert passed,'Do not integrate after a failed gate'
        del base,gt,valid
    assert before==hashlib.sha256(raw.read_bytes()).hexdigest()
    result=dict(status='passed',finished=time.time(),protocol=protocol,rows=rows,model_unchanged=True,
        integrated_into_trainer=False,quality_improvement_proven=False,exact_gaussian_gradient_equivalence=False,
        neutral_image_gradient_bit_identical=True,neutral_parameter_gradients_within_baseline_repeatability=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status='passed',checks=len(rows))),flush=True)


if __name__=='__main__':main()
