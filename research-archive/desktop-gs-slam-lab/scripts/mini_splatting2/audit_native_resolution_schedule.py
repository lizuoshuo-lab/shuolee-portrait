"""Read-only schedule evidence and a bounded follow-up hypothesis; no training mutation."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
REPO=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001/repo-depth-cache')


def main():
    out=ROOT/'native-resolution-schedule-audit-001';out.mkdir(exist_ok=False)
    config=REPO/'config/fast';tree=ast.parse(config.read_text().strip(),mode='eval').body
    values={item.arg:ast.literal_eval(item.value) for item in tree.keywords}
    assert values['densify_until_iter']==values['warn_until_iter']==values['simp_iteration1']==3000
    files=[REPO/'msv2/train_depth_guided.py',REPO/'scene/__init__.py',REPO/'scene/gaussian_model.py',config]
    excerpts=[]
    needles=['if iteration < opt.densify_until_iter','if iteration == args.simp_iteration1','if not viewpoint_stack:',
             'if iteration<=warn_until_iter','factor=args.sampling_factor','if iteration == args.simp_iteration2']
    for path in files:
        lines=path.read_text().splitlines()
        for index,line in enumerate(lines):
            if any(needle in line for needle in needles):
                excerpts.append(dict(file=str(path),line=index+1,text='\n'.join(lines[max(0,index-1):index+5])))
    result=dict(status='source_verified_hypothesis_unproven',created=time.time(),config=values,excerpts=excerpts,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        findings=['Native Gaussian densification is gated by iteration<3000.',
            'Warm-up camera selection uses half resolution through iteration3000; a prefilled camera stack may delay the switch slightly.',
            'First importance sampling at3000 therefore also uses half-resolution images.',
            'The subsequent full-resolution phase optimizes existing parameters and prunes at8000 without adding detail Gaussians.'],
        interpretation='Verified algorithm schedule, not evidence that it caused the observed blur or that changing it helps.',
        candidate_not_implemented_or_trained=dict(name='Post-warmup gradient densification',
            purpose='Test whether full-resolution observations need a chance to allocate local Gaussian detail after coarse simplification.',
            changes=['Preserve original iterations1..3000 and first simplification.',
                'Allow ordinary gradient clone/split/prune only for3000<iteration<5000; retain original aggressive-clone schedule.',
                'Reset topology-sized blur accumulator after the first simplification; use current camera scale.',
                'Keep8000simplification,18000total steps,Mip,depth0.003,mask,images,cameras and reference split unchanged.'],
            checks_before_training=['Neutral path loss/gradient equivalence','Actual post-simplification topology and optimizer-shape smoke','Memory/time gate through the new window'],
            decision='Consider only after the current low/full pair; do not mix head weighting or diffusion into this comparison.'),
        current_resolution_experiment_changed=False)
    (out/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],config=values,training_changed=False)))


if __name__=='__main__':main()
