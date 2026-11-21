"""Interpret read-only path isolation without promoting a modified backward."""
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    folder=ROOT/'native-gradient-path-diagnosis-001'
    data=json.loads((folder/'summary.json').read_text());assert data['model_unchanged']
    rows={}
    for name,value in data['results'].items():
        rows[name]=dict(scale=value['envelope']['scaling'],rotation=value['envelope']['rotation'],
            unique_rgb_hashes=len({r['image_sha256'] for r in value['records'] if 'image_sha256' in r}),
            unique_image_gradient_hashes=len({r['image_gradient_sha256'] for r in value['records'] if 'image_gradient_sha256' in r}),
            top50_covariance_condition_median=statistics.median(value['top_scale_error_gaussians']['condition']),
            top50_covariance_condition_max=max(value['top_scale_error_gaussians']['condition']))
    result=dict(status='reviewed',created=time.time(),rows=rows,training_modified=False,
        findings=['Unmodified RGB alone already shows about2.76percent relative scale-gradient variation across5repeats, with about7e-6maximum absolute difference.',
            'Joint RGB+depth is about2.66percent; depth alone has much smaller absolute variation.',
            'Detaching2D compensation derivatives leaves about2.45percent RGB variation. This does not isolate that compensation as the cause or justify detaching it during training.',
            'RGB and image-gradient hashes stay identical within every RGB-bearing mode.',
            'Top scale-error Gaussians have elongated projected covariances. This is an association, not proof of a kernel bug or of the cause of human blur.'],
        previous_failed_checks_preserved=['native-head-gradient-validation-001','native-head-repeatability-validation-002'],
        exact_gaussian_gradient_equivalence_established=False,head_weighting_integrated=False,head_weighting_trained=False,
        decision='Keep original kernels and joint backward. Do not modify training to suppress an unproven cause.',
        next=['Finish exact neutral RGB and image-gradient invariants on the remaining native fixtures, reporting parameter variation independently.',
            'Any revised validation must state why unchanged native nondeterminism is treated separately; do not report the failed strict gate as passed.',
            'Only then decide a weight2head-only controlled smoke on the original schedule; retain distance/pitch and full-image guardrails.'])
    target=folder/'interpretation.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],training_modified=False)))


if __name__=='__main__':main()
