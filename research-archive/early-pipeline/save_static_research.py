from pathlib import Path
import json,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-surface-comparison-067')
data=dict(checked=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    new_weights_downloaded=0,new_environments_installed=0,
    evidence_boundary='These are code/documentation checks, not successful experiments on this person.',
    candidates=[
        dict(name='Gaussian Opacity Fields',source='https://github.com/autonomousvision/gaussian-opacity-fields',
            renderer='https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/gaussian_renderer/__init__.py',
            license='https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/LICENSE.md',
            mechanism='Extract an opacity level set from 3D Gaussians with surface regularization and adaptive tetrahedral meshing.',
            relevance='An alternative for explicit surface inspection, rather than relying only on point thickness.',
            compatibility_risk='Official render uses 3D-filtered scale/opacity, kernel size and view-to-Gaussian transformations. A file with familiar PLY fields does not establish parity with the existing viewer.',
            decision='Candidate only. Require zero-update native-vs-standard rendering audit and licence review before adoption. Not a solution to unobserved rear/sole evidence by itself.'),
        dict(name='DN-Splatter / AGS-Mesh',source='https://github.com/maturk/dn-splatter',
            paper='https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf',
            mechanism='Depth and normal priors with local smoothness; AGS-Mesh adds depth/normal filtering and surface extraction.',
            relevance='External geometric priors can add information beyond current self-consistency losses.',
            prerequisites='Reliable depth scale alignment; monocular-normal coordinate convention; confidence around hair, glasses, cloth folds and occlusions.',
            decision='Inspect existing priors first. Current model depth is not independent ground truth; do not install another stack or treat predicted/filled back views as measurements.')],
    observed_065='Removing min-scale compression substantially changes axis ratios, but rendered shoe holes and below-view ghosting remain. Sole-cause min-scale explanation not supported by this comparison.',
    pending_066='Fresh multiview-loss ablation still training; no conclusion yet.',
    next_decision='Finish controlled static comparison, then diagnose view coverage and conflicting observations before adopting another geometry prior.')
(B/'open-source-analysis.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(B/'open-source-analysis.json')
