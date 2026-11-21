"""Save exact research sources and input provenance without duplicating checkpoints."""
from pathlib import Path
import json,hashlib,shutil,subprocess
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-pgsr-fresh-064';repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee')
out=R/'source-final';out.mkdir(exist_ok=True)
sourcefiles=['shuolee_static.py','gaussian_renderer/__init__.py','scene/gaussian_model.py','scene/cameras.py',
             'utils/loss_utils.py','arguments/__init__.py','LICENSE.md']
records=[]
for rel in sourcefiles:
    src=repo/rel;dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    records.append(dict(file=rel,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
decision=dict(original_repository='https://github.com/zju3dv/PGSR',commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
    project='https://zju3dv.github.io/pgsr/',paper='https://zju3dv.github.io/pgsr/paper/pgsr.pdf',
    reason='Static surface regularization with planar3D Gaussian depth; RGB retains standard3DGS rendering.',
    alternative=dict(project='https://github.com/hbb1/2d-gaussian-splatting',
        reason_deferred='Oriented2D disks require a different renderer; direct standard3DGS export parity is not guaranteed.'),
    preregistered_comparison='Static062 vs staticPGSR064. All inputs fresh, same200k seed initialization hash; losses, resolution and densification change together, not a single-loss ablation.',
    initialization_sha256=hashlib.sha256((R/'clothing-init/points.npz').read_bytes()).hexdigest(),
    exact_initialization_matches_062=(R/'clothing-init/points.npz').read_bytes()==(B/'current/person-static-world-fresh-062/clothing-init/points.npz').read_bytes(),
    source_files=records,license_scope='Educational/research/nonprofit. PGSR-specific commercial authorization not confirmed; no company deployment.',
    unseen_view_images_are_training_gt=False,geometry_gt_available=False,commercial_acceptance=False)
(out/'manifest.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('sources',len(records),'same fresh initialization',decision['exact_initialization_matches_062'])
