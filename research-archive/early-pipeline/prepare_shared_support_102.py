from pathlib import Path
import ast,json
W=Path(__file__).parent;L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');G=L/'repos/GaussianAvatars_SMPLX-shuolee'
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
check=json.loads((B/'current/shared-finite-support-preflight-102/result.json').read_text());assert check['numerical_preflight_passed']
t=(G/'prepare_shared_face_098.py').read_text()
t=t.replace('person-static-shared-face-fresh-098','person-static-shared-support-fresh-102').replace("round='098'","round='102'")
t=t.replace("current/face-shared-world-points-097/consumer-preflight.json","current/shared-finite-support-preflight-102/result.json")
t=t.replace('098','102')
t=t.replace('External globally triangulated point targets constrain contributing Gaussian planes in every visible view, preventing inconsistent per-view surfaces from being hidden in one blended depth.',
            'Full-rank finite Gaussian support around each unique shared3Dpoint removes the tangent-plane nullspace while retaining covariance-aware local surface shape. Metric covariance is detached in auxiliary gradients; original visibility/alpha remains differentiable.')
t=t.replace('097:50 available shared points from175fixed cameras and5815visible observations; held median5.37px so targets are uncertain, not exact intersections. Corrected renderer/gradient parity and a two-layer non-cancellation test passed.',
            '102preflight: rotation equivariance, valid tangent directnormalgradient, two-layer noncancellation, extreme thin-axis stability, nativeforward/gradientparity and144directcontribution queries pass.100center-depth approximation was rejected.101camera candidate not applied. SharedXYZ remain uncertain, notGT.')
t=t.replace('Add shared-face-world contributor-plane loss; all existing losses, observations, sampler and initialization recipe remain unchanged.',
            'Replace098rank-oneplane auxiliary residual with full-rank finite-support Mahalanobis residual at the same sharedXYZ. All original losses, weights, cameras, observations and fresh initialization recipe stay unchanged.')
t=t.replace('093control full18k and095early1800 are evaluation references only.', '098sharedplane18k primaryreference and093control18k secondaryreference are evaluation only.')
t=t.replace('contributing-plane dispersion','nativecontributor finite3Dsupport and spatial dispersion')
t=t.replace('shared-face-world contributor-plane','shared finite-support')
dest=G/'prepare_shared_support_102.py';assert not dest.exists();ast.parse(t);dest.write_text(t,encoding='utf-8',newline='\n')
sh=(W/'prepare_shared_face_098.sh').read_text().replace('person-static-shared-face-fresh-098','person-static-shared-support-fresh-102').replace('prepare_shared_face_098.py','prepare_shared_support_102.py')
(W/'prepare_shared_support_102.sh').write_text(sh,encoding='utf-8',newline='\n')
ast.parse((L/'repos/PGSR-shuolee/shuolee_static.py').read_text())
print(json.dumps(dict(prepared=True,training_started=False,run='person-static-shared-support-fresh-102')))
