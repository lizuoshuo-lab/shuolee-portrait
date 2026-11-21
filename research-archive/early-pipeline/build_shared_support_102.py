"""Create a finite3Dsupport preflight, with protocol before any measurement."""
from pathlib import Path
import ast,json,datetime,hashlib
L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');P=L/'repos/PGSR-shuolee'
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee';O=B/'current/shared-finite-support-preflight-102'
assert not O.exists();O.mkdir()
source=P/'gaussian_renderer/surface.py';dest=P/'gaussian_renderer/shared_support.py';assert not dest.exists()
t=source.read_text(encoding='utf-8')
t=t.replace('return_depth_normal = True, moment_origin=None):','return_depth_normal = True, moment_origin=None, support_sigma=None):')
t=t.replace('    from surface_concentration import plane_moments','    from shared_face_support import support_moments')
old='    shifted_distance = local_distance + (local_normal * origin_camera).sum(-1)\n    input_all_map[:, 5:] = plane_moments(local_normal, shifted_distance)'
new='''    # Positive full-rank finite Gaussian metric in WORLD coordinates.
    # The metric is a detached local preconditioner; extra gradients cannot
    # enlarge covariance directly just to evade the landmark constraint.
    assert support_sigma is not None and float(support_sigma) > 0
    input_all_map[:, 5:] = support_moments(pc, moment_origin, support_sigma)'''
assert old in t;t=t.replace(old,new)
t=t.replace('"plane_moments": out_all_map[5:15]','"support_moments": out_all_map[5:15]')
t=t.replace('"moment_origin_camera": origin_camera','"moment_origin_camera": origin_camera,\n                    "moment_origin_world": moment_origin')
ast.parse(t);dest.write_text(t,encoding='utf-8',newline='\n')
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),round='102',
 question='Can each view constrain finite Gaussian support around the same Xk without the rank-one point-plane nullspace or center-depth tilt bias?',
 evidence='098point-plane permits distinct tangent-plane positions.099native contributions show dispersed centers and ultra-thin Gaussians.100center-depth candidate rejected because tilted-plane bias exceeds target uncertainty.101jointcameraBA converges with explicit tolerances but has virtually unchanged residuals.',
 representation='One shared static clothed person;50globally shared named facial XYZ from175registered full-image cameras; no per-view targets or deformation.',
 metric='M_j = R_j diag(1/(s_j^2 + sigma0^2)) R_j^T, computed in Gaussian eigenbasis with sigma0=median of fixed image-derived target uncertainties. M is detached during auxiliary-loss differentiation. Gaussian centers and native visibility/alpha weights remain differentiable.',
 loss='E_native_alpha[(mu_j-Xk)^T M_j(mu_j-Xk)] * sigma0^2/sigma_k^2, then the same pseudo-Huber2(sqrt(1+r2)-1). Accumulate10quadratic world-coordinate moments before testing the same uniqueXk; squared terms cannot cancel opposing layers.',
 uncertainty='Fixed from097image-derived covariance, held residuals and detector disagreement. World units uncalibrated. sigma0floor stabilizes original scales as small as1e-15 without editing learned model scales.',
 slanted_surface='The metric rotates with the Gaussian covariance. A target offset within a true tangent plane produces no direct normal-center gradient. Tangential gradients intentionally localize support and may change local sampling: do not claim unchanged geometry or solved quality.',
 single_variable='Replace098rank-one plane residual with this full-rank finite-support residual; fixed plane/RGB/normal/NCC/mask losses, sources, cameras, seed6060, weight.002, startstep1.',
 tests=['positive full-rank metric for ultra-thin and tilted Gaussians','rotation equivariance and zero direct normal gradient on valid tangent support','separated layers and old plane nullspace yield positive restoring gradient','finite-difference agreement with detached metric','native RGB/geometry/old-gradient parity on early499 and mature18k models','native144query weighted Mahalanobis equals10moment computation','all fixed visible observations receive finite support and position gradient'],
 gates=dict(forward_abs=2e-6,old_gradient_abs=2e-5,old_gradient_relative=1e-3,direct_native_relative=2e-4,min_numerical_residual=-2e-4),
 next_training='Only after numerical admission and fresh-init image review: one fresh200k/18k fullbody controlled run. Refit all targetXYZ and head/body initialization from permitted original measurements. No copied learned Gaussians or person fit. No automatic weight sweep.',
 evaluation='Compare against098and093control: shared-point spread, independent whole-body texture correspondences, same-photo face/body render, novel-view appearance, standard+Y allSH/covariance PLY native and actualOpenFile checks. Numericpreflight does not prove reconstruction quality.',
 optimizer_updates=0,training_started=False,quality_accepted=False,previous_turn_classification='progress',
 sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()})
(O/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(protocol=str(O/'protocol.json'),renderer=str(dest),training_started=False)))
