"""Prepare one bounded numerical candidate; never start training here."""
from pathlib import Path
import ast, datetime, hashlib, json

L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
P=L/'repos/PGSR-shuolee'
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
O=B/'current/shared-point-position-preflight-100'
assert not O.exists(), 'Do not overwrite a numerical experiment'
O.mkdir()
source=P/'gaussian_renderer/surface.py'
code=source.read_text(encoding='utf-8')
code=code.replace('    from surface_concentration import plane_moments\n','')
old='    shifted_distance = local_distance + (local_normal * origin_camera).sum(-1)\n    input_all_map[:, 5:] = plane_moments(local_normal, shifted_distance)'
new='''    # Native 3DGS uses center depth for ordering. These are center-depth
    # moments, not ray/plane intersections or exact ellipsoid integration.
    # A shifted origin avoids cancellation of large absolute depths.
    shifted_depth = depth_z - origin_camera[2]
    input_all_map[:, 5] = shifted_depth
    input_all_map[:, 6] = shifted_depth.square()'''
assert old in code
code=code.replace(old,new).replace('"plane_moments": out_all_map[5:15]', '"center_depth_moments": out_all_map[5:7]')
code=code.replace('    Render the scene. ', '    Render unchanged native RGB/plane maps plus center-depth moments. ')
target=P/'gaussian_renderer/shared_depth.py'
assert not target.exists();ast.parse(code);target.write_text(code,encoding='utf-8',newline='\n')
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),round='100',
 user_intent='Each named physical facial feature has exactly one global XYZ shared by all visible cameras. Distinct features have distinct points.',
 evidence='099: coordinate transforms reproduce raw-to-undistorted projection within5.2e-12pixels. Point-to-plane loss permits depth separation; native contributor planes may be near-parallel to rays and ultrathin.099jointcameraBA did not converge and is not accepted.',
 single_main_variable='Replace098expected squared contributor-plane residual with expected squared full position residual under the finite center-depth approximation; no camera, initialization, observation, mask, photometric, NCC or geometric-weight change.',
 formula='For observation(i,k), Xk is unique world point. In camera coordinates Yijk=r_ik*z_ij, where z_ij is Gaussian center depth. L_ik=E_native_alpha[||Yijk-(Ri Xk+ti)||^2]. Square before compositing. Use shifted first/second depth moments; no mean-only loss and no ray-plane denominator.',
 approximation='Y is the observed-ray point at each contributor center depth. It is not the Gaussian center itself, exact ellipsoid integration, a surface intersection, or3Dground truth. Broad/slanted footprints can introduce bias; quantify this before admission.',
 opacity='Native alpha normalized, alpha>1e-6 for finite support; no mature-only threshold or residual trimming.',
 fixed=dict(seed=6060,steps=18000,height=1536,initial_points=200000,weight=.002,shared_start=1,scale_weight=100.,ncc_factor=1.,camera_updates=0),
 numerical_admission=['Opposite layers do not cancel and gradients pull toward target depth',
  'Shifted moment formula equals direct native per-contributor squared3Derror',
  'Native RGB/plane maps and original gradient outputs agree without new loss',
  'Newloss finite on frozen early and mature models; nonzero position gradient; no optimizer changes',
  'Autograd agrees with finite differences for shifted moment algebra',
  'Finite ray-depth behavior with near-parallel normals; quantify slanted-surface footprint bias against fixed image-derived uncertainty'],
 training_admission='Only after all numerical checks pass and slanted-footprint approximation remains below target uncertainty for audited actual samples. Otherwise stop and diagnose; no automatic alternative or weight sweep.',
 training_evaluation=['Fresh refit shared landmarks and neutral body/head;200k/18k no previous learned state',
  'Same target/visibility and unchanged175full-image cameras vs098',
  'Fixed facial real-image crops, all-body landmarks, masks and held independent texture correspondences',
  'Actual continuous novel-view review and standalone+Y fullcovariance/allSHPLY compatibility'],
 stop='Nonfinite loss/gradient, failed parity, camera/input mismatch, lost feature support or fixed point budget violation.',
 optimizer_updates=0,training_started=False,quality_accepted=False,goal_complete=False,
 source_before_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(O/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(prepared=str(O),renderer=str(target),training_started=False)))
