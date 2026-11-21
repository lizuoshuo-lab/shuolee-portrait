"""Create isolated versioned consumers; original project is never edited."""
from pathlib import Path
import hashlib,json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'native-residual-fit'
plan=json.loads((O/'plan.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p)==h for p,h in plan['frozen'].items())
dest=H/'native_surface122_v2.py'
if dest.exists():raise FileExistsError(dest)
s=(H/'native_surface122.py').read_text(encoding='utf-8')
s=s.replace('from native_training_admission122 import','from native_training_admission122_v2 import')
s=s.replace('from pathlib import Path','from pathlib import Path\nfrom native_residual_camera122 import project_world,validate_camera\nfrom native_residual_domain122 import sampled_domain_check')
s=s.replace('def evaluate_support(points,views,K,base,','def evaluate_support(points,views,cameras,base,')
s=s.replace('points=np.asarray(points,np.float64);K=np.asarray(K,np.float64)','points=np.asarray(points,np.float64)\n    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError("Finite world point matrix required")\n    if len({r["frame_id"] for r in views})!=len(views):raise ValueError("Distinct observed views required")')
s=s.replace("E=np.asarray(row['w2c'],np.float64)","camera=cameras[row['frame_id']];validate_camera(camera);sampled_domain_check(camera)\n        if camera['id']!=str(row['frame_id']):raise ValueError('Surface camera/frame identity mismatch')\n        E=np.asarray(camera['w2c'],np.float64)")
s=s.replace('height,width=alpha.shape','height,width=alpha.shape\n        if [width,height]!=[camera["width"],camera["height"]]:raise ValueError("Native mask/camera raster mismatch")')
a=s.index('            p=points[start:start+250000]');b=s.index('            ok=(cam[:,2]>0)',a)
s=s[:a]+'''            p=points[start:start+250000];cam=p@E[:3,:3].T+E[:3,3]
            uv=np.full((len(p),2),np.nan);positive_depth=cam[:,2]>0
            if positive_depth.any():uv[positive_depth]=project_world(p[positive_depth],camera)-.5
            # CameraK uses corners; raw masks use integer centers. Convert once.
'''+s[b:]
s=s.replace('def initialize_surface(fit,views,K,base,','def initialize_surface(fit,views,cameras,base,')
s=s.replace('evaluate_support(world,views,K,base,','evaluate_support(world,views,cameras,base,')
s=s.replace("world_frame_unchanged=True,input_pixel_convention='integer_centers'","world_frame_unchanged=True,input_pixel_convention='integer_centers',camera_projection='opencv_brown12_output',complete_cameras_used=True")
compile(s,str(dest),'exec');dest.write_text(s,encoding='utf-8')

original=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/shuolee_portrait/fitting.py')
dest=H/'fitting_output_residual122.py'
if dest.exists():raise FileExistsError(dest)
s=original.read_text(encoding='utf-8')
s=s.replace('from .runtime import c, ASSETS','from shuolee_portrait.runtime import c, ASSETS\nfrom native_residual_fit122 import FixedOutputProjector,cameras_from_fit_rows,shared_anatomical_visibility')
s=s.replace('    import nvdiffrast.torch as dr\n','')
s=s.replace("if anatomy_mode not in ('none','mesh_surface_zbuffer_v1'):","if anatomy_mode != 'continuous_mesh_rays_v2':")
s=s.replace('from .observation_support import','from shuolee_portrait.observation_support import')
s=s.replace("    E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);times=np.array([r['time_seconds'] for r in rows])","    cameras=cameras_from_fit_rows(rows)\n    E=np.array([r['w2c'] for r in cameras]);K=np.array([r['K'] for r in cameras]);times=np.array([r['time_seconds'] for r in rows])")
a=s.index('    def project(x):');b=s.index('    def forward():',a)
s=s[:a]+'''    fixed_projector=FixedOutputProjector(cameras,device=eg.device,dtype=eg.dtype)
    def project(x):
        return fixed_projector.project(x)
'''+s[b:]
a=s.index('    ctx=dr.RasterizeCudaContext();');b=s.index('    vis,face_vis,feet_vis=visibility(forward())',a)
s=s[:a]+'''    faces=model.faces_tensor.int()
    from smplx.vertex_ids import vertex_ids
    feet_vertices=torch.tensor([vertex_ids['smplx'][x] for x in ['LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']],device='cuda')
    regions=[15,15,15,15,15,16,17,18,19,20,21,1,2,4,5,7,8]
    regional=(model.lbs_weights[:,regions]>.20).float()
    @torch.no_grad()
    def visibility(st):
        cpu=lambda x:x.detach().cpu().numpy()
        region,face,foot=shared_anatomical_visibility(cpu(st['v']),cpu(faces),cameras,cpu(regional),
            cpu(face_ids),cpu(face_bary),cpu(model.lmk_faces_idx),cpu(feet_vertices))
        return g(region),g(face),g(foot)
'''+s[b:]
s=s.replace('anatomical_visibility=anatomy_mode,anatomical_visibility_is_model_derived=True,external_occluders_modeled=False,',
'''anatomical_visibility=anatomy_mode,anatomical_visibility_is_model_derived=True,external_occluders_modeled=False,
        camera_projection='opencv_brown12_output',cameras_frozen=True,pixel_bridge='cornerK projection minus0.5 to rawintegercenter',
        visibility_policy='Continuous exact point-rays at all mesh vertices and face/feet; changed from raster-sampled regional visibility, not a D-only comparison',
        prior_projection='GVHMR remains a coarse learned initialization; not material/camera truth',
        invalid_fit_depth='reject, no clamp or fallback',geometry_adapter_sha256=c.sha(Path(__file__).parent/'native_residual_fit122.py'),''')
compile(s,str(dest),'exec')
assert 'dr.rasterize' not in s and 'cam@kg' not in s and 'surface_supported' not in s
dest.write_text(s,encoding='utf-8')
(O/'implementation.json').write_text(json.dumps(dict(files={str(H/n):sha(H/n) for n in ['native_residual_fit122.py','native_surface122_v2.py','fitting_output_residual122.py']},
    old_sources_unchanged=True,person_entry_wired=False,remaining='prepare-native-person and fit_gate review overlays still require versioned full-camera binding; no positive person execution claimed.'),indent=2),encoding='utf-8')
print('R6_VERSIONED_COMPONENTS_READY')
