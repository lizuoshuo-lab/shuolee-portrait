from pathlib import Path
import json, shutil, hashlib, datetime

lab=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
repo=lab/'repos/PGSR-shuolee'
out=lab/'outputs/mini-splatting2-mip-depth-001/shuolee/current/static-depth-mixture-audit-078'
src=repo/'submodules/diff-plane-rasterization'
dst=repo/'submodules/diff-plane-moment-rasterization'
assert not dst.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','build','__pycache__'])}
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(),stage='isolated CUDA preflight only',optimizer_updates=0,
    hypothesis='Different planes contribute to the same pixel; an average-plane correspondence alone does not constrain their spread.',
    variable='Opacity-weighted point-to-plane squared residual about a detached blended surface hit.',
    source='Local adaptation; not the official 2DGS ray-depth distortion.',
    fixed=['077 checkpoint','175 whole-image SfM cameras','all original loss paths','renderer alpha/order'],
    checks=['same RGB, normals, alpha, distance and plane depth with zero new loss','same old-objective parameter gradients',
        'moment maps agree with saved exact per-Gaussian contributions','new-term finite differences with detached anchor fixed',
        'coherent tilted plane invariant','finite full-frame loss/gradients'],
    stop=['any baseline parity failure','nonfinite or unstable moment estimates','negative residual larger than numerical tolerance',
        'candidate quality gate not passed; no promotion or 00 deployment'],
    proposed_training='Not started. A separate protocol and fresh initialization are required.',
    source_sha256=before)
(out/'surface-preflight-protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__','*.so'))
(dst/'diff_plane_rasterization').rename(dst/'diff_plane_moment_rasterization')
p=dst/'setup.py';t=p.read_text();assert t.count('diff_plane_rasterization')==3;p.write_text(t.replace('diff_plane_rasterization','diff_plane_moment_rasterization'))
p=dst/'cuda_rasterizer/config.h';t=p.read_text();assert '#define NUM_ALL_MAP 5' in t;p.write_text(t.replace('#define NUM_ALL_MAP 5','#define NUM_ALL_MAP 15'))
p=dst/'cuda_rasterizer/backward.cu';t=p.read_text();assert t.count('dL_dout_all_map[MAP_N-1]')==1;p.write_text(t.replace('dL_dout_all_map[MAP_N-1]','dL_dout_all_map[4]'))
# Keep the original renderer and compiled extension untouched.
p=repo/'gaussian_renderer/__init__.py';t=p.read_text()
t=t.replace('from diff_plane_rasterization import','from diff_plane_moment_rasterization import')
t=t.replace('return_plane = True, return_depth_normal = True):','return_plane = True, return_depth_normal = True, moment_origin=None):')
t=t.replace('input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()',"assert moment_origin is not None, 'A fixed world-space origin is required'\n    from surface_concentration import plane_moments\n    origin_camera = moment_origin @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]\n    input_all_map = torch.zeros((means3D.shape[0], 15)).cuda().float()")
t=t.replace('input_all_map[:, 4] = local_distance','input_all_map[:, 4] = local_distance\n    shifted_distance = local_distance + (local_normal * origin_camera).sum(-1)\n    input_all_map[:, 5:] = plane_moments(local_normal, shifted_distance)')
t=t.replace('"rendered_distance": rendered_distance','"rendered_distance": rendered_distance,\n                    "plane_moments": out_all_map[5:15],\n                    "moment_origin_camera": origin_camera')
assert '15)).cuda()' in t and 'plane_moments' in t
(repo/'gaussian_renderer/surface.py').write_text(t,encoding='utf8')
assert before=={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file() and not any(x in p.parts for x in ['.git','build','__pycache__'])}
print(json.dumps(dict(source_preserved=True,new_extension=str(dst),wrapper=str(repo/'gaussian_renderer/surface.py'))))
