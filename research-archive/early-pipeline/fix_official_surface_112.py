from pathlib import Path
import os,sys,json,time,shutil,subprocess,hashlib,difflib
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/surface-native-alignment-112')
P=Path('/opt/gs-slam-lab/shuolee-surface112');src=P/'src-fixed'
assert not src.exists() and not (P/'target-fixed').exists()
shutil.copytree(P/'src',src,ignore=shutil.ignore_patterns('build','*.egg-info','__pycache__'))
path=src/'cuda_rasterizer/backward.cu';old=path.read_text()
before='''\t\t\t\t// // Propagate the gradients of depth
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 6],  s.x * dL_dz);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 7],  s.y * dL_dz);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);'''
after='''\t\t\t\t// The filter changes alpha, not the forward intersection depth.
\t\t\t\t// Differentiate z = dot(Tw, (s.x,s.y,1)) through s as well.
\t\t\t\tconst float2 dL_ds_depth = {dL_dz * Tw.x, dL_dz * Tw.y};
\t\t\t\tconst float gx = dL_ds_depth.x / p.z;
\t\t\t\tconst float gy = dL_ds_depth.y / p.z;
\t\t\t\tconst float3 dL_dp_depth = {gx, gy, -(gx * s.x + gy * s.y)};
\t\t\t\tconst float3 dL_dk_depth = cross(l, dL_dp_depth);
\t\t\t\tconst float3 dL_dl_depth = cross(dL_dp_depth, k);
\t\t\t\tconst float3 dL_dTu_depth = {-dL_dk_depth.x, -dL_dk_depth.y, -dL_dk_depth.z};
\t\t\t\tconst float3 dL_dTv_depth = {-dL_dl_depth.x, -dL_dl_depth.y, -dL_dl_depth.z};
\t\t\t\tconst float3 dL_dTw_depth = {
\t\t\t\t\tpixf.x * dL_dk_depth.x + pixf.y * dL_dl_depth.x + dL_dz * s.x,
\t\t\t\t\tpixf.x * dL_dk_depth.y + pixf.y * dL_dl_depth.y + dL_dz * s.y,
\t\t\t\t\tpixf.x * dL_dk_depth.z + pixf.y * dL_dl_depth.z + dL_dz};
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu_depth.x);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu_depth.y);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu_depth.z);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv_depth.x);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv_depth.y);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv_depth.z);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw_depth.x);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw_depth.y);
\t\t\t\tatomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw_depth.z);'''
assert old.count(before)==1
new=old.replace(before,after);path.write_text(new)
(O/'filtered-depth-backward.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='official/backward.cu',tofile='fixed/backward.cu')))
shutil.copy2(path,O/'backward-filter-depth-fixed.cu')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(scope='Only low-pass branch intersection depth chain rule in isolated official rasterizer',
    forward_unchanged=sha(src/'cuda_rasterizer/forward.cu')==sha(P/'src/cuda_rasterizer/forward.cu'),
    original_backward_sha256=sha(P/'src/cuda_rasterizer/backward.cu'),fixed_backward_sha256=sha(path),
    targets=[str(P/'target'),str(P/'target-fixed')],renderer_training_unchanged=True,
    previous_controls='official-unitquat-result.json',thresholds_unchanged=True,start=time.time())
assert protocol['forward_unchanged'];(O/'fix-protocol.json').write_text(json.dumps(protocol,indent=2))
env=os.environ.copy();env.update(MAX_JOBS='3',TORCH_CUDA_ARCH_LIST='12.0',PYTHONDONTWRITEBYTECODE='1')
print(json.dumps(dict(status='building_fixed_depth_gradient',pid=os.getpid())),flush=True)
with (O/'build-fixed.log').open('w') as log:
    r=subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','--target',str(P/'target-fixed'),str(src)],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
result=dict(exit_code=r.returncode,built=r.returncode==0,seconds=time.time()-protocol['start'],target=str(P/'target-fixed'))
(O/'build-fixed-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
