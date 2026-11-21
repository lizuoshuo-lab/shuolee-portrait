from pathlib import Path
import os,sys,subprocess,json,shutil,hashlib,time
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/surface-native-alignment-112')
UP=O/'diff-surfel-rasterization'
P=Path('/opt/gs-slam-lab/shuolee-surface112')
assert not P.exists(), 'Isolated build already exists; inspect rather than rebuild blindly'
P.mkdir(parents=True)
src=P/'src';shutil.copytree(UP,src,ignore=shutil.ignore_patterns('.git'))
glm=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/PGSR-shuolee/submodules/diff-plane-rasterization/third_party/glm')
assert (glm/'glm/glm.hpp').exists()
shutil.copytree(glm,src/'third_party/glm',dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','doc','test'))
env=os.environ.copy();env.update(MAX_JOBS='3',TORCH_CUDA_ARCH_LIST='12.0',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
protocol=dict(scope='Compile official rasterizer into isolated target; existing site packages and primary renderer unchanged',
              previous_goal_turn='progress',upstream=json.loads((O/'official-source.json').read_text(encoding='utf-8-sig')),
              target=str(P/'target'),source=str(src),glm_source=str(glm),dependencies_installed=False,
              start=time.time(),max_runtime_seconds=300)
(O/'build-protocol.json').write_text(json.dumps(protocol,indent=2))
print(json.dumps(dict(status='building',pid=os.getpid(),target=str(P/'target'))),flush=True)
with (O/'build.log').open('w') as log:
    try:
        r=subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','--target',str(P/'target'),str(src)],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        result=dict(exit_code=r.returncode,target=str(P/'target'),seconds=time.time()-protocol['start'],built=r.returncode==0)
    except subprocess.TimeoutExpired:
        result=dict(exit_code=None,target=str(P/'target'),seconds=time.time()-protocol['start'],built=False,timeout=True)
(O/'build-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
