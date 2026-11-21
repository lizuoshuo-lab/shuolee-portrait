"""Build the small header-layout repair from the previously verified offline package."""
from pathlib import Path
import hashlib, json, zipfile

ROOT=Path(__file__).resolve().parent
PROJECT=Path('D:/00/工作/人体重建')
OUT=Path('D:/00/工作/SMPL-X/server113_pip_repair.pyz')
CACHE=Path('D:/GS-SLAM-Data/shuolee-portrait-transfer-cache/pip-bootstrap')
def sha(data): return hashlib.sha256(data).hexdigest()
original=OUT.read_bytes()
saved=CACHE/('server113_pip_repair-'+sha(original)[:16]+'.pyz')
if not saved.exists(): saved.write_bytes(original)
server=(PROJECT/'scripts/setup_server.sh').read_bytes().replace(b'\r\n',b'\n')
route=(PROJECT/'scripts/setup_113.sh').read_bytes().replace(b'\r\n',b'\n')
assert sha(server)=='c026fd0da9fbe3d82a98f13f0164562c788a369ad4546a449d0e100a581297ce'
assert sha(route)=='6c96721ff67bca9ef3f7709a3a4524cf979b6d262f3dcb4525eb964336542406'
s=server.decode();start=s.index('# Split CUDA toolkit');end=s.index('# Stage native sources',start)
s=s[:start]+'''# Include the full CUDA development layout in host C++ builds as well as nvcc.
CUDA_BUILD_EXPORTS="$("$ENV_PYTHON" -s "$PROJECT_ROOT/scripts/cuda_build_env.py" --probe --report "$SHUOLEE_ENV_DIR/install/cuda-build-headers.json")"
eval "$CUDA_BUILD_EXPORTS"
'''+s[end:]
r=route.decode();start=r.index('export CPATH=');end=r.index('mkdir -p',start)
r=r[:start]+'''CUDA_BUILD_EXPORTS="$("$PY" -s "$PROJECT_ROOT/scripts/cuda_build_env.py" --probe --report "$SHUOLEE_ENV_DIR/install/cuda-build-headers-113.json")"
eval "$CUDA_BUILD_EXPORTS"
'''+r[end:]
payloads=[('scripts/setup_server.sh',server,s.encode()),('scripts/setup_113.sh',route,r.encode()),
          ('scripts/cuda_build_env.py',None,(ROOT/'server113_cuda_build_env.py').read_bytes().replace(b'\r\n',b'\n'))]
patch=dict(reason='QQ compiler log 2026-10-27 10:31: cuda_runtime.h cannot find crt/host_config.h in split CUDA12.8 toolkit.',
           baseline_git='5fda749',files=[dict(path=p,before_sha256=sha(old) if old else None,after_sha256=sha(new)) for p,old,new in payloads])
stage=CACHE/'header-patch';stage.mkdir(exist_ok=True)
for p,_,new in payloads:
    dst=stage/p;dst.parent.mkdir(exist_ok=True);dst.write_bytes(new)
temp=OUT.with_suffix('.partial')
with zipfile.ZipFile(saved) as old, zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
    manifest=json.loads(old.read('manifest.json'))
    for wheel in manifest['wheels']:
        data=old.read('wheels/'+wheel['filename']);assert sha(data)==wheel['sha256']
        z.writestr('wheels/'+wheel['filename'],data)
    z.writestr('manifest.json',json.dumps(manifest,indent=2))
    z.write(ROOT/'server113_pip_repair_main.py','__main__.py')
    z.writestr('install-patch.json',json.dumps(patch,indent=2))
    for p,_,new in payloads:z.writestr('install-patch/'+p,new)
temp.replace(OUT)
OUT.with_name(OUT.name+'.sha256').write_text(sha(OUT.read_bytes())+'  '+OUT.name+'\n',encoding='ascii',newline='\n')
(stage/'manifest.json').write_text(json.dumps(patch,indent=2))
print(json.dumps(dict(bytes=OUT.stat().st_size,sha256=sha(OUT.read_bytes()),preserved_previous_package=str(saved),patch_files=[p for p,_,_ in payloads])))
