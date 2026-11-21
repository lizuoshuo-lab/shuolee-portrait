"""Read back the already built extension; keep the earlier symbol-check failure."""
from pathlib import Path
from datetime import datetime, timezone
import json, sys, os, hashlib, time, traceback, importlib.util

out = Path(sys.argv[1]).resolve()
plan = json.loads((out/'plan.json').read_text())
target = out/'readback-result.json'
assert not target.exists(), 'Readback already recorded'
assert sys.executable == plan['python']
assert os.environ.get('TORCH_EXTENSIONS_DIR') == plan['cache']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p, h in plan['source_sha256'].items():
    assert sha(p) == h, p
sys.path.insert(0, plan['gsplat_repo'])
start = time.monotonic()
r = dict(passed=False, started=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
         previous_result_sha256=sha(out/'result.json'), checker_sha256=sha(__file__),
         render_calls=0, fitting_steps=0, gaussian_steps=0)
try:
    import torch, gsplat
    assert torch.__version__ == '2.7.1+cu128' and gsplat.__version__ == '1.5.3'
    artifact = Path(plan['cache'])/'gsplat_cuda/gsplat_cuda.so'
    before_hash = sha(artifact)
    spec = importlib.util.spec_from_file_location('gsplat_cuda', artifact)
    _C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_C)
    symbols = ['projection_ut_3dgs_fused', 'rasterize_to_pixels_from_world_3dgs_fwd',
               'rasterize_to_pixels_from_world_3dgs_bwd', 'intersect_tile', 'intersect_offset',
               'UnscentedTransformParameters']
    for name in symbols:
        assert hasattr(_C, name), name
    assert hasattr(_C.CameraModelType, 'PINHOLE') and hasattr(_C.ShutterType, 'GLOBAL')
    artifact = Path(_C.__file__).resolve()
    assert artifact.is_relative_to(Path(plan['cache']).resolve())
    assert sha(artifact) == before_hash
    for p, h in plan['source_sha256'].items():
        assert sha(p) == h, p
    r.update(passed=True, python=sys.executable, torch=torch.__version__, gsplat=gsplat.__version__,
             extension=str(artifact), extension_sha256=sha(artifact), extension_bytes=artifact.stat().st_size,
             load_method='Direct existing ExtensionFileLoader, no JIT build request', symbols=symbols,
             cache=plan['cache'], plan_sha256=sha(out/'plan.json'),
             correction='The earlier checker confused Python wrapper names with C++ binding names. No backend source changed.')
except BaseException as e:
    r.update(error=type(e).__name__, message=str(e), traceback=traceback.format_exc())
r.update(seconds=time.monotonic()-start, finished=datetime.now(timezone.utc).isoformat())
with target.open('x', encoding='utf-8') as f:
    json.dump(r, f, indent=2, allow_nan=False)
print(json.dumps(r, indent=2))
raise SystemExit(0 if r['passed'] else 1)
