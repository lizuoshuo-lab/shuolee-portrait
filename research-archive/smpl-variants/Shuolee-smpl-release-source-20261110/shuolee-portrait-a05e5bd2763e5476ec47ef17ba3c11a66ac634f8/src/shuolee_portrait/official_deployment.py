"""Provision the official route's inputs without importing a training runtime."""
import ast
import hashlib
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PROFILE = 'official_smplifyx'
REPOSITORY = 'https://github.com/vchoutas/smplify-x.git'
ASSET_PATHS = (
    'detection/yolov8n.pt', 'rtmw/normalization.json', 'rtmw/rtmw-l-384x288.onnx',
    'sam2/sam2.1_hiera_tiny.pt', 'smplx/SMPLX_NEUTRAL.npz',
    'vposer/V02_05/V02_05.yaml', 'vposer/V02_05/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt',
)


def selected_manifest(manifest, profile):
    if profile == 'legacy':
        return manifest
    if profile != PROFILE:
        raise ValueError('Unknown deployment profile: ' + str(profile))
    rows = {r['path']: r for r in manifest['files']}
    missing = set(ASSET_PATHS)-rows.keys()
    if missing:
        raise ValueError('Missing official model specifications: ' + str(sorted(missing)))
    return dict(manifest, files=[rows[name] for name in ASSET_PATHS])


def official_pin():
    # Bootstrap runs before NumPy/Torch installation. Read the existing literal
    # pin definitions; do not import or execute the fitting module.
    nodes = ast.parse(Path(__file__).with_name('official_smplifyx.py').read_text(encoding='utf-8')).body
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in nodes
              if isinstance(n, ast.Assign) and len(n.targets) == 1 and
              isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('COMMIT', 'PINNED')}
    return values['COMMIT'], values['PINNED']


def check_source(source_root):
    root = Path(source_root).expanduser().resolve()
    commit, pinned = official_pin()
    for relative, expected in pinned.items():
        path = root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Official source is missing or differs from the fixed version: ' + str(path))
    return dict(source_root=str(root), repository=REPOSITORY, commit=commit,
                files_sha256=pinned, source_hashes_verified=True)


def prepare_source(root, explicit=None):
    """Reuse verified sources, or fetch one pinned public Git revision."""
    if explicit is not None:
        return check_source(explicit)
    destination = Path(root).expanduser().resolve()/'sources/smplify-x'
    if destination.exists():
        return check_source(destination)  # Never reset an existing checkout.
    destination.parent.mkdir(parents=True, exist_ok=True)
    commit, _ = official_pin()
    with tempfile.TemporaryDirectory(prefix='.smplify-x-', dir=destination.parent) as staging:
        staging = Path(staging).resolve()
        if staging.parent != destination.parent.resolve():
            raise ValueError('Source staging escaped the requested runtime directory.')
        def git(*args):
            subprocess.run(['git', '-C', str(staging), *args], check=True,
                           stdout=sys.stderr, stderr=sys.stderr)
        git('init', '--quiet')
        git('remote', 'add', 'origin', REPOSITORY)
        git('fetch', '--depth', '1', 'origin', commit)
        git('checkout', '--detach', '--quiet', 'FETCH_HEAD')
        actual = subprocess.check_output(['git', '-C', str(staging), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise ValueError('Official source checkout does not match its pinned commit.')
        check_source(staging)
        if destination.exists():
            raise FileExistsError('Source destination appeared during preparation: ' + str(destination))
        staging.rename(destination)
    return check_source(destination)


def official_module_names():
    return (
        'torch', 'torchvision', 'cv2', 'pycolmap', 'smplx', 'roma', 'rtmlib',
        'ultralytics', 'onnxruntime', 'hydra', 'pytorch_lightning', 'numpy', 'scipy', 'yaml',
        'nvdiffrast.torch', '_nvdiffrast_c', 'pytorch3d._C', 'simple_knn._C',
        'diff_gaussian_rasterization._C', 'sam2.build_sam', 'human_body_prior.models.vposer_model',
    )


def gpu_dependencies():
    from . import runtime as rt
    torch = rt.c.runtime()
    for name in official_module_names():
        importlib.import_module(name)
    import pycolmap
    if pycolmap.__version__ != '3.10.0':
        raise RuntimeError('Requires pycolmap==3.10.0.')
    return dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                pycolmap=pycolmap.__version__, sfm_gpu_available=pycolmap.has_cuda,
                required_modules_imported=True, cuda_available=True)


def doctor(assets, source_root):
    from . import runtime as rt
    rt.configure(assets)
    tools = {}
    for name in ('ffprobe', 'ffmpeg'):
        tools[name] = shutil.which(name)
        if not tools[name]:
            raise RuntimeError('Missing ' + name + '; install FFmpeg on the server.')
    inputs = dict(assets=rt.verify_assets(ASSET_PATHS), official_smplifyx=check_source(source_root))
    return dict(profile=PROFILE, **inputs, **tools, **gpu_dependencies(),
                gvhmr_required=False, training_started=False, quality_accepted=False)


def configure_deployment(assets, runs, output, source_root=None):
    from . import runtime as rt
    assets, runs, output = [Path(p).expanduser().resolve() for p in (assets, runs, output)]
    for label, path in [('assets', assets), ('runs', runs), ('deployment profile', output)]:
        if path == rt.PROJECT or rt.PROJECT in path.parents:
            raise ValueError(label + ' must be outside the Git checkout.')
    source = prepare_source(runs.parent, explicit=source_root)
    checks = doctor(assets, source['source_root'])
    profile = dict(schema='smplx-deployment-v1', profile=PROFILE, created=rt.now(),
        assets=str(assets), runs=str(runs), environment=str(Path(sys.prefix).resolve()),
        official_smplifyx=source, doctor=checks, viewer_required=False, training_started=False)
    rt.save(output, profile)
    return profile


def fitting_options(options, deployment):
    """Fill source location only; shape/reference policy stays caller-owned."""
    if not isinstance(options, dict):
        return options
    result = dict(options)
    if not result.get('source_root'):
        source = deployment.get('official_smplifyx', {}).get('source_root')
        if source:
            result['source_root'] = source
    return result
