"""Keep native build outputs outside the source checkout and WPS directory."""
import json
from pathlib import Path
import shutil
import sys
import tempfile

project = Path(__file__).resolve().parents[1]
install = Path(sys.argv[1]).resolve()
install.mkdir(parents=True, exist_ok=True)
stage = Path(tempfile.mkdtemp(prefix='native-', dir=install))
sources = {
    'diff-gaussian-rasterization': 'src/avatar_core/submodules/diff-gaussian-rasterization',
    'simple-knn': 'src/avatar_core/submodules/simple-knn',
    'nvdiffrast': 'third_party/nvdiffrast',
    'pytorch3d': 'third_party/pytorch3d',
}
for name, relative in sources.items():
    shutil.copytree(project / relative, stage / name,
                    ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', '*.so', '.git'))
(stage / 'source-map.json').write_text(json.dumps(sources, indent=2) + '\n')
print(stage)
