from pathlib import Path
import importlib.metadata as md
import json
import re
import sys

root = Path('/opt/gs-slam-lab/shuolee-portrait-clean-001')
check = json.loads((root/'env/install/environment-check.json').read_text())
assert check['passed'] and check['isolated_package_imports']
assert Path(sys.prefix).resolve() == root/'env'
native = {'diff-gaussian-rasterization', 'simple-knn', 'nvdiffrast', 'pytorch3d'}
canonical = lambda n: re.sub(r'[-_.]+', '-', n).lower()
versions = {canonical(d.metadata['Name']): d.version for d in md.distributions()}
text = '# Observed after isolated install and CUDA forward/backward acceptance.\n'
text += '# Linux x86_64, Python 3.10, PyTorch cu128; native sources are vendored and built separately.\n'
text += '\n'.join(f'{name}=={version}' for name, version in sorted(versions.items()) if name not in native)+'\n'
dest = Path('/mnt/d/00/工作/人体重建/constraints-linux-cu128.txt')
dest.write_text(text)
(root/'env/install/package-versions.json').write_text(json.dumps(versions, indent=2)+'\n')
print(json.dumps({'packages':len(versions), 'constraints':str(dest), 'native_built_separately':sorted(native)}))
