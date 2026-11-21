"""Fix fixed-width COLMAP binary fields in the isolated reader on Windows."""
import difflib
import hashlib
import json
from pathlib import Path
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(pycolmap.__file__).parent / 'scene_manager.py'
EXPECTED_PARENT = ROOT / '.venv-large-gs' / 'Lib' / 'site-packages'
if not TARGET.resolve().is_relative_to(EXPECTED_PARENT.resolve()):
    raise RuntimeError('Refuse to patch a reader outside the dedicated environment')
evidence = ROOT / 'outputs' / 'large-scene-gs-001' / 'environment'
evidence.mkdir(parents=True, exist_ok=True)
original = TARGET.read_bytes()
before = original.decode('utf-8').replace('\r\n', '\n')
substitutions = {
    "struct.unpack('L', f.read(8))": "struct.unpack('<Q', f.read(8))",
    "struct.unpack('IiLL', f.read(24))": "struct.unpack('<IiQQ', f.read(24))",
}
counts = {needle: before.count(needle) for needle in substitutions}
if counts != {"struct.unpack('L', f.read(8))": 3, "struct.unpack('IiLL', f.read(24))": 1}:
    raise RuntimeError(f'Unexpected source or already patched: {counts}')
after = before
for old, new in substitutions.items():
    after = after.replace(old, new)
backup = evidence / 'pycolmap-scene_manager.original.py'
if backup.exists() and backup.read_bytes() != original:
    raise RuntimeError('Preserve earlier reader backup')
backup.write_bytes(original)
patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
    fromfile='pycolmap/scene_manager.py (pinned original)', tofile='pycolmap/scene_manager.py (Windows binary ABI)'))
(evidence / 'pycolmap-windows-binary.patch').write_text(patch, encoding='utf-8')
TARGET.write_text(after, encoding='utf-8', newline='\n')
receipt = {'scope': 'Only four unpack fields; original COLMAP little-endian uint64 camera/image/point counts and uint64 image dimensions. No trainer or data change.',
    'upstream_commit': 'cc7ea4b7301720ac29287dbe450952511b32125e',
    'original_sha256': hashlib.sha256(original).hexdigest(),
    'patched_sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
    'target': str(TARGET), 'replacement_counts': counts,
    'reason': 'Native unsigned long is 4 bytes on Windows versus 8 bytes on Linux x86_64; binary files have fixed 8-byte fields.'}
(evidence / 'pycolmap-windows-binary.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
