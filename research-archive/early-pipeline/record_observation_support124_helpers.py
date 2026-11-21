"""Small filesystem helpers for immutable diagnostic receipts."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, data):
    path=Path(path)
    assert not path.exists(),path
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def windows(path):
    s=str(path)
    if s.startswith('/mnt/'):
        return Path(s[5].upper()+':/'+s[7:])
    if s.startswith('/'):
        return Path('//wsl.localhost/Ubuntu-20.04'+s)
    return Path(s)
