"""Fetch only small primary-source research assets for the existing report."""
from pathlib import Path
import urllib.request
import json
import hashlib
from datetime import datetime, timezone

LAB = Path(__file__).resolve().parents[2]
OUT = LAB/'outputs/mini-splatting2-mip-depth-001/shuolee/current/company-problem-audit/sources'
ASSETS = {
    'absgs-method.png': 'https://ty424.github.io/AbsGS.github.io/resources/image/pip.png',
    'pixelgs-method.png': 'https://pixelgs.github.io/resources/method.png',
    'gaussianavatars-method.jpg': 'https://raw.githubusercontent.com/ShenhanQian/GaussianAvatars/main/media/method.jpg',
    'absgs-gaussian_model.py': 'https://raw.githubusercontent.com/TY424/AbsGS/main/scene/gaussian_model.py',
    'pixelgs-gaussian_model.py': 'https://raw.githubusercontent.com/zhengzhang01/Pixel-GS/main/scene/gaussian_model.py',
    'gaussianavatars-readme.md': 'https://raw.githubusercontent.com/ShenhanQian/GaussianAvatars/main/README.md',
    'vhap-readme.md': 'https://raw.githubusercontent.com/ShenhanQian/VHAP/main/README.md',
}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name,url in ASSETS.items():
        p=OUT/name
        if not p.exists():
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'shuolee-research/1.0'}),timeout=40) as response:
                data=response.read(8*1024*1024+1)
                assert len(data)<=8*1024*1024, 'Large download belongs on computer 00'
            p.write_bytes(data)
        data=p.read_bytes()
        rows.append(dict(file=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (OUT/'manifest.json').write_text(json.dumps(dict(retrieved=datetime.now(timezone.utc).isoformat(),assets=rows,training_weights_downloaded=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__': main()
