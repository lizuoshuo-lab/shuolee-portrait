"""Copy unmodified source imagery and seal the reviewed local implementation."""
from pathlib import Path
import hashlib,json
from PIL import Image

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

source=LAB/'data/dynamic-human/dynamic3dgaussians/data/basketball/ims/10/000015.jpg'
target=OUT/'assets/basketball-source-camera10-frame015.png'
Image.open(source).convert('RGB').save(target)
figures=[
 dict(file='assets/dynamic3dgaussians-author-teaser.png',source='https://github.com/JonathonLuiten/Dynamic3DGaussians/blob/7dbbd4dec404308524ff402756bdb8143a2589b0/teaser_figure.png',kind='author complete teaser',transformation='none',license='image-specific license not explicitly stated; do not infer from code license'),
 dict(file='assets/omnire-figure02.png',source='https://arxiv.org/pdf/2408.16760v2',kind='complete Figure 2, PDF page 5',transformation='rendered at 5x; PyMuPDF rect [109,63,504,216] crops only content outside the figure; all figure panels and labels retained',license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',pdf_sha256=sha(OUT/'sources/omnire.pdf')),
 dict(file='assets/basketball-source-camera10-frame015.png',source='https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip',kind='original public camera image, not a local reconstruction',transformation='JPEG decoded and saved losslessly as PNG; no crop, resize, enhancement, or synthesis',source_sha256=sha(source),license='source data terms apply; no additional image license asserted')
]
for item in figures:
    p=OUT/item['file'];item['sha256']=sha(p);item['dimensions']=list(Image.open(p).size)
write(OUT/'figure-provenance.json',dict(figures=figures))
paths=[LAB/'scripts/dynamic_human'/f for f in ['adapter.py','train_benchmark.py','evaluate_benchmark.py']]
paths += [LAB/'repos/Dynamic3DGaussians'/f for f in ['train.py','helpers.py','external.py','README.md']]
write(OUT/'implementation-receipt.json',dict(stage='sealed before held-out model evaluation',files=[dict(path=p.relative_to(LAB).as_posix(),sha256=sha(p)) for p in paths]))
print('Source figure and implementation receipts saved.')
