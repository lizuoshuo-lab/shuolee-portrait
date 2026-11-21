"""Faithful Poppler crops; PDF figures and their labels are not redrawn."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import fitz
from PIL import Image

LAB = Path(__file__).resolve().parents[1]
PDF = LAB / 'outputs/3dgs-foundations-figures/source.pdf'
OUT = LAB / 'outputs/3dgs-paper-course-001/figures'
OUT.mkdir(parents=True, exist_ok=True)
POPPLER = shutil.which('pdftoppm')
if not POPPLER:
    raise SystemExit('Poppler pdftoppm is required for faithful original figure crops.')
SOURCE = 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf'
# Page, crop in PDF points, DPI, purpose. Boxes include every panel and panel label.
SPECS = {
 'paper-fig-01.png': (1,[51,167,562,287],350,'渲染质量、训练时间与帧率的作者示例'),
 'paper-fig-02.png': (5,[48,77,563,174],350,'方法总览'),
 'paper-fig-03.png': (5,[49,225,296,328],450,'各向异性高斯的三维形状可视化'),
 'paper-fig-04.png': (6,[49,78,297,222],450,'复制与分裂机制'),
 'paper-fig-05.png': (7,[51,76,562,648],350,'完整九场景新视角定性对照'),
 'paper-fig-06.png': (8,[51,187,296,337],450,'7K与30K训练对照'),
 'paper-fig-07.png': (10,[51,77,296,272],450,'随机初始化与SfM初始化'),
 'paper-fig-08.png': (10,[51,320,296,615],450,'复制与分裂消融'),
 'paper-fig-09.png': (10,[317,77,562,159],450,'限制反向传播的高斯数量'),
 'paper-fig-10.png': (11,[51,77,562,180],350,'各向异性与各向同性消融'),
 'paper-fig-11.png': (11,[51,395,296,468],450,'训练场景观察不足时的局部失败案例'),
 'paper-fig-12.png': (11,[51,550,296,634],450,'偏离训练观察方向的新视角失败案例'),
 'paper-table-01.png': (8,[51,106,562,176],450,'三个数据集的质量、训练时间、帧率和显存'),
 'paper-table-02.png': (8,[51,442,296,490],500,'Synthetic NeRF定量结果'),
 'paper-table-03.png': (9,[51,108,562,201],450,'完整模块消融表'),
}
doc = fitz.open(PDF)
records=[]
for name,(page,box,dpi,purpose) in SPECS.items():
    ratio=dpi/72
    x0,y0,x1,y1=box
    x,y=math.floor(x0*ratio),math.floor(y0*ratio)
    w,h=math.ceil(x1*ratio)-x,math.ceil(y1*ratio)-y
    prefix=OUT/Path(name).stem
    subprocess.run([POPPLER,'-f',str(page),'-l',str(page),'-r',str(dpi),
        '-x',str(x),'-y',str(y),'-W',str(w),'-H',str(h),'-singlefile','-png',str(PDF),str(prefix)],check=True,capture_output=True)
    number=int(Path(name).stem.split('-')[-1])
    kind='Fig.' if '-fig-' in name else 'Table'
    captions=[b[4].strip().replace('\n',' ') for b in doc[page-1].get_text('blocks')
        if b[4].startswith(f'{kind} {number}.')]
    im=Image.open(prefix.with_suffix('.png'))
    records.append({'filename':name,'kind':'paper-figure' if kind=='Fig.' else 'paper-table',
        'figureNumber':number if kind=='Fig.' else None,'tableNumber':number if kind=='Table' else None,
        'pdfPage1Based':page,'cropBBoxPDFPoints':box,'renderDpi':dpi,'pixelSize':list(im.size),
        'sha256':hashlib.sha256(prefix.with_suffix('.png').read_bytes()).hexdigest(),
        'sourceURL':SOURCE,'readingUse':purpose,'captionOriginal':' '.join(captions),
        'modification':'Faithful Poppler rendering and rectangular crop removing surrounding prose. All original panels and panel labels retained; figure content unchanged.',
        'license':'CC-BY-4.0','licenseURL':'https://creativecommons.org/licenses/by/4.0/',
        'licenseEvidenceUrl':'https://arxiv.org/abs/2308.04079'})
manifest={'paperTitle':'3D Gaussian Splatting for Real-Time Radiance Field Rendering',
    'authors':['Bernhard Kerbl','Georgios Kopanas','Thomas Leimkühler','George Drettakis'],
    'year':2023,'sourceURL':SOURCE,'sourcePDFSHA256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
    'licenseEvidence':{'url':'https://arxiv.org/abs/2308.04079','observedLicenseHref':'http://creativecommons.org/licenses/by/4.0/',
    'verifiedOn':'2026-10-11','note':'The arXiv record explicitly links CC BY 4.0. The author-hosted PDF contains stale ACM template copyright placeholders; those placeholders were not used to infer permission.'},
    'paperAssets':records}
(OUT/'paper-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'count':len(records),'dimensions':{r['filename']:r['pixelSize'] for r in records}},ensure_ascii=False))
