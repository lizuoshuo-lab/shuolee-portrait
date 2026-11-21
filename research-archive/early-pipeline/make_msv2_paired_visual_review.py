"""Create new scientific contact sheets from completed, unchanged face PNGs."""
from pathlib import Path
import hashlib
import json
import math
from PIL import Image,ImageDraw,ImageFont

record=Path('D:/00/记录/2026-11-06_MSV2_video15_子母图联合训练')
out=record/'visual-review'
out.mkdir(exist_ok=False)
fonts=Path('C:/Windows/Fonts/msyh.ttc')
font=ImageFont.truetype(str(fonts),17)
large=ImageFont.truetype(str(fonts),22)
small=ImageFont.truetype(str(fonts),14)
stems=('original_face_4x.png','face_composite4x_face_4x.png')
labels=('原始1倍训练','新子母图联合训练')
rows=json.loads((record/'face_composite_view_templates.json').read_text(encoding='utf-8'))['records']
splits={r['frame']:r['split'] for r in rows}
files=sorted(p for p in (record/'frames').rglob('*.png'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
before={str(p.relative_to(record)):sha(p) for p in files}
pages=[]
frames=sorted(p for p in (record/'frames').iterdir() if p.is_dir())
assert len(frames)==65
for start in range(0,len(frames),8):
    batch=frames[start:start+8]
    cellw,cellh=262,256
    header=83
    canvas=Image.new('RGB',(4*cellw,header+4*cellh+30),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((14,8),f'MSV2 面部概览 · {start+1}–{start+len(batch)} / 65',font=large,fill='#111111')
    draw.text((14,40),'仅公平缩略显示；原始1倍与新子母图使用同一完整4倍相机渲染像素。',font=small,fill='#333333')
    draw.text((14,61),'细节与来源以原六面板PNG为准；不代表未知视角或3D质量通过。',font=small,fill='#333333')
    for index,folder in enumerate(batch):
        row=index//2
        pair=index%2
        for mode,(stem,label) in enumerate(zip(stems,labels)):
            x=(pair*2+mode)*cellw
            y=header+row*cellh
            split='留出' if splits[folder.name]=='test' else '训练'
            draw.rectangle((x+3,y+2,x+cellw-4,y+cellh-3),outline='#c7ccd1',width=1)
            draw.text((x+9,y+7),f'{int(folder.name[-6:]):02d} {split} | {label}',font=font,fill='#111111')
            with Image.open(folder/stem) as image:
                image=image.convert('RGB')
                image.thumbnail((cellw-18,cellh-43),Image.Resampling.LANCZOS)
                canvas.paste(image,(x+(cellw-image.width)//2,y+34+(cellh-43-image.height)//2))
    path=out/f'overview_{start//8+1:02d}.png'
    canvas.save(path)
    pages.append(dict(file=path.name,frames=[p.name for p in batch],size=list(canvas.size)))
after={str(p.relative_to(record)):sha(p) for p in files}
assert before==after
state=dict(schema='msv2-paired-readonly-visual-contactsheet-v1',quality_accepted=False,
    source_frames=65,overview_pages=pages,viewed_panels=['original1','paired'],
    thumbnail_method='Equal cell size, aspect-preserving PIL Lanczos; no sharpening or image alteration',
    source_png_count=len(files),source_png_before=before,source_png_after=after,source_png_unchanged=True,
    planned_focus_frames=[1,9,17,25,33,41,49,57,65],
    limitation='Visual inspection of fitted views, including heldout source images; no novel-view or real3D ground truth acceptance')
(out/'contactsheet-manifest.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(pages=pages,source_png_count=len(files),source_png_unchanged=True),ensure_ascii=False))
