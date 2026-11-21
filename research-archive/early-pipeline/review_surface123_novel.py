"""Make unlabelled ordered contact sheets from existing final native renders."""
from pathlib import Path
import hashlib,json,argparse
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--subject',required=True);a=p.parse_args()
top=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
out=top/'shuolee/current/local-clothed-surface-123'/f'{a.subject}-novel-final-review'
src=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')/a.subject/'export/review'
out.mkdir(exist_ok=True)
files=sorted(src.glob('novel_*.png'));assert len(files)==36
entries=[];sheets=[]
for n in range(0,36,6):
    sheet=Image.new('RGB',(768,768),'black')
    for j,f in enumerate(files[n:n+6]):
        with Image.open(f) as im:
            assert im.size==(512,768)
            sheet.paste(im.resize((256,384),Image.Resampling.LANCZOS),((j%3)*256,(j//3)*384))
        entries.append(dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),angle_degrees=(n+j)*10))
    name=f'views_{n:03d}_{n+5:03d}.png';sheet.save(out/name)
    sheets.append(dict(file=name,frame=f'novel_{n:03d}..{n+5:03d}',region='continuous_novel_views',columns=['row-major six native rendered views, 10 degree spacing'],sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
(out/'index.json').write_text(json.dumps(dict(subject=a.subject,ordered_render_count=36,sequence='row-major; 10 degrees between consecutive views',source=str(src),images=sheets,source_images=entries,novel_view_ground_truth_available=False,image_labels_burned_in=False),indent=2)+'\n',encoding='utf-8')
print(out)
