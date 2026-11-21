"""CPU-only export review contact sheets; all36 angles retained in order."""
from pathlib import Path
from PIL import Image
import argparse, json, shutil
from contact_capture118_reviews import sheet
ap=argparse.ArgumentParser();ap.add_argument('--subject',required=True);a=ap.parse_args()
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118/all-persons-20261027')
root=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-capture118-all-persons-20261027')/a.subject
result=json.loads((root/'gaussian-run/result.json').read_text());steps=result['steps']
d=R/f'{a.subject}-gaussian-{steps:06d}-review'
sheet(sorted(d.glob('*_face_raw_mask_render.png')),d/'face-contact.png',1000)
sheet(sorted(d.glob('*_body_raw_mask_render.png')),d/'body-contact.png',1000)
out=R/f'{a.subject}-final-review';out.mkdir(exist_ok=True)
paths=sorted((root/'export/review').glob('novel_*.png'));assert len(paths)==36
for b in range(3):
    canvas=Image.new('RGB',(4*320,3*480),'black')
    for j,p in enumerate(paths[b*12:(b+1)*12]):
        im=Image.open(p);assert im.size==(512,768)
        canvas.paste(im.resize((320,480)),((j%4)*320,(j//4)*480))
        shutil.copy2(p,out/p.name)
    canvas.save(out/f'orbit-{b:02d}.png')
print(str(out))
