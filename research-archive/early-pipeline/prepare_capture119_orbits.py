"""Keep all36 exported novel angles for actual inspection; never label pixels."""
from pathlib import Path
import argparse,json,shutil
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--subject',required=True);a=p.parse_args()
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/clothed-surface-119')
root=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-clothed-surface-119')/a.subject
out=R/(a.subject+'-final-review');out.mkdir(exist_ok=True)
paths=sorted((root/'export/review').glob('novel_*.png'));assert len(paths)==36
for b in range(3):
    canvas=Image.new('RGB',(1280,1440),'black')
    for j,path in enumerate(paths[b*12:(b+1)*12]):
        im=Image.open(path);assert im.size==(512,768)
        canvas.paste(im.resize((320,480)),((j%4)*320,(j//4)*480));shutil.copy2(path,out/path.name)
    canvas.save(out/f'orbit-{b:02d}.png')
print(out)
