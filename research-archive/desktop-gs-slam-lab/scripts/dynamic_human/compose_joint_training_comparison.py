"""Compose four verified saved training views without image enhancement."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-joint-pose-surface-024'
CONTROL=LAB/'outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/surface_and_appearance'

def main():
    assert (OUT/'independent-image-audit.json').exists()
    target=OUT/'paired-audit';target.mkdir(exist_ok=True)
    receipt=target/'training-figures.json'
    assert not receipt.exists(),'Preserve completed comparison'
    rows=json.loads((OUT/'training-rois.json').read_text())['regions']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    figures=[]
    for frame,camera in [(129,4),(131,4),(149,6),(151,6)]:
        stem=f'frame{frame:03d}-camera{camera}'
        row=next(r for r in rows if (r['frame'],r['camera'])==(frame,camera))
        box=tuple(row['box'])
        paths=[OUT/'training-ground-truth'/(stem+'.png'),
               OUT/'training-evaluation/before'/(stem+'-prediction.png'),
               CONTROL/'training-evaluation/after'/(stem+'-prediction.png'),
               OUT/'training-evaluation/after'/(stem+'-prediction.png')]
        panels=[]
        for path in paths:
            with Image.open(path) as im:
                assert list(im.size)==row['image_size']
                panels.append(im.convert('RGB').crop(box))
        with Image.open(CONTROL/'training-ground-truth'/(stem+'.png')) as ctrl_gt:
            with Image.open(paths[0]) as gt:assert np.array_equal(np.asarray(gt),np.asarray(ctrl_gt))
        scale=min(3.,310/panels[0].width,620/panels[0].height)
        size=tuple(round(v*scale) for v in panels[0].size)
        canvas=Image.new('RGB',(1320,size[1]+86),'white');draw=ImageDraw.Draw(canvas)
        for col,(title,panel) in enumerate(zip(['原始观测','共同起点','表面与颜色','姿态、表面与颜色'],panels)):
            draw.text((col*330+10,8),title,font=font,fill='#182d39')
            canvas.paste(panel.resize(size,Image.Resampling.NEAREST),(col*330+(330-size[0])//2,38))
        draw.text((10,size[1]+49),f'训练帧 {frame} / 相机 {camera} · 后两组各 100 次全区域更新 · 未锐化',font=font,fill='#182d39')
        output=target/(stem+'-training-comparison.png');assert not output.exists();canvas.save(output)
        figures.append(dict(frame=frame,camera=camera,box=list(box),file=output.name,
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            sources=[dict(path=str(p.relative_to(LAB)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    receipt.write_text(json.dumps(dict(figures=figures,visual_review_pending=True),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(training_comparisons=len(figures))))

if __name__=='__main__':main()
