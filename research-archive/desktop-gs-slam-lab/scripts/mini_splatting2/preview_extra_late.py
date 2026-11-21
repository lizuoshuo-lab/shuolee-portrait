"""All24 secondary head comparisons, keeping fixed source-frame ordering."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    source=ROOT/'viewer/native-late-assets/extra'
    data=json.loads((source/'summary.json').read_text())
    out=ROOT/'native-late-extra-visual-review-001';out.mkdir(exist_ok=False)
    frames=data['frames'];assert len(frames)==24
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for start in range(0,24,4):
        sheet=Image.new('RGB',(1008,1416),'#edf1f1');draw=ImageDraw.Draw(sheet)
        for row,name in enumerate(frames[start:start+4]):
            stem=Path(name).stem
            for col,(arm,label) in enumerate([('gt','Real photo'),('low','Original schedule'),('full','Late densification')]):
                image=Image.open(source/arm/(stem+'-head.png')).convert('RGB');image.thumbnail((328,316))
                sheet.paste(image,(col*336,row*354+34));draw.text((col*336+3,row*354+7),stem+' '+label,font=font,fill='#16343d')
        sheet.save(out/f'extra-heads-{start//4+1}.png')
    (out/'manifest.json').write_text(json.dumps(dict(status='prepared_pending_visual_review',frames=frames,
        scope='All24fixed secondary frames; contact sheets are thumbnails, original pixels retained',accepted=False),indent=2))
    print(str(out))


if __name__=='__main__':main()
