"""Contact sheets of all real heldout head crops; no synthesized illustration."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    out=ROOT/'native-resolution-visual-review-001';out.mkdir(exist_ok=False)
    models={arm:ROOT/f'native-resolution-cache-{arm}-18000-s0/resolution-evaluation-v2' for arm in ['low','full']}
    summaries={key:json.loads((path/'summary.json').read_text()) for key,path in models.items()}
    assert summaries['low']['references']==summaries['full']['references']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    records=[]
    for target in ['full','low']:
        rows=[row for row in summaries['low']['rows'] if row['target']==target];assert len(rows)==8
        tiles=[]
        for row in rows:
            name=row['frame'];tile=Image.new('RGB',(1008,354),'#edf1f1');draw=ImageDraw.Draw(tile)
            files=[('Real photo',models['low']/target/(name+'-head-gt.png')),
                ('Low-resolution training',models['low']/target/(name+'-head.png')),
                ('Full-resolution training',models['full']/target/(name+'-head.png'))]
            dimensions=[]
            for index,(label,path) in enumerate(files):
                image=Image.open(path).convert('RGB');dimensions.append(image.size);image.thumbnail((328,316))
                tile.paste(image,(index*336,34));draw.text((index*336+3,7),name+' '+label,fill='#16343d',font=font)
            assert len(set(dimensions))==1
            records.append(dict(frame=name,target=target,native_crop_size=list(dimensions[0])))
            tiles.append(tile)
        for start in [0,4]:
            sheet=Image.new('RGB',(1008,1416),'#edf1f1')
            for index,tile in enumerate(tiles[start:start+4]):sheet.paste(tile,(0,index*354))
            sheet.save(out/f'{target}-heads-{1+start//4}.png')
    (out/'manifest.json').write_text(json.dumps(dict(status='prepared_pending_visual_review',records=records,
        thumbnails_for_overview_only=True,original_pixels_in_each_run_evaluation=True),indent=2));print(str(out))


if __name__=='__main__':main()
