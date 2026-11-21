"""Prepare deterministic extra-frame sheets and audit source split; no model edits."""
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'

def main():
    out=ROOT/'native-resolution-extra-visual-review-001';out.mkdir(exist_ok=False)
    source=json.loads((ROOT/'source-resolution-rectified-002/manifest.json').read_text())['rows']
    extra=json.loads((ROOT/'additional-frame-resolution-005/summary.json').read_text())
    rows=[r for r in extra['models']['low']['rows'] if r['divisor']==1]
    original={r['source_frame_index'] for r in source}
    additional={r['source_index'] for r in rows}
    assert len(original)==59 and len(additional)==24 and not original.intersection(additional)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    assets=ROOT/'viewer/native-resolution-assets/extra'
    for start in range(0,24,4):
        sheet=Image.new('RGB',(1008,1416),'#edf1f1');draw=ImageDraw.Draw(sheet)
        for offset,row in enumerate(rows[start:start+4]):
            stem=Path(row['frame']).stem
            for col,(folder,label) in enumerate([('gt','Real photo'),('low','Low-res training'),('full','Full-res training')]):
                photo=Image.open(assets/folder/(stem+'-head.png')).convert('RGB');photo.thumbnail((328,316))
                sheet.paste(photo,(col*336,offset*354+34))
                draw.text((col*336+3,offset*354+7),f'{row["source_index"]:06d} {label}',fill='#16343d',font=font)
        sheet.save(out/f'extra-heads-{start//4+1}.png')
    result=dict(status='prepared_pending_visual_review',created=time.time(),frames=24,source_index_overlap=[],
                original_frames=59,train_frames=51,additional_frames=24,same_video=True,
                all_frames_in_temporal_order=True,thumbnails_for_overview_only=True)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
