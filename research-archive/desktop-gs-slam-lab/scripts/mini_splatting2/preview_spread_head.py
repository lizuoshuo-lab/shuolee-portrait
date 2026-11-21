"""CPU-only first-candidate preview; preserves the final four-model audit."""
import ast
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_head_detail import metric

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    out=ROOT/'native-spread003-head-preview-001';out.mkdir(exist_ok=False)
    control=ROOT/'native-depthspread000-18000-s0';candidate=ROOT/'native-depthspread003-18000-s0'
    result=json.loads((control/'evaluation-summary.json').read_text())
    assert json.loads((candidate/'evaluation-summary.json').read_text())['status']=='complete'
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);rows=[]
    for i,row in enumerate(result['native_rows']):
        relative=Path('model/test/ours_18000');filename=f'{i:05d}.png'
        assert (control/relative/'gt'/filename).read_bytes()==(candidate/relative/'gt'/filename).read_bytes()
        images=[Image.open(control/relative/'gt'/filename).convert('RGB'),
            Image.open(control/relative/'renders'/filename).convert('RGB'),
            Image.open(candidate/relative/'renders'/filename).convert('RGB')]
        w,h=images[0].size;x0,y0,x1,y1=[round(a*b) for a,b in zip(boxes[int(row['frame'])],[w,h,w,h])]
        y1=y0+round((y1-y0)/3);crops=[im.crop((x0,y0,x1,y1)) for im in images]
        arrays=[np.asarray(im,dtype='float32')/255 for im in crops]
        values=[metric(arr,arrays[0]) for arr in arrays[1:]]
        diff={key:values[1][key]-values[0][key] for key in values[0]}
        rows.append(dict(frame=row['frame'],control=values[0],spread003=values[1],change=diff))
        cw,ch=crops[0].size;panel=Image.new('RGB',(cw*3,ch+38),'white');draw=ImageDraw.Draw(panel)
        for j,(title,crop) in enumerate(zip(['Real '+row['frame'],'Control 0','Spread 0.003'],crops)):
            panel.paste(crop,(j*cw,38));draw.text((j*cw+8,9),title,fill='#173d47',font=font)
        panel.save(out/(row['frame']+'.png'))
    aggregate={key:statistics.mean(r['change'][key] for r in rows) for key in rows[0]['change']}
    data=dict(status='complete_partial_stage_preview',candidate='native-depthspread003-18000-s0',
        control='native-depthspread000-18000-s0',rows=rows,paired_head_change=aggregate,
        accepted=False,other_strength_pending=True,scope='Same8 development views; one seed; visual preview without geometry acceptance')
    (out/'summary.json').write_text(json.dumps(data,indent=2));print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
