"""Independently check PNGs and preserve full, unenhanced comparisons."""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_street_joint_images import check_quantization,sha

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-capacity-components-022'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
CANDIDATE=LAB/'outputs/dynamic-human-street-local-capacity-020/adjacent_pair'

def pixels(p):return np.asarray(Image.open(p).convert('RGB'))

def main():
    assert (OUT/'completion.json').exists()
    assert not (OUT/'independent-image-audit.json').exists()
    p=json.loads((OUT/'protocol.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert sha(OUT/'diagnose_street_capacity_components.py')==p['script_sha256']
    assert sha(SOURCE/'checkpoint-refined.pt')==p['source_checkpoint_sha256']
    assert sha(CANDIDATE/'checkpoint-refined.pt')==p['candidate_checkpoint_sha256']
    rows=result['rows'];views={(r['frame'],r['camera']):r for r in p['views']}
    assert len(views)==15 and len(rows)==105
    assert {(r['mode'],r['frame'],r['camera']) for r in rows}=={(m,f,c) for m in p['modes'] for f,c in views}
    checked=[]
    for row in rows:
        f,c=row['frame'],row['camera'];mode=row['mode'];stem=f'frame{f:03d}-camera{c}'
        path=OUT/'renders'/mode/(stem+'-prediction.png');pred=pixels(path)
        target=np.asarray(Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((pred.shape[1],pred.shape[0]),Image.Resampling.BILINEAR))
        assert np.array_equal(target,pixels(OUT/'ground-truth'/(stem+'.png')))
        assert row['box']==views[(f,c)]['box'];x0,y0,x1,y1=row['box']
        crop=pred[y0:y1,x0:x1];gt=target[y0:y1,x0:x1]
        assert np.array_equal(crop,pixels(OUT/'renders'/mode/(stem+'-person79.png')))
        psnr,diff=check_quantization(gt,crop,row['psnr_db'])
        reproduced=None
        if mode in ('source','candidate'):
            folder=SOURCE if mode=='source' else CANDIDATE
            folder=folder/('evaluation/final' if row['heldout'] else 'training-evaluation/after')
            actual=int(np.abs(pred.astype(np.int16)-pixels(folder/(stem+'-prediction.png')).astype(np.int16)).max())
            assert actual==row['source_png_max_byte_difference'] and actual<=1;reproduced=actual
        checked.append(dict(mode=mode,frame=f,camera=c,sha256=sha(path),png_psnr_db=psnr,
            quantization_difference_db=diff,existing_png_max_byte_difference=reproduced))
    pairs=OUT/'paired-audit';pairs.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    groups={
        'geometry-color': [('observed','原始观测'),('source','源模型'),('color_only','仅换颜色'),('shape_opacity_only','仅换几何与不透明度'),('candidate','全部替换')],
        'color-opacity': [('observed','原始观测'),('candidate','全部替换'),('candidate_source_directional_color','恢复源方向相关颜色'),('candidate_source_dc_color','恢复源零阶颜色'),('candidate_source_opacity','恢复源不透明度')],
    }
    figures=[]
    for (f,c),roi in views.items():
        stem=f'frame{f:03d}-camera{c}';x0,y0,x1,y1=roi['box'];size=(x1-x0,y1-y0)
        scale=min(2.,250/size[0],600/size[1]);display=tuple(round(v*scale) for v in size)
        for group,items in groups.items():
            canvas=Image.new('RGB',(5*280,display[1]+80),'white');draw=ImageDraw.Draw(canvas)
            for col,(mode,label) in enumerate(items):
                raw=Image.open(OUT/'ground-truth'/(stem+'.png')).convert('RGB').crop(roi['box']) if mode=='observed' else Image.open(OUT/'renders'/mode/(stem+'-person79.png')).convert('RGB')
                draw.text((col*280+9,8),label,font=font,fill='#182d39')
                canvas.paste(raw.resize(display,Image.Resampling.NEAREST),(col*280+(280-display[0])//2,38))
            draw.text((12,canvas.height-31),stem+' · 参数替换诊断；未训练、未锐化',font=font,fill='#182d39')
            path=pairs/(stem+'-'+group+'.png');canvas.save(path)
            figures.append(dict(frame=f,camera=c,group=group,path=str(path.relative_to(OUT)),sha256=sha(path)))
    report=dict(status='all105full_pngs_and105_target_crops_verified',images=checked,figures=figures,
        original_gt_resize_exact=True,existing_source_candidate_reproduced=True,optimizer_updates=0,
        boundaries=['Independent PNG PSNR checked with quantization interval; SSIM remains the float renderer evaluator result.',
            'Thirty comparison figures generated; actual visual inspection must be recorded separately.'])
    (OUT/'independent-image-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'],len(checked),len(figures))

if __name__=='__main__':main()
