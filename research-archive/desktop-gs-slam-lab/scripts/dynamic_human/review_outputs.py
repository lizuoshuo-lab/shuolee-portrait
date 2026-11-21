"""Independent saved-PNG metric checks and unenhanced all-camera comparisons."""
from pathlib import Path
import json,math,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003'
DATA=LAB/'data/dynamic-human/dynamic3dgaussians/data/basketball'
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def psnr(a,b):return -10*math.log10(max(float(np.mean((a.astype(np.float64)/255-b.astype(np.float64)/255)**2)),1e-12))

def main():
    result=json.loads((OUT/'test-metrics.json').read_text());protocol=json.loads((OUT/'evaluation-protocol.json').read_text());rows=result['per_view']
    assert len(rows)==248 and len({(r['mode'],r['frame'],r['camera']) for r in rows})==248
    assert {r['camera'] for r in rows}=={0,10,15,30}
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',20);small=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',15)
    review=OUT/'review';review.mkdir(exist_ok=True);checks=[];bycam={};outputs=[]
    for camera in [0,10,15,30]:
        roi=protocol['fixed_human_rois_xyxy'][str(camera)];x0,y0,x1,y1=roi;rect=(slice(y0,y1),slice(x0,x1));w=x1-x0;h=y1-y0;panels=[]
        for t in protocol['frames']:
            gt=Image.open(DATA/f'ims/{camera}/{t:06d}.jpg').convert('RGB');images=[gt]
            for mode in ['static','dynamic']:
                pred=Image.open(OUT/f'renders/{mode}/{camera}/{t:04d}.png').convert('RGB');images.append(pred)
                r=next(r for r in rows if r['mode']==mode and r['camera']==camera and r['frame']==t)
                value=psnr(np.asarray(pred)[rect],np.asarray(gt)[rect]);difference=abs(value-r['person_region']['psnr'])
                checks.append(dict(camera=camera,frame=t,mode=mode,saved_png_person_psnr=value,absolute_difference_from_float_render_psnr=difference))
            panel=Image.new('RGB',(w*3,h+57),'white');draw=ImageDraw.Draw(panel)
            for j,(name,im) in enumerate(zip(['原始观测','静态表示','动态表示'],images)):
                panel.paste(im.crop(roi),(j*w,30));draw.text((j*w+w/2,3),name,font=font,anchor='mt',fill='#183747')
            draw.text((8,h+34),f'留出相机 {camera} · 帧 {t:03d} · 原生像素裁剪',font=small,fill='#183747');panels.append(panel)
        panels[0].save(review/f'camera{camera}-all31.gif',save_all=True,append_images=panels[1:],duration=80,loop=0,optimize=False)
        sheet=Image.new('RGB',(w*3,(h+57)*3),'white')
        for n,t in enumerate([0,15,30]):sheet.paste(panels[t],(0,n*(h+57)))
        sheet.save(review/f'camera{camera}-frames000-015-030.png')
        outputs += [f'review/camera{camera}-all31.gif',f'review/camera{camera}-frames000-015-030.png']
        bycam[str(camera)]={mode:{metric:float(np.mean([r['person_region'][metric] for r in rows if r['mode']==mode and r['camera']==camera])) for metric in ['psnr','ssim','reference_gradient_mae']} for mode in ['static','dynamic']}
    maxdifference=max(r['absolute_difference_from_float_render_psnr'] for r in checks)
    assert maxdifference<.08,f'PNG metric mismatch exceeds declared rounding tolerance: {maxdifference}'
    worst=min((r for r in rows if r['mode']=='dynamic'),key=lambda r:r['person_region']['psnr'])
    write(OUT/'saved-output-audit.json',dict(status='numeric_audit_passed_visual_review_pending',test_metrics_sha256=hashlib.sha256((OUT/'test-metrics.json').read_bytes()).hexdigest(),paired_views=124,roi_png_psnr_max_rounding_difference_db=maxdifference,quantization_tolerance_db=.08,per_camera=bycam,worst_dynamic_person_psnr_view=worst,comparisons=outputs,saved_png_checks=checks))
    print(json.dumps(dict(per_camera=bycam,worst_dynamic_view=worst,max_rounding_difference_db=maxdifference),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
