"""Reuse the frozen photometric localizer once on the second distinct pigment."""
from pathlib import Path
from datetime import datetime,timezone
import time,json
import numpy as np
from PIL import Image,ImageDraw
from localize_grayT123_skin_spot import fit_patch
from audit_grayT123_camera_support import REC,ROOT,read,save,sha,native,stats
OUT=REC/'grayT-temple-validation-review'
def main():
    start=time.monotonic()
    if (OUT/'localization-started.json').exists():raise RuntimeError('One localization run already started; inspect evidence.')
    tr=read(OUT/'tracking-result.json');plan=read(OUT/'plan.json');assert tr['algorithmic_checks_passed']
    rows=read(ROOT/'data/training-inputs.json')['records'];helper=Path(__file__).with_name('localize_grayT123_skin_spot.py')
    save(OUT/'localization-started.json',dict(created=datetime.now(timezone.utc).isoformat(),helper_sha256=sha(helper),
        tracking_sha256=sha(OUT/'tracking-result.json'),plan_sha256=sha(OUT/'plan.json')))
    out=[];fail=[];raw=Image.new('RGB',(512,384),(245,245,245));marked=raw.copy()
    for n,o in enumerate(tr['observations']):
        old=np.array(o['xy']);center=np.rint(old).astype(int);cx,cy=map(int,center);row=rows[o['frame_number']-1]
        with Image.open(native(row['image'])) as im:
            im=im.convert('RGB');small=np.array(im.crop((cx-6,cy-6,cx+7,cy+7)),float)/255;tile=im.crop((cx-32,cy-32,cx+32,cy+32))
        rr=fit_patch(small@np.array([.299,.587,.114]),old-center);new=rr.pop('center')+center;rr['parameters']=rr['parameters'].tolist()
        entry=dict(frame=o['frame'],frame_number=o['frame_number'],role=o['role'],saved_LK_xy=old.tolist(),xy=new.tolist(),
            displacement_px=float(np.linalg.norm(new-old)),**rr)
        if not rr['success'] or rr['bound_indices']:fail.append(dict(frame=o['frame'],reason='solver_or_photometric_parameter_bound',bounds=rr['bound_indices']))
        out.append(entry);raw.paste(tile,((n%8)*64,(n//8)*64));tile.save(OUT/f'frame_{o["frame_number"]:03d}_localized_native.png')
        draw=ImageDraw.Draw(tile)
        for xy,color in [(old,(0,255,255)),(new,(255,176,0))]:
            x,y=xy-(center-32);draw.line((x-2,y,x+2,y),fill=color);draw.line((x,y-2,x,y+2),fill=color)
        marked.paste(tile,((n%8)*64,(n//8)*64))
        if time.monotonic()-start>120:raise RuntimeError('Localizer budget exhausted; no repeat.')
    raw.save(OUT/'all41_localized_native.png');marked.save(OUT/'all41_localized_locator.png')
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,observations=out,failed_frames=fail,
        algorithmic_checks_passed=not fail,center_changes_px=stats([o['displacement_px'] for o in out]),
        conditional95=stats([o['conditional_center_95_radius_px'] for o in out]),conditional_uncertainty_not_truth=True,quality_accepted=False)
    save(OUT/'localization-result.json',result)
    save(OUT/'index.json',dict(images=[dict(frame='40..80, chronological,8columns',region='第二处颞部色素斑：全部41帧原生64px',
        file='all41_localized_native.png',raw='all41_localized_native.png',locator='all41_localized_locator.png',columns=['cyan LK / orange intensity center'])]))
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},ensure_ascii=False))
if __name__=='__main__':main()
