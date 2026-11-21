"""Rebuild selected real-image previews with floating-point alpha composition."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_enerf_input import read_cameras,digest
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-performer-selection-031';report=json.loads((out/'selection.json').read_text());root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);checks=[]
for seq in report['results']:
    data=root/seq['dataset'];c=read_cameras(data/'optimized')['08'];K=np.array(c['K']);D=np.array(c['D']);maps=None;lookup={r['frame']:r for r in seq['source_files']}
    for selected in seq['selected']:
        t=selected['center'];canvas=Image.new('RGB',(1920,404),'white');draw=ImageDraw.Draw(canvas)
        for j,frame in enumerate([t-10,t,t+10]):
            row=lookup[frame];path=data/row['image_path'];mask=data/row['mask_path'];assert digest(path)==row['sha256'] and digest(mask)==row['mask_sha256'];rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);matte=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if maps is None:maps=cv2.initUndistortRectifyMap(K,D,None,K,(rgb.shape[1],rgb.shape[0]),cv2.CV_32FC1)
            rgb=cv2.remap(rgb,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);matte=cv2.remap(matte,*maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            alpha=matte.astype('float64')[...,None]/255;pixels=np.rint(rgb.astype('float64')*alpha+127.5*(1-alpha)).clip(0,255).astype('uint8')
            im=Image.fromarray(pixels);name=f'{seq["dataset"]}-camera08-frame{frame:06d}-foreground.png';im.save(out/name);assert np.array_equal(np.array(Image.open(out/name)),pixels)
            canvas.paste(im.resize((640,360),Image.Resampling.LANCZOS),(j*640,44));draw.text((j*640+8,12),f'{seq["dataset"]} | training camera08 | frame{frame}',font=font,fill='black');checks.append(dict(dataset=seq['dataset'],camera='08',frame=frame,native_preview=name,source_sha256=row['sha256'],saved_pixels_exact=True))
        canvas.save(out/selected['contact_sheet'])
(out/'preview-pixel-audit.json').write_text(json.dumps(dict(status='all18native_source_composites_and6contact_sheets_rebuilt',checks=checks,correction='Initial display-only RGBxmatte uint8 multiplication overflowed; rejected previews and source preserved in preview-uint8-overflow-rejected. Motion statistics used original uint8 images without that product and were unaffected. Corrected composition uses float64 before multiplication.',scope='Only training camera08. No heldout appearance images opened; not model results.'),indent=2)+'\n',encoding='utf8')
print('REBUILT',len(checks),'source previews')
