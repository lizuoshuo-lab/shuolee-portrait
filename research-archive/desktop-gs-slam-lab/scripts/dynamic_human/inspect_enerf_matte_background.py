"""Training-only source/matte/background evidence; no automatic mask cleanup."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from audit_enerf_input import read_cameras,digest
from enerf_background_data import load_background

LAB=Path(__file__).resolve().parents[2];root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');out=LAB/'outputs/dynamic-human-enerf-matte-background-034';out.mkdir(exist_ok=True)
assert not (out/'analysis.json').exists(), 'Keep completed inspection'
protocol=dict(status='fixed_before_training_image_diagnostics',camera='08',frames=[740,750,760],heldout_images_read=0,
    scope='Same already selected training camera/window as031. Source/reference comparison, not a revised mask or trained human result.',
    crop=[520,400,950,900],crop_selection='Fixed region spanning wall shadow and left person contact area, chosen from training frame750 source RGB.',
    annotation=dict(frame=750,box=[650,500,720,550],meaning='Visually selected wall surface patch under cast shadow, away from person silhouette; small illustrative region, not full semantic labels'),
    photometric_fit='Per-channel affine empty-background reference fit on valid matte<.01 pixels, evenly subsampled max50000, 4iterations keeping80percent smallest channel residuals; gains clipped[.7,1.3], biases[-.1,.1]',
    residual_test='RGB mean absolute residual to photometrically adjusted empty reference;95percentile in low-matte region is descriptive threshold, not a ground-truth segmentation rule',
    preservation='Source RGB/masks untouched; no generated, inpainted or sharpened outputs')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
bg=load_background(root,['08'])[0];c=read_cameras(root/'optimized')['08'];K=np.array(c['K']);D=np.array(c['D']);mx,my=cv2.initUndistortRectifyMap(K,D,np.eye(3),K,(1920,1080),cv2.CV_32FC1);valid=bg['valid'];B=bg['rgb'].astype(np.float64)/255
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);records=[]
for t in protocol['frames']:
    image_path=root/'images/08'/f'{t:06d}.jpg';mask_path=root/'bgmtv2/08'/f'{t:06d}.jpg'
    rgb=cv2.cvtColor(cv2.remap(cv2.imread(str(image_path)),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT),cv2.COLOR_BGR2RGB)
    matte=cv2.remap(cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float64)/255
    I=rgb.astype(np.float64)/255; indices=np.flatnonzero(valid&(matte<.01));indices=indices[np.linspace(0,len(indices)-1,min(50000,len(indices))).astype(int)]
    adjusted=np.empty_like(B);fits=[]
    for channel in range(3):
        x=B[...,channel].ravel()[indices];y=I[...,channel].ravel()[indices];A=np.c_[x,np.ones(len(x))];keep=np.ones(len(x),bool)
        for _ in range(4):
            beta=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0];beta=np.clip(beta,[.7,-.1],[1.3,.1]);res=np.abs(A@beta-y);keep=res<=np.quantile(res,.8)
        adjusted[...,channel]=B[...,channel]*beta[0]+beta[1];fits.append(beta.tolist())
    residual=np.abs(I-adjusted).mean(2);threshold=float(np.quantile(residual[valid&(matte<.01)],.95));soft=valid&(matte>.05)&(matte<.95)
    composite=np.rint((I*matte[...,None]+.5*(1-matte[...,None]))*255).clip(0,255).astype('uint8')
    values=[rgb,bg['rgb'],np.repeat(np.rint(matte*255).astype('uint8')[...,None],3,axis=2),composite]
    labels=['Source RGB','Empty reference','Released matte','Matte composite']
    for name,value in zip(['source','background','matte','foreground'],values):Image.fromarray(value).save(out/f'camera08-frame{t:06d}-{name}.png')
    x0,y0,x1,y1=protocol['crop'];w=x1-x0;h=y1-y0;panel=Image.new('RGB',(w*4,h+44),'white');draw=ImageDraw.Draw(panel)
    for j,(value,label) in enumerate(zip(values,labels)):
        panel.paste(Image.fromarray(value[y0:y1,x0:x1]),(j*w,44));draw.text((j*w+8,12),label,font=font,fill='black')
    panel.save(out/f'camera08-frame{t:06d}-wall-matte-comparison.png')
    row=dict(frame=t,source_sha256=digest(image_path),mask_sha256=digest(mask_path),reference_sha256=bg['sha256'],affine_RGB_gain_bias=fits,low_matte_residual95=threshold,soft_matte_pixels=int(soft.sum()),soft_pixels_background_compatible=int((soft&(residual<=threshold)).sum()),soft_pixels_above_reference_residual=int((soft&(residual>threshold)).sum()))
    if t==750:
        a,b,d,e=protocol['annotation']['box'];m=matte[b:e,a:d];r=residual[b:e,a:d]
        row['illustrative_wall_region']=dict(box=[a,b,d,e],matte_mean=float(m.mean()),matte_above_point1_fraction=float((m>.1).mean()),rgb_mae_to_adjusted_empty_reference=float(r.mean()),interpretation='Released matte gives a nonzero foreground weight to this observed wall/shadow patch. This is not a human surface or proof that every low-confidence pixel is wrong.')
    records.append(row)
report=dict(status='training_source_comparisons_written_visual_review_pending',protocol='protocol.json',records=records,
    cautions=['Real cast shadows change RGB on static surfaces; raw background differences cannot be used directly as body occupancy.','A wall-like color can also belong to clothing; no masks automatically removed or relabeled.','Photometric gain/bias is only a source diagnostic and was not applied to background training or evaluation.'])
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report,indent=2))
