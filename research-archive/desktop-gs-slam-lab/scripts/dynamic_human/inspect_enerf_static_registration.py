"""Does the released empty reference register to current wall texture? Training only."""
from pathlib import Path
import cv2,json,numpy as np
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-matte-background-034'
cv2.setNumThreads(8);cv2.setRNGSeed(76);sift=cv2.SIFT_create(nfeatures=10000);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
reference=np.array(Image.open(out/'camera08-frame000750-background.png').convert('RGB'));band=np.zeros(reference.shape[:2],np.uint8);band[20:820,20:1900]=255
kb,db=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),band);xb=np.array([p.pt for p in kb]);rows=[]
for t in [740,750,760]:
    rgb=np.array(Image.open(out/f'camera08-frame{t:06d}-source.png').convert('RGB'));matte=np.array(Image.open(out/f'camera08-frame{t:06d}-matte.png').convert('L'))
    mask=cv2.erode(((matte<3)&(band>0)).astype(np.uint8),np.ones((11,11),np.uint8))*255
    ks,ds=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask);xs=np.array([p.pt for p in ks]);matcher=cv2.BFMatcher(cv2.NORM_L2)
    ab=matcher.knnMatch(db,ds,k=2);ba=matcher.knnMatch(ds,db,k=2);reverse={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.7*n.distance};matches=np.array([(m.queryIdx,m.trainIdx) for m,n in ab if m.distance<.7*n.distance and reverse.get(m.trainIdx)==m.queryIdx])
    assert len(matches)>100
    x=xb[matches[:,0]];y=xs[matches[:,1]];test=np.arange(len(x))%5==0
    H,inliers=cv2.findHomography(x[~test],y[~test],cv2.RANSAC,1.5,maxIters=10000,confidence=.999)
    assert H is not None
    transformed=cv2.perspectiveTransform(x[None].astype('float64'),H)[0]
    original=np.linalg.norm(x-y,axis=1);error=np.linalg.norm(transformed-y,axis=1)
    xy=np.array([[960,540],[100,100],[1800,100],[100,800],[1800,800]],np.float64);displacement=cv2.perspectiveTransform(xy[None],H)[0]-xy
    warped=cv2.warpPerspective(reference,H,(1920,1080));box=[80,200,480,600];x0,y0,x1,y1=box;panel=Image.new('RGB',(1200,444),'white');draw=ImageDraw.Draw(panel)
    for j,(im,label) in enumerate(zip([rgb,reference,warped],['Source RGB','Empty reference','2D registration diagnostic'])):
        panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(j*400,44));draw.text((j*400+6,12),label,font=font,fill='black')
    panel.save(out/f'camera08-frame{t:06d}-static-registration.png')
    np.savez_compressed(out/f'camera08-frame{t:06d}-static-correspondences.npz',reference_xy=x,source_xy=y,check=test,H=H)
    rows.append(dict(frame=t,matches=len(x),fit_matches=int((~test).sum()),fit_inliers=int(inliers.sum()),check_matches=int(test.sum()),check_identity_median_px=float(np.median(original[test])),check_H_median_px=float(np.median(error[test])),check_H_p90_px=float(np.quantile(error[test],.9)),H=H.tolist(),sample_locations=xy.tolist(),sample_displacements_px=displacement.tolist(),crop=box))
report=dict(status='training_camera_planar_registration_measured_not_applied_to_models',camera='08',rows=rows,
    method='SIFT mutual ratio.7, wall-band low-matte source pixels; deterministic80percent correspondences for RANSAC H,20percent geometric checks. Check points are from same images, not unseen image evaluation.',
    scope='2D wall-plane registration diagnostic, not a recovered camera trajectory or a proven cause of prior actor1_4 hand failure. Original images and model cameras unchanged.',
    limits='Shadows, repetitive textures, lens/plane approximations and appearance changes remain; geometric check alone is not model quality.')
(out/'static-registration.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({**report,'rows':[{k:v for k,v in r.items() if k not in ['H','sample_locations']} for r in rows]},indent=2))
