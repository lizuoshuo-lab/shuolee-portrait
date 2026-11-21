"""Read-only native-camera consistency: foreground and background at frame 750.

Mutual descriptor matching is selected before looking at calibrated epipolar
errors. No target heldout view, camera correction or dataset mutation is used.
"""
from pathlib import Path
import json, hashlib, shutil, time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from audit_enerf_input import read_cameras, digest

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
DATA = Path('/opt/gs-slam-lab/4k4d/actor5_6-mask039')
OUT = LAB/'outputs/dynamic-human-enerf-foreground-geometry-045'
OUT.mkdir(exist_ok=True)
assert not (OUT/'summary.json').exists(), 'Preserve completed diagnostics'
cv2.setNumThreads(8)
cams = read_cameras(DATA/'optimized')
names = [f'{i:02d}' for i in range(18) if i not in (0, 6, 12)]
centers = np.array([cams[c]['camera_center_world'] for c in names])
dist = np.linalg.norm(centers[:, None]-centers[None], axis=2)
pairs = sorted({tuple(sorted((names[i], names[j]))) for i in range(len(names))
                for j in np.argsort(dist[i])[1:4]})
protocol = dict(frame=750, cameras=names, pairs=pairs, resolution=[1920,1080],
    selection='Each training camera connected to three nearest camera centers; undirected union.',
    matching='SIFT maximum 6000 features per region; mutual nearest descriptor matches, Lowe ratio <0.7. No epipolar gate or estimated F.',
    regions='Foreground: candidate039 mask eroded by7x7; background: outside its41x41 dilation. Imperfect masks, not semantic truth.',
    score='Absolute signed Sampson residual in native pixels, from unchanged optimized K/R/T. Also unfiltered linear-triangulation reprojection and cheirality.',
    limits=['Descriptor matches require image identity review.', 'Foreground match support favors textured clothes over skin.',
            'Single-frame residual cannot isolate time offset, camera calibration, rolling shutter, or matching errors.',
            'This checks input consistency, not learned geometry quality or final human clarity.'], goal_complete=False)
def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
save(OUT/'protocol.json', protocol)
sift = cv2.SIFT_create(nfeatures=6000)
matcher = cv2.BFMatcher()
features, images, files = {}, {}, []
began = time.time()
for name in names:
    ip = DATA/'images'/name/'000750.png'; mp = DATA/'bgmtv2'/name/'000750.png'
    img = cv2.imread(str(ip)); mask = cv2.imread(str(mp), 0)
    assert img.shape == (1080,1920,3) and mask.shape == (1080,1920)
    images[name] = img[..., ::-1].copy()
    fg = cv2.erode((mask>=128).astype(np.uint8), np.ones((7,7),np.uint8))
    bg = 1-cv2.dilate((mask>=128).astype(np.uint8), np.ones((41,41),np.uint8))
    bg[:20] = 0; bg[-20:] = 0; bg[:,:20] = 0; bg[:,-20:] = 0
    for label, region in [('foreground', fg), ('background', bg)]:
        kp, desc = sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), region*255)
        features[name,label] = (np.array([k.pt for k in kp], np.float64),desc)
    files.append(dict(camera=name,image_sha256=digest(ip),mask_sha256=digest(mp)))
    print('FEATURES', name, len(features[name,'foreground'][0]),len(features[name,'background'][0]),flush=True)

def fundamental(a,b):
    A=np.array(cams[a]['w2c']); B=np.array(cams[b]['w2c'])
    R=B[:3,:3]@A[:3,:3].T; t=B[:3,3]-R@A[:3,3]
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    return np.linalg.inv(cams[b]['K']).T@cross@R@np.linalg.inv(cams[a]['K'])

rows=[]
for a,b in pairs:
    F=fundamental(a,b)
    P=np.array(cams[a]['K'])@np.array(cams[a]['w2c'])[:3]
    Q=np.array(cams[b]['K'])@np.array(cams[b]['w2c'])[:3]
    for region in ('foreground','background'):
        x,dx=features[a,region]; y,dy=features[b,region]
        if dx is None or dy is None or min(len(dx),len(dy))<2: continue
        fw={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dx,dy,k=2) if m.distance<.7*n.distance}
        bw={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dy,dx,k=2) if m.distance<.7*n.distance}
        ij=np.array([(i,j) for i,j in fw.items() if bw.get(j)==i],dtype=int)
        if len(ij)<4: continue
        xx=x[ij[:,0]]; yy=y[ij[:,1]]
        hx=np.c_[xx,np.ones(len(xx))]; hy=np.c_[yy,np.ones(len(yy))]
        ly=hx@F.T; lx=hy@F
        numerator=np.sum(hy*ly,axis=1)
        residual=numerator/np.sqrt((lx[:,:2]**2).sum(1)+(ly[:,:2]**2).sum(1))
        X=cv2.triangulatePoints(P,Q,xx.T,yy.T).T; X=X/X[:,3:4]
        pa=X@P.T; pb=X@Q.T
        reproj=np.stack([np.linalg.norm(pa[:,:2]/pa[:,2:3]-xx,axis=1),
                          np.linalg.norm(pb[:,:2]/pb[:,2:3]-yy,axis=1)],1)
        positive=(pa[:,2]>0)&(pb[:,2]>0)
        np.savez_compressed(OUT/f'{region}-{a}-{b}.npz',xy_a=xx,xy_b=yy,F=F,P=P,Q=Q,
                            signed_sampson=residual,world_points=X[:,:3],reprojection=reproj,positive=positive)
        row=dict(a=a,b=b,region=region,matches=len(xx),median_px=float(np.median(abs(residual))),
                 p90_px=float(np.quantile(abs(residual),.9)),below1px=float(np.mean(abs(residual)<1)),
                 positive_fraction=float(positive.mean()),median_reprojection_px=float(np.median(reproj)))
        rows.append(row); print('PAIR',json.dumps(row),flush=True)

# Deterministic native patch sheets: three smallest and three largest foreground
# median errors among adequately covered pairs. Match identities remain visible.
eligible=sorted([r for r in rows if r['region']=='foreground' and r['matches']>=20],key=lambda r:r['median_px'])
selected=eligible[:3]+[r for r in eligible[-3:] if r not in eligible[:3]]
panels=[]
def patch(img,xy):
    x,y=np.rint(xy).astype(int); z=np.full((120,120,3),235,np.uint8)
    x0,y0=max(x-60,0),max(y-60,0); x1,y1=min(x+60,1920),min(y+60,1080)
    z[y0-y+60:y1-y+60,x0-x+60:x1-x+60]=img[y0:y1,x0:x1]
    return z
for r in selected:
    a,b=r['a'],r['b']; z=np.load(OUT/f'foreground-{a}-{b}.npz')
    order=np.argsort(abs(z['signed_sampson'])); ids=order[np.rint(np.linspace(0,len(order)-1,12)).astype(int)]
    panel=Image.new('RGB',(4*250,3*156),'white'); draw=ImageDraw.Draw(panel)
    for n,k in enumerate(ids):
        tx=(n%4)*250; ty=(n//4)*156
        draw.text((tx+3,ty+3),f'{a}/{b} match{k}: {abs(z["signed_sampson"][k]):.2f}px',fill='black')
        for side,name in enumerate((a,b)):
            xy=z['xy_a' if side==0 else 'xy_b'][k]
            panel.paste(Image.fromarray(patch(images[name],xy)),(tx+side*124,ty+27))
            # Thin crosshair only; source patch pixels otherwise unchanged.
            cx=tx+side*124+60; cy=ty+27+60
            draw.line((cx-4,cy,cx+4,cy),fill='red',width=1)
            draw.line((cx,cy-4,cx,cy+4),fill='red',width=1)
    fn=f'foreground-{a}-{b}-native-match-patches.png'; panel.save(OUT/fn)
    panels.append(dict(file=fn,indices=ids.tolist(),sha256=digest(OUT/fn)))
summary=dict(status='native_input_epipolar_check_complete_identity_review_pending',records=rows,files=files,
             panels=panels,elapsed_seconds=time.time()-began,goal_complete=False)
save(OUT/'summary.json',summary)
shutil.copy2(__file__,OUT/Path(__file__).name)
print('DONE',len(rows),'pair-region records',flush=True)
