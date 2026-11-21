"""Native, training-only local flow with explicit ambiguity rejection."""
from pathlib import Path
import json,hashlib,time,shutil,argparse
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('--engine',choices=['dis','raft'],default='dis');args=ap.parse_args()
L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-enerf-hand-motion-082';BASE=L/'outputs/dynamic-human-enerf-open-hand-flow-083';O=BASE if args.engine=='dis' else BASE/'raft'
O.mkdir(exist_ok=True);(O/'pairs').mkdir(exist_ok=True);(O/'review').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'preparation.json').exists()
poses=read(H/'observations.json')['records'];by={(r['camera'],r['frame']):r for r in poses};cams=sorted({r['camera'] for r in poses});assert not set(cams)&{'00','06','12'}
save(O/'protocol.json',dict(status='fixed_before_flow_candidates',scene='actor5_6',frames=[728,730,732],pairs=[[728,730],[730,728],[730,732],[732,730]],cameras=cams,
    crop='Union of both right-hand candidate bounding rectangles plus96px native context. Keypoints only locate rectangular window; no anatomical constraint or visibility claim.',
    source='Unchanged079native RGB; even-time training only; no00/06/12oroddimages.',
    method=('OpenCV DIS medium preset, native grayscale' if args.engine=='dis' else 'Official torchvision RAFT-Large weightsC_T_SKHT_V2,float32,20iterations,nativeRGB'),flow_engine=args.engine,
    gates=dict(forward_backward_px=.5,patch_size=9,min_patch_std_u8=5,min_ncc=.97,max_mae_u8=12,max_motion_px=40,destination_matte_min=.9,uniqueness_radius_px=20,second_peak_exclusion_radius_px=3,min_peak_gap=.02,max_unique_peak_distance_px=1.5),
    note='Onlycorrespondence hypotheses; clothing/toys insidewindow must be visually reviewed. Frozen gates before model comparison; no Gaussian training until visual review.',goal_complete=False))
cv2.setNumThreads(4);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);rows=[];start=time.monotonic()
def sample(a,p):return cv2.remap(a.astype(np.float32),p[:,0,None].astype(np.float32),p[:,1,None].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(len(p),-1)
def hb(r):
    p=np.array(r['keypoints']);return np.r_[p.min(0)-8,p.max(0)+8]
for c in cams:
    data={}
    for t in [728,730,732]:
        r=by[c,t];p=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(p)==r['input_sha256']
        with np.load(p) as z:data[t]={k:z[k] for k in ['rgb','matte','valid']}
        data[t]['gray']=cv2.cvtColor(data[t]['rgb'],cv2.COLOR_RGB2GRAY)
    sheet=Image.new('RGB',(1050,810),'white');draw=ImageDraw.Draw(sheet);draw.text((8,4),f'Camera{c}: original source / source+accepted pixels / target+correspondence / target warped to source',font=font,fill='black')
    count=0
    for first,last in [(728,730),(730,732)]:
        boxes=np.array([hb(by[c,t]) for t in [first,last]]);lo=np.floor(boxes[:,:2].min(0)-96).astype(int);hi=np.ceil(boxes[:,2:].max(0)+96).astype(int);lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,[1920,1080]);crop=[*lo,*hi]
        if args.engine=='raft':
            path=BASE/'raft-fields'/f'camera{c}-frames{first}-{last}.npz';record=next(r for r in read(BASE/'raft-fields/completion.json')['records'] if r['camera']==c and r['source_frame']==first)
            assert sha(path)==record['sha256']
            with np.load(path) as q:crop=q['context_crop'].tolist();lo=np.array(crop[:2]);hi=np.array(crop[2:]);flows=[q['forward'],q['backward']]
        else:
            gs=[np.ascontiguousarray(data[t]['gray'][lo[1]:hi[1],lo[0]:hi[0]]) for t in [first,last]]
            flows=[cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gs[i],gs[1-i],None) for i in [0,1]]
        for direction in [0,1]:
            a,b=([first,last] if direction==0 else [last,first]);F,B=flows[direction],flows[1-direction]
            box=hb(by[c,a]);xx,yy=np.meshgrid(np.arange(np.ceil(box[0]),np.floor(box[2])+1,2),np.arange(np.ceil(box[1]),np.floor(box[3])+1,2));xy=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float32)
            flow=sample(F,xy-lo);dest=xy+flow;back=sample(B,dest-lo);fb=np.linalg.norm(flow+back,axis=-1)
            candidate=(fb<.5)&(np.linalg.norm(flow,axis=1)<40)&(sample(data[a]['matte'],xy)[:,0]>.9)&(sample(data[b]['matte'],dest)[:,0]>.9)&(sample(data[a]['valid'],xy)[:,0]>.999)&(sample(data[b]['valid'],dest)[:,0]>.999)
            ncc=np.full(len(xy),-1.);mae=np.full(len(xy),255.);gap=np.full(len(xy),-1.);distance=np.full(len(xy),1e6);std=np.zeros(len(xy));accepted=np.zeros(len(xy),bool)
            for i in np.flatnonzero(candidate):
                p=cv2.getRectSubPix(data[a]['gray'],(9,9),tuple(map(float,xy[i]))).astype(np.float32);q=cv2.getRectSubPix(data[b]['gray'],(9,9),tuple(map(float,dest[i]))).astype(np.float32)
                std[i]=p.std();pp=p-p.mean();qq=q-q.mean();den=np.linalg.norm(pp)*np.linalg.norm(qq);ncc[i]=float(np.sum(pp*qq)/den) if den>1e-5 else -1;mae[i]=float(abs(p-q).mean())
                if std[i]<5 or ncc[i]<.97 or mae[i]>12:continue
                search=cv2.getRectSubPix(data[b]['gray'],(49,49),tuple(map(float,dest[i]))).astype(np.float32);scores=cv2.matchTemplate(search,p,cv2.TM_CCOEFF_NORMED);peak=np.unravel_index(np.argmax(scores),scores.shape);distance[i]=np.linalg.norm(np.array(peak)-20);gy,gx=np.ogrid[:41,:41];other=(gy-peak[0])**2+(gx-peak[1])**2>9;gap[i]=float(scores[peak]-scores[other].max())
                accepted[i]=bool(distance[i]<=1.5 and gap[i]>=.02)
            stem=f'camera{c}-frame{a:06d}-to{b:06d}';out=O/'pairs'/f'{stem}.npz'
            np.savez_compressed(out,xy=xy,destination=dest,flow=flow,accepted=accepted,fb=fb,ncc=ncc,mae=mae,std=std,peak_gap=gap,peak_distance=distance,hand_box=box,context_crop=crop)
            row=dict(camera=c,source_frame=a,target_frame=b,source_sha256=by[c,a]['input_sha256'],target_sha256=by[c,b]['input_sha256'],path=f'pairs/{stem}.npz',sha256=sha(out),seeds=len(xy),fb_matte_candidates=int(candidate.sum()),accepted=int(accepted.sum()));rows.append(row)
            center=(box[:2]+box[2:])/2;x0,y0=(center-80).round().astype(int);panelcrop=[x0,y0,x0+160,y0+160];raw=Image.fromarray(data[a]['rgb']).crop(panelcrop);src=raw.copy();target=Image.fromarray(data[b]['rgb']).crop(panelcrop);sd=ImageDraw.Draw(src);td=ImageDraw.Draw(target)
            for i in np.flatnonzero(accepted):
                for dd,point in [(sd,xy[i]),(td,dest[i])]:x,y=point-[x0,y0];dd.ellipse((x-.7,y-.7,x+.7,y+.7),fill=(0,255,60))
            xx,yy=np.meshgrid(np.arange(x0,x0+160),np.arange(y0,y0+160));grid=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float32);ff=sample(F,grid-lo);warp=sample(data[b]['rgb'],grid+ff).reshape(160,160,3).round().clip(0,255).astype(np.uint8)
            py=46+count*190;draw.text((8,py-18),f'{a} -> {b} | {accepted.sum()}/{len(xy)} accepted | raw photographs, no RGB reconstruction',font=font,fill='black')
            for k,im in enumerate([raw,src,target,Image.fromarray(warp)]):sheet.paste(im,(8+k*170,py))
            count+=1
    sheet.crop((0,0,690,810)).save(O/'review'/f'camera{c}-flow.png');print('OPEN_HAND_FLOW',c,sum(r['accepted'] for r in rows if r['camera']==c),flush=True)
save(O/'preparation.json',dict(status='60native_training_flow_pairs_ready_for_visual_review',records=rows,total_accepted=sum(r['accepted'] for r in rows),total_seeds=sum(r['seeds'] for r in rows),elapsed_seconds=time.monotonic()-start,training_approved=False,goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print('FLOW_PREPARATION_COMPLETE',sum(r['accepted'] for r in rows),flush=True)
