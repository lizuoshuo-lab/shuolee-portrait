"""Two-way depth reprojection on the bounded DA3 pilot and original DA2 maps."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shuolee_geometry_diagnosis import BASE, ROOT, OUT, sha, write_json


def warp_check(di,dj,ci,cj,ri,rj,wi,wj):
    h,w=di.shape; yy,xx=np.mgrid[:h,:w].astype('float32')
    ki,kj=ci['K'],cj['K']; vi,vj=ci['V'],cj['V']
    p=np.stack([(xx-ki[0,2])/ki[0,0]*di,(yy-ki[1,2])/ki[1,1]*di,di],-1)
    world=(p-vi[:3,3])@vi[:3,:3]
    q=world@vj[:3,:3].T+vj[:3,3]
    ux=(q[...,0]/np.maximum(q[...,2],1e-6)*kj[0,0]+kj[0,2]).astype('float32')
    uy=(q[...,1]/np.maximum(q[...,2],1e-6)*kj[1,1]+kj[1,2]).astype('float32')
    observed=cv2.remap(dj,ux,uy,cv2.INTER_LINEAR)
    weight=cv2.remap(wj,ux,uy,cv2.INTER_LINEAR)
    color=cv2.remap(rj,ux,uy,cv2.INTER_LINEAR)
    valid=(wi>.25)&(weight>.25)&(di>.2)&(q[...,2]>.2)&(observed>.2)&(ux>=1)&(uy>=1)&(ux<w-2)&(uy<h-2)
    rel=abs(q[...,2]-observed)/np.maximum(observed,1e-6)
    target=np.stack([(ux-kj[0,2])/kj[0,0]*observed,(uy-kj[1,2])/kj[1,1]*observed,observed],-1)
    back=((target-vj[:3,3])@vj[:3,:3])@vi[:3,:3].T+vi[:3,3]
    bx=back[...,0]/np.maximum(back[...,2],1e-6)*ki[0,0]+ki[0,2]
    by=back[...,1]/np.maximum(back[...,2],1e-6)*ki[1,1]+ki[1,2]
    cycle=np.sqrt((bx-xx)**2+(by-yy)**2)
    photometric=abs(color-ri).mean(-1)
    occluded=valid&(q[...,2]>observed*1.05)
    accepted=valid&(~occluded)&(rel<.10)&(cycle<2)&(photometric<.15)
    return valid,occluded,accepted


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-training',action='store_true')
    args=parser.parse_args()
    prefix='dense-consistency-full' if args.all_training else 'dense-consistency'
    prediction='da3-full-predictions.npz' if args.all_training else 'da3-predictions.npz'
    stride=3 if args.all_training else 1
    assert not (OUT/(prefix+'.json')).exists()
    cv2.setNumThreads(2)
    with np.load(OUT/prediction) as cache:
        da3=cache['depth'].astype('float32');K=cache['intrinsics'];V=cache['extrinsics'];names=cache['frames'].tolist()
    h,w=da3.shape[1:]
    camjson={c['img_name']:c for c in json.loads((BASE/'viewer/cameras.json').read_text())}
    head_boxes={r['frame']:r['box'] for r in json.loads((ROOT/'training-head-regions-003/manifest.json').read_text())['rows']}
    cams=[];images=[];weights=[];heads=[];da2=[]
    for i,name in enumerate(names):
        c=camjson[name]
        expected_k=np.array([[c['fx']*w/c['width'],0,w/2],[0,c['fy']*h/c['height'],h/2],[0,0,1]])
        assert np.allclose(K[i],expected_k,rtol=1e-5,atol=1e-4),'Unexpected DA3 crop or intrinsics: remap inputs before comparison'
        original=np.asarray(Image.open(BASE/f'hover-sfm-001/dataset/images/{name}.jpg').convert('RGB'))
        images.append(cv2.resize(original,(w,h),interpolation=cv2.INTER_AREA).astype('float32')/255)
        with np.load(ROOT/f'depth-priors/{name}.npz') as prior:
            inv=cv2.resize(prior['inverse_depth'].astype('float32'),(w,h),interpolation=cv2.INTER_LINEAR)
            conf=cv2.resize(prior['weight'].astype('float32'),(w,h),interpolation=cv2.INTER_LINEAR)
        da2.append(1/np.maximum(inv,1e-6))
        motion=np.asarray(Image.open(ROOT/f'motion-masks-001/{name}.png').convert('L'))
        conf*=cv2.resize((motion==0).astype('float32'),(w,h),interpolation=cv2.INTER_NEAREST)
        weights.append(conf)
        box=np.array(head_boxes[name])*np.array([w/c['width'],h/c['height'],w/c['width'],h/c['height']])
        x0,y0,x1,y1=np.round(box).astype(int);head=np.zeros((h,w),bool);head[y0:y1,x0:x1]=True;heads.append(head)
        cams.append(dict(K=K[i],V=V[i]))
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_sha256=sha(Path(__file__)),
        prediction_file=prediction,prediction_sha256=sha(OUT/prediction),resolution=[w,h],
        neighbors=f'previous/next {stride} selected training indices',neighbor_stride=stride,
        common_confidence='Original DA2 confidence>0.25 and same motion masks for both methods; no DA3 confidence cherry-picking.',
        occlusion='projected depth more than5percent behind target depth, reported separately',
        acceptance='relative depth<0.10, forward/back pixel distance<2px, RGB L1<0.15, nonoccluded; accepted by at least one neighbor',
        geometry_GT=False,scope='Same-image multi-view self-consistency at low resolution. Common-confidence domain excludes unsupported pixels. Not a rendering-quality result.')
    write_json(OUT/(prefix+'-protocol.json'),protocol)
    results={};masks={}
    for method,depths in [('da2',np.array(da2)),('da3',da3)]:
        rows=[];masks[method]=[]
        for i,name in enumerate(names):
            accepted=np.zeros((h,w),bool);tested=np.zeros((h,w),bool);occluded=np.zeros((h,w),bool)
            for j in [i-stride,i+stride]:
                if not 0<=j<len(names):continue
                va,oc,ac=warp_check(depths[i],depths[j],cams[i],cams[j],images[i],images[j],weights[i],weights[j])
                tested|=va;occluded|=oc;accepted|=ac
            eligible=weights[i]>.25;masks[method].append(accepted)
            rows.append(dict(frame=name,**{label:dict(eligible_pixels=int((mask&eligible).sum()),
                accepted_pixels=int((accepted&mask).sum()),
                accepted_over_eligible=float((accepted&mask).sum()/max((eligible&mask).sum(),1)),
                tested_over_eligible=float((tested&mask).sum()/max((eligible&mask).sum(),1)),
                any_occluded_over_eligible=float((occluded&mask).sum()/max((eligible&mask).sum(),1)))
                for label,mask in [('head_crop',heads[i]),('outside_head_crop',~heads[i])]}))
        results[method]=dict(rows=rows,aggregate={label:dict(
            accepted_over_eligible=sum(r[label]['accepted_pixels'] for r in rows)/sum(r[label]['eligible_pixels'] for r in rows),
            eligible_pixels=sum(r[label]['eligible_pixels'] for r in rows)) for label in ['head_crop','outside_head_crop']})
    # Actual estimated depth maps on one shared scale, not a hand-drawn schematic.
    for name in ['000014','000042']:
        i=names.index(name); tiles=[]
        for title,im in [('Input photograph',np.uint8(images[i]*255)),
            ('DA2 depth (SfM scale)',cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip((da2[i]-1)/9,0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)),
            ('DA3 depth (same scale)',cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip((da3[i]-1)/9,0,1)*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB))]:
            tile=Image.new('RGB',(w,h+40),'white');tile.paste(Image.fromarray(im),(0,40));ImageDraw.Draw(tile).text((6,12),title,fill='black');tiles.append(tile)
        panel=Image.new('RGB',(3*w,h+64),'white')
        for j,tile in enumerate(tiles):panel.paste(tile,(j*w,0))
        ImageDraw.Draw(panel).text((6,h+44),'Shared depth scale: 1 to 10 (arbitrary SfM units); farther values clipped. Blue: near; red: far.',fill='black')
        panel.save(OUT/f'{prefix}-depth-{name}.png')
    if args.all_training:
        np.savez_compressed(OUT/(prefix+'-masks.npz'),da2=np.array(masks['da2']),da3=np.array(masks['da3']),frames=np.array(names),common_weight=np.array(weights))
    write_json(OUT/(prefix+'.json'),dict(status='complete',protocol=protocol,models=results,models_modified=False,
        prior_ready_for_training=False,next='Evaluate independent trajectories and dense boundary reliability; select a bounded matched-control training intervention.'))
    print(json.dumps({k:v['aggregate'] for k,v in results.items()},indent=2))


if __name__=='__main__':main()
