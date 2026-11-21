"""Label observed portions of the native person/background edge for a head model."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from fit_side_frames import alpha_maps
from prepare_native_skin_profile import _native_cameras, decode, load_parser, PARSER_WEIGHTS, profile_pose_crop
from prepare_vhap_outline_targets import consistent_probabilities
from prepare_video_scene import sha256


def edge_labels(alpha, prob, lower_neck_y):
    binary = (alpha >= 128).astype(np.uint8)
    contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    labels = np.zeros_like(binary)
    if not contours:
        return labels
    points = max(contours,key=cv2.contourArea)[:,0]
    h,w = alpha.shape
    reach=max(3,int(round(min(h,w)*.018)))
    tangent=np.roll(points,reach,axis=0)-np.roll(points,-reach,axis=0)
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))/np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1)
    a,b=(np.rint(points+sign*reach*normal).astype(int) for sign in (-1,1))
    valid=((a>=0)&(a<[w,h])&(b>=0)&(b<[w,h])).all(1)
    a,b,points=a[valid],b[valid],points[valid]
    flip=alpha[a[:,1],a[:,0]]<alpha[b[:,1],b[:,0]]
    inside,outside=np.where(flip[:,None],b,a),np.where(flip[:,None],a,b)
    ix,iy=inside.T
    ox,oy=outside.T
    kind=prob[:,iy,ix].argmax(0)
    # A parser can call dark hair a hat, or classify a turned neck as background.
    # The RVM crossing locates the observed outer contour. Semantics assign regions;
    # a confident clothing interior excludes shoulders. The projected open neck rim
    # supplies only a conservative lower crop and never generates target pixels.
    reliable=(alpha[iy,ix]>=195)&(alpha[oy,ox]<=60)&(points[:,1]<lower_neck_y)
    reliable&=~np.isin(kind,[15,16])
    labels[points[reliable,1],points[reliable,0]]=4
    for code,selector in ((1,np.isin(kind,[17,18])),(2,(kind>=1)&(kind<=13)),(3,kind==14)):
        x,y=points[reliable&selector].T
        labels[y,x]=code
    return labels


def run(source):
    project=Path(__file__).resolve().parent
    weights=project/'assets/face-parsing/resnet18.pt'
    if sha256(weights)!=PARSER_WEIGHTS:
        raise ValueError('Unexpected face parser weights')
    scene=source/'scene'
    rows=json.loads((source/'head-neck/initialization.json').read_bytes())['frames']
    K,size,_=_native_cameras(scene)
    _,_,parity,maps,_=alpha_maps(scene,size,K,[rows[0]['name']])
    model=load_parser(project/'assets/face-parsing/code',weights,'cpu')
    import pickle
    from fit_native_skin_profile import camera
    with (project/'upstream/VHAP-6e1e770/asset/flame/FLAME_masks.pkl').open('rb') as fp:
        regions=pickle.load(fp,encoding='latin1')
        rim=np.intersect1d(regions['neck'],regions['boundary'])
        if rim.size<3:
            raise ValueError('Missing anatomical lower-neck boundary')
    with np.load(source/'geometry-binding/registered-head.npz',allow_pickle=False) as reg:
        if reg['filenames'].tolist()!=[r['name'] for r in rows]:
            raise ValueError('Neck ROI and camera order differ')
        world=(reg['vertices_tracker'][:,rim] @ reg['tracker_to_world_rotation'].T)*float(reg['scene_scale'])+reg['scene_origin']
    dest=source/'native-outer-contours'
    dest.mkdir(exist_ok=False)
    counts={}
    for i,row in enumerate(rows):
        name=row['name']
        rgb,_=decode(scene/'orig_undist/images'/name,cv2.IMREAD_COLOR)
        coarse,_=decode(scene/'headmask_undist/images'/Path(name).with_suffix('.png'),0)
        raw,_=decode(scene/'alpha_maps'/name,0)
        _,(x,y,x1,y1)=profile_pose_crop(rgb,coarse>=128)
        alpha=cv2.remap(raw,maps[0][y:y1,x:x1],maps[1][y:y1,x:x1],cv2.INTER_LINEAR)
        prob=consistent_probabilities(model,rgb[y:y1,x:x1],'cpu')
        intr,rot,center=camera(row)
        screen=(world[i]-center) @ rot @ intr.T
        lower=float(np.median(screen[:,1]/screen[:,2]))-y
        edge=edge_labels(alpha,prob,lower)
        usable=(alpha<128)|(prob[[15,16]].max(0)<.35)
        np.savez_compressed(dest/Path(name).with_suffix('.npz'),origin=[x,y],alpha=alpha,edge=edge,usable=usable)
        counts[name]=np.bincount(edge.ravel(),minlength=5)[1:].tolist()
    if not np.all(np.sum(list(counts.values()),axis=0)[:3]>0):
        raise ValueError('Missing anatomical outline support')
    (dest/'manifest.json').write_text(json.dumps(dict(size=size,matte_parity=parity,
        labels=['hair or headwear','profile skin','parsed exposed neck','other observed head/neck edge'],frames=counts),indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    run(p.parse_args().source.resolve())
