"""Measure native shoe corners where the seated seed actually has a visible foot."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

import fit_video21_seated149 as seated
from extract_video21_shoe_depth162 import FROZEN, replace
from fit_video21_native148 import NATIVE
from prepare_video21_shoe_binding163 import FOOT_BONES, bind_one
from review_full_sequence123 import relocate
from track_video21_background154 import load_gray
from update_leafwoman129_state import atomic, read, sha


BASE=seated.base.OUT
OUT=BASE/'shoe-overlap-material164'
POLYGONS={
    'frame_000029':dict(origin=[1106,1112],right=[(118,100),(151,89),(188,103),(192,133),(165,150),(105,154),(105,138)],
                        left=[(238,106),(266,103),(276,132),(271,177),(225,187),(223,151)]),
    'frame_000053':dict(origin=[1285,1105],right=[(111,138),(136,127),(161,134),(177,158),(193,187),(188,211),(141,202),(112,173)],
                        left=[(156,92),(181,87),(210,98),(235,127),(241,149),(198,144),(167,122)]),
}


def candidates():
    intake=relocate(read(NATIVE/'result.json')['data_root'])/'data/inputs.json'
    assert sha(intake)==read(BASE/'dense-camera-tracks158/plan.json')['input_sha256']
    source={r['frame']:r for r in read(intake)['records']}
    seed={r['frame']:r for r in read(seated.OUT/'result.json')['frames']}
    asset=seated.base.original.ASSET
    with np.load(asset,allow_pickle=False) as z:
        weights=z['weights']
    output={}
    for f,regions in POLYGONS.items():
        image=load_gray(source[f],source[f]['wh'])
        assert sha(seed[f]['output'])==seed[f]['sha256']
        with np.load(seed[f]['output'],allow_pickle=False) as z:
            v=z['vertices_camera'][0].astype(float);faces=z['faces'];K=z['K_integer_centers']
        tags=np.full(len(faces),'other',dtype='<U5')
        for side,indices in FOOT_BONES.items():
            foot=weights[:,indices].sum(1)>.5
            tags[foot[faces].all(1)]=side
        candidates=[]
        for side in ('left','right'):
            xy=np.asarray(regions[side],np.int32)+np.asarray(regions['origin'],np.int32)
            x0,y0=xy.min(0);x1,y1=xy.max(0)+1
            mask=np.zeros((y1-y0,x1-x0),np.uint8)
            cv2.fillPoly(mask,[xy-[x0,y0]],255)
            p=cv2.goodFeaturesToTrack(image[y0:y1,x0:x1],maxCorners=20,qualityLevel=.01,minDistance=7,mask=mask)
            if p is None:
                continue
            count=0
            for pixel in p[:,0]+[x0,y0]:
                hit=bind_one(v,faces,tags,K,pixel,side)
                if not hit['bound'] or hit['pixel_distance']!=0:
                    continue
                label=f'{side}_{count:02d}'
                candidates.append(dict(label=label,side=side,source_uv=pixel.tolist(),binding=hit))
                count+=1
                if count==3:
                    break
            print('SHOE_OVERLAP_CORNERS',f,side,count,flush=True)
        output[f]=candidates
    return output


def main():
    assert not OUT.exists()
    frozen=FROZEN/'source.py'
    before=read(BASE/'fixedk-dense159-reference/review/visual-decision.json')
    assert before['local_shoe_source_sha256']==sha(frozen)
    found=candidates()
    receipt=OUT.with_name('shoe-overlap-seeds164.json')
    assert not receipt.exists()
    atomic(receipt,dict(updated=datetime.now(timezone.utc).isoformat(),worker_sha256=sha(__file__),
        original_tracker_sha256=sha(frozen),polygon_evidence=POLYGONS,candidates=found,
        side_rule='Original camera pixel lies inside reviewed shoe polygon and nearest visible SMPL-X triangle belongs to the matching foot.',
        selection='Up to three corners per shoe in cv2 corner score order, before using geometry or errors from other views.'))
    source=frozen.read_text(encoding='utf-8')
    start,end=source.index('spec=[\n'),source.index('\nrecords={}')
    spec=[(int(f[-3:]),[0,0],[(r['label'],r['source_uv']) for r in rows]) for f,rows in found.items()]
    assert all(rows for _,_,rows in spec)
    source=source[:start]+'spec='+repr(spec)+source[end:]
    start,end=source.index('        x0,y0,x1,y1=np.asarray(local)'),source.index('        obs={center:seed}')
    source=source[:start]+'        seed=np.asarray(local,dtype=np.float32)\n'+source[end:]
    source=replace(source,"destination=OUT/'shoe-material-local161'","destination=OUT/'shoe-overlap-material164'")
    source=replace(source,"triangulated_from=[a[0],b[0]])",
                   """triangulated_from=[a[0],b[0]],xyz_arbitrary_scale=h[:3].tolist(),
                                  observed_uv={str(i):xy.tolist() for i,xy,_,_ in supported})""")
    sys.argv[1:]=[str(Path(__file__).resolve().parent)]
    exec(compile(source,str(frozen),'exec'),{})
    measured=read(OUT/'result.json')
    assert len(measured['rows'])==sum(map(len,found.values()))
    atomic(OUT/'source-receipt.json',dict(seed_manifest=str(receipt),seed_manifest_sha256=sha(receipt),
        source_result_sha256=sha(OUT/'result.json'),method_source_sha256=sha(frozen),
        source_material_review_required=True,gaussian_started=False))


if __name__=='__main__':
    main()
