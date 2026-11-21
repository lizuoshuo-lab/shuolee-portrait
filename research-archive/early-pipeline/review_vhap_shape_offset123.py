"""Native, unwarped all-frame component overlays; not a new fit."""
import os
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image
from review_full_sequence123 import mesh_renderer, joined, read, sha
from run_official_gaussians123 import save, now

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
RUN=ROOT/'smplify-vhap-shape-offset-audit'
SOURCE=ROOT/'fullbody-smplifyx/full-sequence'
DATA=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3')
RECORD=Path('/mnt/d/00/记录/demo')
LABELS=('initial_shape_no_offset','current_shape_no_offset','current_shape_current_offset')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    cv2.setNumThreads(1)
    report=read(RUN/'result.json')
    assert report['frames']==59 and report['optimizer_updates']==0
    out=RUN/'comparison';out.mkdir(exist_ok=False)
    with np.load(RUN/'diagnostic-components.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    rows=read(DATA/'data/inputs.json')['records']
    assert list(arrays['frame_ids'])==[r['frame'] for r in rows]
    render,renderer=mesh_renderer();started=time.monotonic();results=[]
    for i,row in enumerate(rows):
        assert sha(row['image'])==row['source_sha256']
        prior=read(SOURCE/'review'/(row['frame']+'_saved-review.json'))
        raw=Image.open(row['image']).convert('RGB')
        overlays=[render(raw,np.array([0,0,*raw.size]),dict(vertices_camera=arrays[label][i:i+1],
                  faces=arrays['faces'],K_integer_centers=arrays['K_integer_centers'][i])) for label in LABELS]
        images=[]
        for region in ('body','face','left_hand','right_hand','feet'):
            box=next(x['crop_integer_xyxy'] for x in prior['files'] if x['region']==region)
            panels=[im.crop(box) for im in [raw,*overlays]]
            path=RECORD/f'SMPL-X_外扩分解_{row["frame"]}_{region}_原图_同姿态初始体型_仅体型_体型加偏移.png'
            assert not path.exists()
            joined(panels).save(path)
            with Image.open(path) as check:
                assert np.array_equal(np.array(check)[:,:panels[0].width],np.array(panels[0]))
            images.append(dict(region=region,record_path=str(path),sha256=sha(path),crop_integer_xyxy=box,
                               panel_wh=list(panels[0].size),panels=['raw',*LABELS],resampling=False,text_added=False))
        result=dict(frame_id=row['frame'],images=images,metrics=report['records'][i]['variants'],
                    renderer_sha256=sha(renderer),quality_accepted=False)
        save(out/(row['frame']+'.json'),result);results.append(result)
        print(json.dumps(dict(frame=row['frame'],completed=len(results))),flush=True)
    save(out/'result.json',dict(updated=now(),state='complete_pending_visual_review',frames=59,images=295,
          per_frame=results,actual_visual_review=False,quality_accepted=False,review_seconds=time.monotonic()-started,
          scope='Same saved poses and camera across every diagnostic column. Component removal is not new training.',
          component_archive_sha256=sha(RUN/'diagnostic-components.npz')))


if __name__=='__main__':main()
