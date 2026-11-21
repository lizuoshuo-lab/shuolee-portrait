"""Compare two saved official meshes in native source pixels without fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from review_full_sequence123 import read, sha, mesh_renderer, joined, relocate


def main(baseline_root, candidate_dir, frame, output):
    base, candidate, out=map(relocate,(baseline_root,candidate_dir,output))
    assert not out.exists();out.mkdir(parents=True)
    directories=[base/'frames'/frame,candidate]
    data=[]
    for directory in directories:
        assert read(directory/'result.json')['state']=='complete'
        with np.load(directory/'winner-camera-space.npz',allow_pickle=False) as z:data.append(dict(z))
    assert all(str(d['frame_id'].item())==frame for d in data)
    assert np.array_equal(data[0]['K_integer_centers'],data[1]['K_integer_centers'])
    rows=read(base/'data/training-inputs.json')['records'];slot=[r['frame'] for r in rows].index(frame);row=rows[slot]
    raw=Image.open(relocate(row['image'])).convert('RGB')
    assert sha(relocate(row['image']))==row['source_sha256']
    with np.load(base/'fit/fitted.npz',allow_pickle=False) as z:
        assert z['frame_ids'][slot]==frame
        target=z['observed_native_keypoints'][slot]
    mesh,_=mesh_renderer()
    overlays=[mesh(raw,np.array([0,0,*raw.size]),d) for d in data]
    images=[]
    for item in read(directories[0]/'saved-mesh-review.json')['files']:
        if item['panels']!='原图_官方SMPLX透明覆盖':continue
        label=next((x for x in ['全身','面部','双脚','左手','右手'] if '_'+x+'_' in Path(item['path']).name),None)
        if label is None:continue
        box=item['crop_integer_xyxy']
        result=joined([raw.crop(box),*[im.crop(box) for im in overlays]])
        result.save(out/f'{label}.png')
        records=relocate('D:/00/记录/005')/('smpl-x' if label=='全身' else '分析')
        target_image=records/f'20261030_共享体型正面保护_{frame}_{label}_左原图_中原体型_右侧面更新体型.png'
        assert not target_image.exists();result.save(target_image)
        images.append(dict(label=label,path=str(out/f'{label}.png'),record=str(target_image),crop_integer_xyxy=box))
    groups=dict(body17=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]),feet6=np.arange(19,25),
                left_hand21=np.arange(25,46),right_hand21=np.arange(46,67),face51=np.arange(67,118))
    metrics=[]
    valid=target[:,2]>0;valid[[1,9,12]]=False
    for d in data:
        errors=np.linalg.norm(d['joints_projected_native'][0]-target[:,:2],axis=1)
        row_stats={}
        for key,ids in groups.items():
            e=errors[ids[valid[ids]]]
            row_stats[key]=dict(count=len(e),mean=float(e.mean()),median=float(np.median(e)),maximum=float(e.max()))
        metrics.append(row_stats)
    result=dict(frame_id=frame,groups_before=metrics[0],groups_after=metrics[1],images=images,
        source_hashes={str(p/'winner-camera-space.npz'):sha(p/'winner-camera-space.npz') for p in directories},
        actual_visual_review=False,same_native_K=True,source_image_sha256=row['source_sha256'],
        no_resize=True,no_image_text=True,GPU_calls=0,fit_updates=0,
        metric_scope='Same fitted detector observations; not independent physical ground truth.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out),before=metrics[0],after=metrics[1]),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',required=True);p.add_argument('--candidate-dir',required=True)
    p.add_argument('--frame',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.baseline_root,a.candidate_dir,a.frame,a.output)
