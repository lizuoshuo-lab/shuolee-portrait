"""Reuse the existing observation contract on demo59; preserve all targets."""
from pathlib import Path
import sys
import shutil

import numpy as np
from PIL import Image, ImageDraw
import replay_demo_limb_losses123 as base


def main():
    sys.path.insert(0,str(base.PROJECT/'src'))
    from shuolee_portrait.observation_contract import inspect_coco133_targets
    output=base.ROOT.parent/'observation-force-audit'
    result_path=output/'shared-wrist-observations.json'
    assert output.is_dir() and not result_path.exists()
    data=Path('/opt/gs-slam-lab/shuolee-local-surface-123/demo-vhap-fps3/data')
    with np.load(data/'observations.npz',allow_pickle=False) as z:
        obs=dict(z)
    with np.load(base.ROOT/'fit/fitted.npz',allow_pickle=False) as z:
        fit={k:z[k].copy() for k in ('frame_ids','observed_native_keypoints','joints_projected_native')}
    ids=fit['frame_ids'].astype(str).tolist()
    order=obs['frame_ids'].astype(str).tolist()
    slots=[order.index(frame) for frame in ids]
    xy=obs['keypoints'][slots]
    report=inspect_coco133_targets(xy,obs['joint_names'].astype(str).tolist(),ids,
        detector_scores=obs['scores'][slots],required_error_px=1.0)
    alias_pairs=[('left',7,25,9,91),('right',4,46,10,112)]
    actual=[]
    for name,body,hand,source_body,source_hand in alias_pairs:
        # The unchanged original input adapter explicitly stores float32 targets.
        target_dtype=fit['observed_native_keypoints'].dtype
        assert np.array_equal(xy[:,source_body].astype(target_dtype),fit['observed_native_keypoints'][:,body,:2])
        assert np.array_equal(xy[:,source_hand].astype(target_dtype),fit['observed_native_keypoints'][:,hand,:2])
        delta=float(np.abs(fit['joints_projected_native'][:,body]-fit['joints_projected_native'][:,hand]).max())
        assert delta==0
        actual.append(dict(side=name,body_destination=body,hand_destination=hand,
            body_source=source_body,hand_source=source_hand,saved_projection_alias_max_px=delta))
    report.update(actual_original118_consumer_alias_check=actual,
        source_observations_sha256=base.sha(data/'observations.npz'),baseline_fit_sha256=base.sha(base.ROOT/'fit/fitted.npz'),
        contract_source_sha256=base.sha(base.PROJECT/'src/shuolee_portrait/observation_contract.py'),
        independent_truth=False,visibility_not_inferred=True,weights_changed=False,fit_changed=False,
        diagnostic_repair='First CPU audit asserted raw float64 equals stored float32 without applying the existing input cast. It stopped before writing outputs. Comparison now reproduces the original cast; no fitting or target changes.')
    images=[]
    records={r['frame']:r for r in base.read(base.ROOT/'data/training-inputs.json')['records']}
    delivery=Path('/mnt/d/00/记录/demo')
    delivery.mkdir(parents=True,exist_ok=True)
    for frame in ('frame_000012','frame_000017','frame_000018','frame_000033'):
        slot=ids.index(frame);row=records[frame]
        assert base.sha(row['image'])==row['source_sha256']
        image=Image.open(row['image']).convert('RGB')
        x1,y1,x2,y2=row['initial_box_xyxy']
        crop=(max(0,int(x1)-24),max(0,int(y1)-24),min(image.width,int(x2)+25),min(image.height,int(y2)+25))
        raw=image.crop(crop);observed=raw.copy();predicted=raw.copy()
        d2=ImageDraw.Draw(observed);d3=ImageDraw.Draw(predicted)
        point=lambda v:(float(v[0]-crop[0]),float(v[1]-crop[1]))
        target=fit['observed_native_keypoints'][slot,:,:2]; projection=fit['joints_projected_native'][slot]
        for chain in ((2,3,4),(5,6,7)):
            d2.line([point(target[i]) for i in chain],fill=(0,190,255),width=2)
            d3.line([point(projection[i]) for i in chain],fill=(40,220,70),width=2)
        for draw,xy0,indices,color,radius in ((d2,target,(4,7),(0,190,255),4),
                    (d2,target,(25,46),(255,135,0),4),(d3,projection,(4,7),(40,220,70),4)):
            for i in indices:
                x,y=point(xy0[i]); draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
        canvas=Image.new('RGB',(raw.width*3,raw.height),(255,255,255))
        for i,part in enumerate((raw,observed,predicted)):
            canvas.paste(part,(i*raw.width,0))
        name=f'特征点_手腕重复观测_{frame}_原图_2D观测_3D投影.png'
        local=output/name;canvas.save(local)
        dest=delivery/name
        assert not dest.exists(), 'Keep existing record images.'
        shutil.copyfile(local,dest)
        images.append(dict(frame=frame,path=str(local),delivery=str(dest),sha256=base.sha(local),crop=list(crop),
            panels=['raw','native2D body wrists cyan and hand roots orange','saved3D arm and wrist projections green'],
            resized=False,text_drawn=False,source_image_sha256=row['source_sha256']))
    report['images']=images
    base.save(result_path,report)
    print('SHARED_WRIST_CHECK',len(ids),'frames',len(images),'native triptychs')


if __name__=='__main__':main()
