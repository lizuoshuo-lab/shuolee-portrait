"""Read-only comparison of every grass temporal checkpoint pose."""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import run_grass_temporal124 as c
from review_full_sequence123 import mesh_renderer, crop_box, joined
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster


def main(checkpoint):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import cv2
    cv2.setNumThreads(1)
    suffix = '' if checkpoint == 13 else '-027'
    request = c.read(c.OUT/('review-request'+suffix+'.json'))
    assert len(request['frames']) == checkpoint
    out = c.OUT/f'review-{checkpoint:03d}'; out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(), 'Review the saved images; never overwrite earlier evidence.'
    rows = c.read(c.SRC/'data/inputs.json')['records']; byid = {r['frame']:r for r in rows}
    with np.load(c.SRC/'data/observations.npz', allow_pickle=False) as z: obs = dict(z)
    confidence, _ = relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    targets = np.zeros((len(rows),118,3), np.float32)
    dst=np.r_[c.entry.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    targets[:,dst,:2]=obs['keypoints'][:,src]; targets[:,dst,2]=confidence[:,src]
    renderer,_ = mesh_renderer(); record=Path('/mnt/d/00/记录/005'); record.mkdir(exist_ok=True)
    reports, tiles, fits = [], [], {}
    for name in sorted(request['frames']):
        row=byid[name]; i=[r['frame'] for r in rows].index(name)
        target=mask_outside_raster(targets[i:i+1],row['wh'])[0][0]
        oldfile=c.BASE/'frames'/name/'winner-camera-space.npz'
        newfile=c.OUT/'official-fit/frames'/name/'winner-camera-space.npz'
        if not newfile.exists():
            assert name==request['last_frame']
            newfile=c.OUT/('review-pause-pose'+suffix+'.npz')
        with np.load(oldfile,allow_pickle=False) as z: old=dict(z)
        with np.load(newfile,allow_pickle=False) as z: new=dict(z)
        assert np.array_equal(old['betas'],new['betas']) and np.array_equal(old['faces'],new['faces'])
        assert np.array_equal(old['K_integer_centers'],new['K_integer_centers'])
        assert np.array_equal(old['camera_translation'],new['camera_translation'])
        assert c.sha(row['image'])==row['source_sha256'] and c.sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB'); mask=Image.open(row['mask']).convert('L')
        box=crop_box(mask.getbbox(),raw.size,30)
        boxes=dict(body=box,face=crop_box(row['face_box_xyxy'],raw.size,35),
            feet=crop_box([box[0],box[1]+.72*(box[3]-box[1]),box[2],box[3]],raw.size))
        images={}
        for region, bb in boxes.items():
            crop=raw.crop(bb); panel=joined([crop]+[renderer(crop,np.array(bb),a) for a in (old,new)])
            path=record/f'smpl-x_草地连续帧124_{name}_{region}_原图_独立拟合_连续约束.png'
            if path.exists(): assert np.array_equal(np.array(Image.open(path)),np.array(panel))
            else: panel.save(path)
            images[region]=dict(path=str(path),sha256=c.sha(path),native_crop=list(map(int,bb)),resized=False)
        v=np.concatenate([a['vertices_camera'][0] for a in (old,new)])
        center=(v.min(0)+v.max(0))/2; extent=float(np.max(np.ptp(v,axis=0)))
        K=np.array([[1150.,0,170],[0,1150.,240],[0,0,1.]])
        views=Image.new('RGB',(680,480*4))
        for j,deg in enumerate([0,90,180,-90]):
            R=Rotation.from_rotvec([0,np.deg2rad(deg),0]).as_matrix()
            for k,a in enumerate((old,new)):
                vv=(a['vertices_camera'][0]-center)@R.T; vv[:,2]+=3*extent
                im=renderer(Image.new('RGB',(340,480)),np.array([0,0,340,480]),
                    dict(vertices_camera=vv[None],faces=a['faces'],K_integer_centers=K))
                views.paste(im,(340*k,480*j))
        path=record/f'smpl-x_草地连续帧124_{name}_同相机四方向_独立拟合_连续约束.png'
        if path.exists(): assert np.array_equal(np.array(Image.open(path)),np.array(views))
        else: views.save(path)
        images['sides']=dict(path=str(path),sha256=c.sha(path),synthetic_not_truth=True)
        tile=Image.new('RGB',(1520,480)); nat=Image.open(images['body']['path']);nat.thumbnail((840,480))
        tile.paste(nat,(0,0));tile.paste(views.crop((0,480,680,960)),(840,0));tiles.append(tile)
        metrics={}
        for label,a in [('independent',old),('temporal',new)]:
            error=np.linalg.norm(a['joints_projected_native'][0]-target[:,:2],axis=-1)
            metric={}
            for group,slots in [('body',np.array([0,2,3,4,5,6,7,8,10,11,13,14,15,16,17,18])),
                ('face',np.arange(67,118)),('hands',np.arange(25,67)),('feet',np.arange(19,25))]:
                w=target[slots,2];metric[group+'_weighted_mean_px']=float(np.sum(error[slots]*w)/np.sum(w)) if np.sum(w)>0 else None
            metric['all118_error_px']=error.tolist(); metric['arm_error_px']=error[[2,3,4,5,6,7]].tolist()
            metric['relative_right_wrist_model_units']=(a['joints_camera'][0,4]-a['joints_camera'][0,2]).tolist()
            metrics[label]=metric
        fits[name]=(old,new)
        reports.append(dict(frame=name,images=images,metrics=metrics,confidence=target[:,2].tolist(),
            same_K_shape_camera_translation=True,old_sha256=c.sha(oldfile),new_sha256=c.sha(newfile)))
    transitions=[]; ids=sorted(fits)
    for a,b in zip(ids[:-1],ids[1:]):
        assert int(b[-6:])-int(a[-6:])==1
        arms={}
        for index,label in [(0,'independent'),(1,'temporal')]:
            fa,fb=fits[a][index],fits[b][index]
            ra=Rotation.from_rotvec(fa['body_pose_axis_angle'].reshape(-1,3)).as_matrix()
            rb=Rotation.from_rotvec(fb['body_pose_axis_angle'].reshape(-1,3)).as_matrix()
            angles=np.degrees(Rotation.from_matrix(ra.transpose(0,2,1)@rb).magnitude())
            va=fa['joints_camera'][0,4]-fa['joints_camera'][0,2]
            vb=fb['joints_camera'][0,4]-fb['joints_camera'][0,2]
            Ea,Eb=np.array(byid[a]['w2c']),np.array(byid[b]['w2c'])
            arms[label]=dict(max_local_joint_deg=float(angles.max()),max_joint=int(angles.argmax()),
                wrist_camera_delta_model_units=float(np.linalg.norm(vb-va)),
                wrist_world_delta_model_units=float(np.linalg.norm(Eb[:3,:3].T@vb-Ea[:3,:3].T@va)))
        transitions.append(dict(previous=a,current=b,dt=byid[b]['time_seconds']-byid[a]['time_seconds'],**arms))
    pages=[]
    for start in range(0,len(tiles),4):
        canvas=Image.new('RGB',(1520,480*min(4,len(tiles)-start)))
        for j,tile in enumerate(tiles[start:start+4]):canvas.paste(tile,(0,480*j))
        path=out/f'page-{start//4+1:02d}.png';canvas.save(path);pages.append(str(path))
    c.save(out/'result.json',dict(updated=c.now(),frames=checkpoint,records=reports,transitions=transitions,pages=pages,
        actual_visual_review=False,quality_accepted=False,gaussian_admitted=False,
        columns_native=['raw','independent','temporal'],columns_sides=['independent','temporal'],
        detector_residuals_not_ground_truth=True,original_images_and_parameters_unchanged=True,
        source_script_sha256=c.sha(__file__),text_on_images=False))
    print('GRASS_REVIEW_READY',checkpoint,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=int,choices=[13,27],required=True)
    main(ap.parse_args().checkpoint)
