"""CPU-only native and novel-view review of the saved eight-view candidate."""
from pathlib import Path
import os,json,hashlib
from datetime import datetime,timezone
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from review_full_sequence123 import mesh_renderer,crop_box,joined

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT=ROOT/'grass-selected-view-pose124'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return dict(z)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cv2.setNumThreads(1)
    plan=read(OUT/'plan.json');result=read(OUT/'result.json')
    assert read(OUT/'progress.json')['state']=='completed_review_pending'
    assert sha(OUT/'selected-fit.npz')==result['fit_sha256']
    review=OUT/'review';review.mkdir(exist_ok=True)
    assert not (review/'result.json').exists()
    record=Path('/mnt/d/00/记录/005');renderer,renderer_source=mesh_renderer()
    records=[];tiles=[]
    for row in plan['rows']:
        name=row['frame'];newfile=OUT/'frames'/name/'winner-camera-space.npz'
        oldfile=ROOT/'grass-temporal-refinement/official-fit/frames'/name/'winner-camera-space.npz'
        old,new=load(oldfile),load(newfile)
        assert np.array_equal(old['betas'],new['betas']) and np.array_equal(old['faces'],new['faces'])
        assert np.array_equal(old['K_integer_centers'],new['K_integer_centers'])
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        box=crop_box(mask.getbbox(),raw.size,30)
        boxes=dict(body=box,face=crop_box(row['face_box_xyxy'],raw.size,35),
            feet=crop_box([box[0],box[1]+.72*(box[3]-box[1]),box[2],box[3]],raw.size))
        images={}
        for region,bb in boxes.items():
            crop=raw.crop(bb);panel=joined([crop]+[renderer(crop,np.array(bb),arr) for arr in (old,new)])
            path=record/f'smpl-x_草地八视角联合124_{name}_{region}_原图_旧连续约束_共享局部姿态.png'
            assert not path.exists();panel.save(path)
            images[region]=dict(path=str(path),sha256=sha(path),native_crop=bb,resized=False)
        vertices=np.concatenate([old['vertices_camera'][0],new['vertices_camera'][0]])
        center=(vertices.min(0)+vertices.max(0))/2;extent=float(np.ptp(vertices,axis=0).max())
        K=np.array([[1150.,0,170],[0,1150.,240],[0,0,1.]])
        views=Image.new('RGB',(680,1920))
        for j,deg in enumerate([0,90,180,-90]):
            R=Rotation.from_rotvec([0,np.deg2rad(deg),0]).as_matrix()
            for k,arr in enumerate((old,new)):
                v=(arr['vertices_camera'][0]-center)@R.T;v[:,2]+=3*extent
                img=renderer(Image.new('RGB',(340,480)),np.array([0,0,340,480]),
                    dict(vertices_camera=v[None],faces=arr['faces'],K_integer_centers=K))
                views.paste(img,(340*k,480*j))
        path=record/f'分析_草地八视角联合124_{name}_相同四方向_旧连续约束_共享局部姿态.png'
        assert not path.exists();views.save(path)
        images['sides']=dict(path=str(path),sha256=sha(path),synthetic_not_truth=True)
        tile=Image.new('RGB',(1520,480));nat=Image.open(images['body']['path']);nat.thumbnail((840,480))
        tile.paste(nat,(0,0));tile.paste(views.crop((0,480,680,960)),(840,0));tiles.append(tile)
        metrics={}
        target=new['observed_native_keypoints'][0]
        for label,arr in [('old_temporal',old),('shared_eight',new)]:
            err=np.linalg.norm(arr['joints_projected_native'][0]-target[:,:2],axis=1)
            d={}
            for group,idx in [('body',[0,2,3,4,5,6,7,8,10,11,13,14,15,16,17,18]),
                ('face',list(range(67,118))),('hands',list(range(25,67))),('feet',list(range(19,25)))]:
                w=target[idx,2]**2
                d[group+'_weighted_rms_px']=float(np.sqrt((w*err[idx]**2).sum()/w.sum()))
                d[group+'_supported_max_px']=float(err[idx][w>0].max())
            local=(arr['joints_camera'][0]-arr['joints_camera'][0,2:3])@Rotation.from_rotvec(arr['global_orient'][0]).as_matrix()
            d.update(right_wrist_relative_right_shoulder_body_local_m=local[4].tolist(),
                arm_error_px=err[[2,3,4,5,6,7]].tolist())
            metrics[label]=d
        records.append(dict(frame=name,images=images,metrics=metrics,old_sha256=sha(oldfile),new_sha256=sha(newfile),
            same_K_shape_pixels=True,pose_and_body_placement_changed=True))
    pages=[]
    for start in range(0,len(tiles),4):
        panel=Image.new('RGB',(1520,480*min(4,len(tiles)-start)))
        for i,tile in enumerate(tiles[start:start+4]):panel.paste(tile,(0,i*480))
        path=review/f'page-{start//4+1:02d}.png';panel.save(path);pages.append(str(path))
    report=dict(updated=datetime.now(timezone.utc).isoformat(),records=records,pages=pages,
        columns_native=['raw','old_temporal','shared_eight'],columns_sides=['old_temporal','shared_eight'],
        actual_visual_review=False,quality_accepted=False,gaussian_admitted=False,detector_error_not_truth=True,
        original_inputs_unchanged=True,text_on_images=False,renderer_sha256=sha(renderer_source),source_sha256=sha(__file__))
    (review/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('EIGHT_VIEW_NATIVE_REVIEW_READY',flush=True)

if __name__=='__main__':main()
