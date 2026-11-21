"""Deliver the measured SfM point cloud as points, not as a fake Gaussian PLY."""
from pathlib import Path
import json,sys,hashlib,os,datetime
import numpy as np

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee';V=B.parent/'viewer';O=B/'current/full-scene-localization-074'
MODEL='full-scene-localization-074'
sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
import read_write_model as rw
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def save(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)

def main():
    assert (O/'evaluation.json').exists()
    dest=V/'models'/MODEL;dest.mkdir(exist_ok=False)
    raw=read(O/'cameras.json');gauge=read(O/'camera-gauge-comparison.json')
    # Old scene is used only to orient this display. Camera fitting did not use it.
    up=read(V/'models/whole-body-static-registered-camera/delivery.json')['world_to_viewer']
    A=np.array(up['rotation']);Q=np.array(gauge['rotation']);s=gauge['scale']
    rotation=A@Q;translation=A@(np.array(gauge['translation'])-np.array(up['origin_world']))
    cc,ims,points=rw.read_model(str(O/'refined_sparse'),ext='.bin')
    ids=sorted(points);xyz=np.array([points[i].xyz for i in ids]);rgb=np.array([points[i].rgb for i in ids],dtype=np.uint8)
    transformed=s*xyz@rotation.T+translation
    assert np.isfinite(transformed).all() and s>0 and abs(np.linalg.det(rotation)-1)<1e-8
    packed=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,k in enumerate(['x','y','z']):packed[k]=transformed[:,j]
    for j,k in enumerate(['red','green','blue']):packed[k]=rgb[:,j]
    p=dest/'shuolee_074_scene_sparse.ply'
    header='ply\nformat binary_little_endian 1.0\ncomment Colored SfM sparse scene; not Gaussian data\n'+f'element vertex {len(xyz)}\n'+'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with p.open('wb') as f:f.write(header.encode('ascii'));f.write(packed.tobytes())
    cameras=[]
    intr=raw['intrinsics'];rows_by_id={r['image_id']:r for r in raw['records']}
    errors=[]
    for j in range(0,len(ids),max(1,len(ids)//1000)):
        point=points[ids[j]];im=ims[int(point.image_ids[0])];E=np.array(rows_by_id[im.id]['w2c'])
        original=E[:3,:3]@point.xyz+E[:3,3]
        cam_center=-E[:3,:3].T@E[:3,3];new_center=s*rotation@cam_center+translation
        rounded=np.array([packed[k][j] for k in ['x','y','z']])
        reloaded=(E[:3,:3]@rotation.T)@(rounded-new_center)
        a=original[:2]/original[2];b=reloaded[:2]/reloaded[2]
        errors.append(float(np.linalg.norm((a-b)*intr[0])))
    assert max(errors)<.01,f'Point export projection error{max(errors)}'
    for i,r in enumerate(raw['records']):
        E=np.array(r['w2c']);pos=s*rotation@(-E[:3,:3].T@E[:3,3])+translation
        rot=rotation@E[:3,:3].T
        cameras.append(dict(id=i,img_name=f'raw_{r["source_frame_index"]:06d}',position=pos.tolist(),rotation=rot.tolist(),
            width=raw['width'],height=raw['height'],fx=intr[0],fy=intr[0],cx=intr[1],cy=intr[2],
            raw_source_frame=r['source_frame_index'],camera_kind='fresh_full_scene_SfM',
            label=f'原视频{r["source_frame_index"]} · 074整图SfM',projectionModel='rectified_pinhole',
            split='SfM_registration_view_not_independent_test'))
    centers=np.array([r['position'] for r in cameras]);directions=np.array([r['rotation'] for r in cameras])[:,:,2]
    P=np.eye(3)[None]-directions[:,:,None]*directions[:,None,:]
    focus=np.linalg.lstsq(P.sum(0),np.einsum('nij,nj->i',P,centers),rcond=None)[0]
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    meta=dict(label='074 · 整场景定位 · 稀疏点云（非高斯）',assetType='sfm',model=f'models/{MODEL}/{p.name}',sha256=digest,
        pointCount=len(xyz),gaussianCount=0,trained=False,steps=0,qualityAccepted=False,deliveryAccepted=True,
        status='175帧整图重新定位完成；这是SfM稀疏点及相机，未生成新的高斯，质量结论见核验页。',
        cameraSource='sfm',cameras=cameras,realFramesUsed=175,registeredCameras=175,interpolatedCameras=0,
        initialView=99,focus=focus.tolist(),report='body-smplx.html#full-scene-localization',
        coordinateSystem='Display+Yup; global similarity only, no point deletion/warping. Original SfM frame preserved separately.',
        projection='Ideal pinhole view obtained by removing estimated radial distortion, with fitted principal point. Raw photos are not overlaid in this viewer.',
        fromScratch=True,oldMapUsedForFitting=False,oldMapUsedOnlyForDisplayGauge=True,
        rawIntrinsics=intr,rawModel='SIMPLE_RADIAL',rawCameraFile=str(O/'cameras.json'))
    save(V/f'model-{MODEL}.json',meta)
    receipt=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),type='SfM_points_not_Gaussian',
        vertices=len(xyz),source_model=str(O/'refined_sparse'),source_cloud=str(O/'whole_scene_sparse.ply'),
        delivered_path=str(p),sha256=digest,rotation=rotation.tolist(),scale=s,translation=translation.tolist(),
        points_dropped=0,projection_check_points=len(errors),projection_max_difference_px=max(errors),
        viewer_open_test_pending=True,gaussian_training_steps=0,quality_accepted=False,
        camera_projection='Fitted pinhole intrinsics with principal point, ideal undistorted coordinates')
    save(dest/'delivery.json',receipt);save(O/'viewer-delivery.json',receipt)
    js=V/'viewer.js';src=js.read_text(encoding='utf8')
    if '"'+MODEL+'"' not in src:src=src.replace('const BUILTINS = [','const BUILTINS = ["'+MODEL+'", ',1)
    js.write_text(src,encoding='utf8')
    html=V/'index.html';shtml=html.read_text(encoding='utf8').replace('已训练结果与导入文件','迭代结果与导入文件');html.write_text(shtml,encoding='utf8')
    print(json.dumps(dict(path=str(p),vertices=len(xyz),projection_error=max(errors),sha256=digest),ensure_ascii=False))


if __name__=='__main__':main()
