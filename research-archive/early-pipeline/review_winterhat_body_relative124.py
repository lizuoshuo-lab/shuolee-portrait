"""Separate fitted root orientation from local body posture, read-only CPU."""
from pathlib import Path
import os,time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import mesh_renderer

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT=ROOT/'winterhat-selected-view-pose124/body-relative-review'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    oldfile=ROOT/'winterhat-temporal/official-fit/frames/frame_000056/winner-camera-space.npz'
    newfile=ROOT/'winterhat-selected-view-pose124/frames/frame_000056/winner-camera-space.npz'
    with np.load(newfile.parent.parent.parent/'selected-fit.npz') as z:
        assert z['body_pose_axis_angle'].shape==(1,63)
    paths=[Path(__file__),oldfile,newfile,newfile.parent.parent.parent/'selected-fit.npz']
    frozen={str(p):sha(p) for p in paths};OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),scope='Read-only common local-body orientation for already reviewed winterhat56; shared body pose means this is the same local gesture at all8 fitted times, but face/fingers may differ.',
        transform='v_body=(v_camera-j_camera[8])*R_global. OpenPose25index8 maps to original SMPLXpelvis0. This removes fitted root orientation and translation, not joint pose or shape.',
        caveat='Body-root axes are a model gauge, not a measured gravity frame. Synthetic views are not3D truth.',
        budget=dict(CPU_seconds=120,new_fit_calls=0,GPU_calls=0),source_hashes=frozen))
    start=time.monotonic();arrays=[];vertices=[];checks=[]
    for file in (oldfile,newfile):
        with np.load(file) as z:d=dict(z)
        R=Rotation.from_rotvec(d['global_orient'][0]).as_matrix();pivot=d['joints_camera'][0,8]
        body=(d['vertices_camera'][0]-pivot)@R
        restored=body@R.T+pivot
        error=float(np.abs(restored-d['vertices_camera'][0]).max());assert error<2e-6
        vertices.append(body);arrays.append(d);checks.append(dict(file=str(file),camera_roundtrip_max_abs_m=error))
    union=np.concatenate(vertices);center=(union.max(0)+union.min(0))/2;extent=float(np.ptp(union,axis=0).max())
    renderer,renderer_source=mesh_renderer();K=np.array([[1150.,0,170],[0,1150.,240],[0,0,1.]])
    image=Image.new('RGB',(680,1920));front=np.diag([1.,-1.,-1.])
    for row,deg in enumerate((0,90,180,-90)):
        R=front@Rotation.from_rotvec([0,np.deg2rad(deg),0]).as_matrix()
        for col,(v,d) in enumerate(zip(vertices,arrays)):
            camera=(v-center)@R.T;camera[:,2]+=3*extent
            rendered=renderer(Image.new('RGB',(340,480)),np.array([0,0,340,480]),
                dict(vertices_camera=camera[None],faces=d['faces'],K_integer_centers=K))
            image.paste(rendered,(col*340,row*480))
    output=Path('/mnt/d/00/记录/001/分析_冬帽身体相对坐标124_frame_000056_去除整体朝向_旧逐帧_八视角共享姿态.png')
    assert not output.exists();image.save(output)
    assert all(sha(p)==h for p,h in frozen.items()) and time.monotonic()-start<120
    save(OUT/'result.json',dict(updated=now(),state='body_relative_views_ready',seconds=time.monotonic()-start,
        file=str(output),image_sha256=sha(output),renderer_sha256=sha(renderer_source),roundtrip=checks,
        source_hashes_unchanged=True,actual_visual_review=False,quality_accepted=False,new_fit_calls=0))
    print('BODY_RELATIVE_VIEW_READY',output,flush=True)


if __name__=='__main__':main()
