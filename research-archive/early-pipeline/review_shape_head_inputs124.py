"""Native diagnostic images of frozen shape/head inputs, no optimized mesh."""
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import audit_shape_head_inputs124 as a
from review_full_sequence123 import mesh_renderer,crop_box

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (a.OUT/'visual-review.json').exists()
result=a.read(a.OUT/'result.json');assert all(x['passed'] for x in result['replay_checks'].values())
with np.load(a.OUT/'diagnostic-geometries.npz',allow_pickle=False) as z:geometry=dict(z)
with np.load(a.ROOT/'face-absence-input124/indoor-diagnostic-input.npz',allow_pickle=False) as z:observations=dict(z)
rows=a.read(a.ROOT/'indoor-input/data/inputs.json')['records'];roster={r['frame']:r for r in rows}
record=Path('/mnt/d/00/记录/008');render,_=mesh_renderer()


def join(images,columns=None):
    w,h=images[0].size;cols=columns or len(images);out=Image.new('RGB',(w*cols,h*((len(images)+cols-1)//cols)))
    for i,im in enumerate(images):assert im.size==(w,h);out.paste(im,((i%cols)*w,(i//cols)*h))
    return out


def save_image(name,im,**info):
    path=record/(name+'.png');assert not path.exists();im.save(path)
    return dict(path=str(path),sha256=a.sha(path),**info)


native={}
for label in ['indoor_reference','original_frame_000040','face_absence_frame_000040']:
    name='frame_000105' if label=='indoor_reference' else 'frame_000040';row=roster[name]
    raw=Image.open(row['image']).convert('RGB');K=np.asarray(row['K_integer_centers']);vv={k:geometry[label+'__'+k] for k in ['saved','zero_expression','zero_beta']}
    points=[]
    for v in vv.values():
        q=v[0,geometry['head_vertex_mask']]@K.T;points.append(q[:,:2]/q[:,2:])
    points=np.concatenate(points)
    person=np.asarray(row['initial_box_xyxy']);top=person.copy();top[3]=person[1]+.3*(person[3]-person[1])
    union=np.r_[np.minimum(points.min(0),top[:2]),np.maximum(points.max(0),top[2:])]
    box=crop_box(union,raw.size,25);crop=raw.crop(box)
    panels=[crop]+[render(crop,np.asarray(box),dict(vertices_camera=v,faces=geometry['faces'],K_integer_centers=K)) for v in vv.values()]
    native[label]=save_image('分析_124_体型分解_'+label+'_头部_原图_保存结果_仅表情置零_仅体型置零_未重新拟合',join(panels),
        crop=list(map(int,box)),columns=['raw','saved','zero_expression','zero_beta'],same_pose_camera=True,refitted=False)
labels=['neutral','winterhat_reference','sofa_reference','indoor_reference']
allv=np.concatenate([geometry['rest_'+k] for k in labels]);center=(allv.min(0)+allv.max(0))/2;extent=np.max(np.ptp(allv,axis=0))
panels=[];K=np.array([[1150,0,170],[0,1150,240],[0,0,1.]])
for degree in [0,90]:
    co,si=np.cos(np.deg2rad(degree)),np.sin(np.deg2rad(degree));R=np.array([[co,0,si],[0,1,0],[-si,0,co]])@np.diag([1,-1,-1])
    for label in labels:
        v=(geometry['rest_'+label]-center)@R.T;v[:,2]+=3*extent
        panels.append(render(Image.new('RGB',(340,480)),np.array([0,0,340,480]),dict(vertices_camera=v[None],faces=geometry['faces'],K_integer_centers=K)))
rest=save_image('分析_124_共享体型静态分解_中性_冬帽_沙发_室内_上正面下侧面_同相机同尺度',join(panels,4),
    columns=labels,degrees=[0,90],same_camera_scale=True,no_physical_truth=True,center=center.tolist(),extent=float(extent))
points_images={}
for frame in a.read(a.OUT/'plan.json')['visual_frames']:
    name=f'frame_{frame:06d}';i=[r['frame'] for r in rows].index(name);row=rows[i]
    raw=Image.open(row['image']).convert('RGB');person=np.asarray(row['initial_box_xyxy']);person[3]=person[1]+.3*(person[3]-person[1])
    box=crop_box(person,raw.size,20);crop=raw.crop(box);overlay=crop.copy();draw=ImageDraw.Draw(overlay)
    targets=observations['original_mapped'][i]
    for ids,color,radius in [(np.arange(67,118),'magenta',2),(a.HEAD,'lime',4)]:
        for x,y,c in targets[ids]:
            if c<=0:continue
            x-=box[0];y-=box[1];draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
    points_images[name]=save_image('特征点_124_头部输入语义_'+name+'_原图_绿身体鼻眼耳5_紫内脸51_原生像素',join([crop,overlay]),
        crop=list(map(int,box)),body_head5_confidence=targets[a.HEAD,2].tolist(),associated_count=int(observations['associated_count'][i]))
a.save(a.OUT/'visual-review.json',dict(updated=a.now(),state='generated_pending_actual_review',native=native,rest=rest,
    points=points_images,actual_visual_review=False,changed_original_pixels=False,images_text_free=True,
    counterfactual_not_new_fits=True,quality_accepted=False,gaussian_admitted=False))
print('SHAPE_HEAD5_REVIEW_READY',flush=True)
