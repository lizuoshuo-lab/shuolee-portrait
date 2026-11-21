"""Read-only CPU counterfactual: outside-camera pixels are unobserved, not background.

No training, source edits, changed cameras, or admitted replacement initialization.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import numpy as np, torch, cv2
from PIL import Image
from capture119_clothed import ROOT,REC,rt
torch.set_num_threads(4)
root=ROOT/'sofa';out=REC/'sofa-fov-diagnostic';out.mkdir(exist_ok=True)
assert not (out/'result.json').exists(), 'Do not replace a completed diagnostic'
rt.save(out/'serialization-repair.json',dict(previous_error='numpy.int64 crop not JSON serializable',repair='Convert crop coordinates to Python integers; repeat the same read-only CPU calculation.'))
cfg=rt.read(root/'config.json');fit=dict(np.load(root/'fit/fitted.npz'))
rows=rt.read(root/'data/training-inputs.json')['records'];scale=float(fit['world_scale'])
R=fit['root_rotation'][0].astype(float);t=fit['world_translation'][0].astype(float)
local=(fit['vertices_world'][0]-t)@R/scale
axes=[np.arange(local[:,j].min()-cfg['surface_padding_m'],local[:,j].max()+cfg['surface_padding_m'],cfg['surface_spacing_m']) for j in range(3)]
shape=tuple(len(x) for x in axes);grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
points=torch.tensor(grid@R.T*scale+t,dtype=torch.float32,device='cpu')
votes=torch.zeros(len(points),dtype=torch.int16);valid_count=torch.zeros_like(votes)
kernel=np.ones((5,5),np.uint8);frame_counts=[]
with torch.no_grad():
 for row in rows:
    assert rt.sha(row['mask'])==row['mask_sha256']
    mask=cv2.dilate((np.array(Image.open(row['mask']).convert('L'))>127).astype(np.uint8),kernel)
    im=torch.tensor(mask,dtype=torch.float32)[None,None];E=torch.tensor(row['w2c'],dtype=torch.float32);K=torch.tensor(row['K'],dtype=torch.float32)
    wh=torch.tensor(row['wh'],dtype=torch.float32);outside=0
    for begin in range(0,len(points),250000):
        chunk=points[begin:begin+250000];cam=chunk@E[:3,:3].T+E[:3,3];q=cam@K.T;uv=q[:,:2]/q[:,2:].clamp_min(.001)-.5
        normal=2*uv/(wh-1)-1;inside=torch.nn.functional.grid_sample(im,normal[None,:,None],align_corners=True,padding_mode='zeros')[0,0,:,0]>.5
        good=(cam[:,2]>.001)&(uv[:,0]>=0)&(uv[:,0]<=wh[0]-1)&(uv[:,1]>=0)&(uv[:,1]<=wh[1]-1)
        votes[begin:begin+len(chunk)]+=(inside&(cam[:,2]>.001)).to(torch.int16)
        valid_count[begin:begin+len(chunk)]+=good.to(torch.int16)
        outside+=int((~good).sum())
    frame_counts.append(dict(frame=row['frame'],out_of_image_proposal_voxels=outside))
old=votes.numpy()/len(rows)>cfg['surface_vote_fraction']
# This threshold only bounds this read-only counterfactual. It is not a new training gate.
minimum=int(np.ceil(.5*len(rows)))
counter=(valid_count.numpy()>=minimum)&(votes.numpy()/np.maximum(valid_count.numpy(),1)>cfg['surface_vote_fraction'])
recovered=counter&~old
pix=[];w=1367;h=768;world=points.numpy()
for idx in [0,17,35,52,70,88]:
    row=rows[idx];E=np.asarray(row['w2c']);K=np.diag([w/row['wh'][0],h/row['wh'][1],1])@np.asarray(row['K'])
    raw=Image.open(row['image']).convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    mask=np.array(Image.open(row['mask']).resize((w,h),Image.Resampling.NEAREST))>127
    silhouettes=[]
    for chosen in [old,counter]:
        cam=world[chosen]@E[:3,:3].T+E[:3,3];q=cam@K.T;uv=np.rint(q[:,:2]/q[:,2:]-0.5).astype(int)
        good=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        canvas=np.zeros((h,w),np.uint8);canvas[uv[good,1],uv[good,0]]=255
        silhouettes.append(canvas)
    ys,xs=np.where(mask);box=tuple(map(int,(max(0,xs.min()-15),max(0,ys.min()-15),min(w,xs.max()+16),min(h,ys.max()+16))))
    imgs=[raw.crop(box)]+[Image.fromarray(s).convert('RGB').crop(box) for s in silhouettes]
    board=Image.new('RGB',(imgs[0].width*3,imgs[0].height))
    for k,img in enumerate(imgs):board.paste(img,(k*img.width,0))
    board.save(out/f'{row["frame"]}_raw_oldvote_validvote.png')
    pix.append(dict(frame=row['frame'],crop=box,columns=['raw','original_all_views_denominator_voxel_projection','counterfactual_visible_views_denominator_voxel_projection']))
value=dict(created=rt.now(),subject='sofa',read_only=True,training_inputs_changed=False,training_restarted=False,
    cpu_only=True,all_views=len(rows),grid_shape=shape,source_code_sha256=rt.sha(__file__),original_camera_sha256=rt.sha(root/'data/training-inputs.json'),
    original_voxel_count_cpu=int(old.sum()),original_voxel_count_gpu_recorded=rt.read(root/'clothing-init/audit.json')['occupied_voxels'],
    diagnostic_min_valid_views=minimum,counterfactual_voxels=int(counter.sum()),recovered_voxels=int(recovered.sum()),
    recovered_local_bounds=[grid[recovered].min(0).tolist(),grid[recovered].max(0).tolist()] if recovered.any() else None,
    recovered_valid_view_quantiles={str(q):float(np.quantile(valid_count.numpy()[recovered],q)) for q in [0,.5,1]} if recovered.any() else {},
    findings='Original grid_sample zero padding and division by all89 views count outside-image geometry as negative silhouette evidence. This counterfactual changes only visibility accounting; it is not a calibrated replacement model.',
    limits=['Counterfactual is voxel occupancy, not trained Gaussian rendering or physical truth.','Visible mask errors, occlusion and camera/motion errors still apply.','50percent minimum support is a diagnostic bound, not a parameter sweep or admitted training value.','GPU and CPU floating-point edge decisions may differ.'],
    frame_counts=frame_counts,images=pix,actual_visual_review_pending=True)
rt.save(out/'result.json',value)
print({k:value[k] for k in ['original_voxel_count_cpu','original_voxel_count_gpu_recorded','counterfactual_voxels','recovered_voxels','recovered_valid_view_quantiles']})
