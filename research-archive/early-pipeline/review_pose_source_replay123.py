"""All-frame fixed-ROI audit and changed-frame native image review; CPU only."""
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now
from replay_smplify_pose_source123 import OUT,OLD,SNAP,checked
from review_full_sequence123 import mesh_renderer,joined


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    checked();result=read(OUT/'result.json')
    assert result['gaussian_parameters_bitwise_unchanged'] and result['optimizer_updates']==0
    dest=OUT/'review';dest.mkdir(exist_ok=False)
    record=Path('/mnt/d/00/记录/demo');record.mkdir(exist_ok=True)
    rows=read(OLD/'data/training-inputs.json')['records']
    selected=set(read(OLD/'selection.json')['selected_frames'])
    boxes={r['frame']:r['images'] for r in read(SNAP/'review/result.json')['records']}
    with np.load(OUT/'pose-replay.npz',allow_pickle=False) as z:data=dict(z)
    render,source=mesh_renderer();records=[]
    for i,row in enumerate(rows):
        f=row['frame'];assert str(data['frame_ids'][i])==f
        raw=Image.open(row['image']).convert('RGB')
        a=Image.open(OUT/(f+'_old.png')).convert('RGB');b=Image.open(OUT/(f+'_new.png')).convert('RGB')
        assert raw.size==a.size==b.size==(2704,1520)
        mask=np.asarray(Image.open(row['mask']).convert('L'))>127
        target=np.asarray(raw,dtype=np.float32)
        err={k:np.abs(np.asarray(v,dtype=np.float32)-target).mean(-1)/255 for k,v in [('old',a),('new',b)]}
        regions={};images=[]
        if data['changed'][i]:
            meshes=[render(raw,np.array([0,0,*raw.size]),dict(vertices_camera=v[i:i+1],faces=data['faces'],K_integer_centers=np.asarray(row['K_integer_centers']))) for v in (data['old_vertices'],data['new_vertices'])]
        for item in boxes[f]:
            region=item['region'];x0,y0,x1,y1=box=item['crop_integer_xyxy'];m=mask[y0:y1,x0:x1]
            if m.any():regions[region]={k:float(v[y0:y1,x0:x1][m].mean()) for k,v in err.items()}
            if data['changed'][i]:
                panel=joined([p.crop(box) for p in (raw,meshes[0],a,meshes[1],b)])
                name=f'高斯_姿态来源只读对照_{f}_{region}_原图_早期mesh_早期高斯_修正mesh_修正高斯.png'
                path=record/name;panel.save(path)
                images.append(dict(region=region,path=str(path),sha256=sha(path),crop_integer_xyxy=box,
                    panels=['source','early_fit_mesh','early_pose_Gaussian','reviewed_fit_mesh','reviewed_pose_Gaussian'],
                    text_added=False,resampling=False))
        records.append(dict(frame=f,pose_changed=bool(data['changed'][i]),train8=f in selected,regions=regions,images=images))
    records.sort(key=lambda r:r['frame'])
    groups={}
    for name,rs in [('changed12',[r for r in records if r['pose_changed']]),('unchanged47',[r for r in records if not r['pose_changed']]),('train8',[r for r in records if r['train8']]),('excluded51',[r for r in records if not r['train8']]),('all59',records)]:
        groups[name]={}
        for region in ('body','face','left_hand','right_hand','feet'):
            values=[r['regions'][region] for r in rs if region in r['regions']]
            changes=np.array([v['new']/v['old']-1 for v in values])
            groups[name][region]=dict(frames=len(values),improved_frames=int((changes<-1e-6).sum()),
                old_mae_mean=float(np.mean([v['old'] for v in values])),new_mae_mean=float(np.mean([v['new'] for v in values])),
                relative_change_median=float(np.median(changes)),relative_change_mean=float(changes.mean()))
    pages=[];changed=[r for r in records if r['pose_changed']]
    for region in ('body','face','left_hand','right_hand','feet'):
        count=4 if region=='body' else 6
        for start in range(0,len(changed),count):
            batch=changed[start:start+count];pictures=[]
            for r in batch:
                item=next(x for x in r['images'] if x['region']==region)
                im=Image.open(item['path']).convert('RGB')
                if region=='body':
                    s=min(1,370/im.height);im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.LANCZOS)
                pictures.append(im)
            width=max(p.width for p in pictures);height=sum(p.height for p in pictures)
            canvas=Image.new('RGB',(width,height),'white');y=0
            for p in pictures:canvas.paste(p,(0,y));y+=p.height
            path=dest/f'{region}-{start//count+1:02d}.png';canvas.save(path)
            pages.append(dict(region=region,path=str(path),sha256=sha(path),frames=[r['frame'] for r in batch]))
    save(dest/'result.json',dict(updated=now(),state='ready_for_actual_visual_review',records=records,groups=groups,pages=pages,
        native_images=60,frame_count=59,changed_frame_count=12,renderer_source=str(source),renderer_sha256=sha(source),
        metric_scope='Fixed-mask native RGB MAE; detector-fitted meshes and training images are not independent physical ground truth.',
        review_scope='All12 changed pose frames, five regions; unchanged47 have replay equality controls.',
        quality_accepted=False,actual_visual_review=False,new_Gaussian_training=False))
    print('POSE_SOURCE_REVIEW_READY',groups['changed12'],flush=True)


if __name__=='__main__':main()
