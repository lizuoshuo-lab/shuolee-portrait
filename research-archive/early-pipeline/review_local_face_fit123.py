"""Saved meshes compared with unchanged native crop boxes and original images."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from review_full_sequence123 import read,sha,relocate,joined,mesh_renderer

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle/local-face-input-control'
F=O.parent/'full94-shared-shape';OUT=O/'fit-review'

def stats(a):return dict(count=int(a.size),median=float(np.median(a)),mean=float(a.mean()),maximum=float(a.max()))
def main():
    done=read(O/'official-fits/result.json');assert done['state']=='complete_visual_review_pending'
    assert not OUT.exists();OUT.mkdir()
    rows={r['frame']:r for r in read(F/'data/training-inputs.json')['records']}
    mesh,renderer_source=mesh_renderer();results=[]
    for frame in read(O/'plan.json')['conditional_fit_budget']['frames']:
        paths=[F/'frames'/frame/'winner-camera-space.npz',O/'official-fits'/frame/'winner-camera-space.npz']
        data=[]
        for path in paths:
            with np.load(path,allow_pickle=False) as z:data.append(dict(z))
        old,new=data;row=rows[frame]
        assert all(str(d['frame_id'].item())==frame for d in data)
        assert np.array_equal(old['betas'],new['betas']) and np.array_equal(old['faces'],new['faces'])
        assert all(np.array_equal(d['K_integer_centers'],row['K_integer_centers']) for d in data)
        fixed=new['baseline_native_keypoints'][0];changed=new['observed_native_keypoints'][0]
        assert np.array_equal(fixed[:67],changed[:67]) and np.array_equal(fixed[:,2],changed[:,2])
        raw=Image.open(relocate(row['image'])).convert('RGB');assert sha(relocate(row['image']))==row['source_sha256']
        overlays=[mesh(raw,np.array([0,0,*raw.size]),d) for d in data]
        images=[]
        dest=OUT/frame;dest.mkdir()
        for item in read(F/'frames'/frame/'saved-mesh-review.json')['files']:
            if item['panels']!='原图_官方SMPLX透明覆盖':continue
            label=next((s for s in ['全身','面部','双脚','左手','右手'] if '_'+s+'_' in Path(item['path']).name),None)
            if label is None:continue
            box=item['crop_integer_xyxy'];panel=joined([raw.crop(box),*[im.crop(box) for im in overlays]])
            name=f'20261030_局部人脸输入拟合对照_{frame}_{label}_左原图_中RTMW_右FAN.png'
            path=Path('D:/00/记录/005')/('smpl-x' if label=='全身' else '分析')/name
            assert not path.exists();panel.save(path);panel.save(dest/f'{label}.png')
            images.append(dict(label=label,path=str(path),review=str(dest/f'{label}.png'),sha256=sha(path),crop_xyxy=box))
        groups=dict(body17=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]),feet6=np.arange(19,25),left_hand21=np.arange(25,46),right_hand21=np.arange(46,67))
        valid=fixed[:,2]>0;valid[[1,9,12]]=False
        metrics=[];cross=[]
        for d in data:
            prediction=d['joints_projected_native'][0];errors=np.linalg.norm(prediction-fixed[:,:2],axis=1)
            metrics.append({name:stats(errors[idx[valid[idx]]]) for name,idx in groups.items()})
            cross.append(dict(vs_fixed_rtmw51=stats(np.linalg.norm(prediction[67:118]-fixed[67:118,:2],axis=1)),
                              vs_new_fan51=stats(np.linalg.norm(prediction[67:118]-changed[67:118,:2],axis=1))))
        results.append(dict(frame_id=frame,images=images,source_image_sha256=row['source_sha256'],
            source_fit_hashes={str(p):sha(p) for p in paths},non_face_same_targets_before=metrics[0],non_face_same_targets_after=metrics[1],
            face_residuals=dict(baseline_fit=cross[0],new_face_input_fit=cross[1]),
            same_shape_camera_confidence=True,official_objective_optimizer_unchanged=True))
    result=dict(state='complete_actual_visual_review_pending',rows=results,actual_visual_review=False,
        no_new_fitting=True,GPU_calls=0,no_resize=True,no_image_text=True,
        independent_ground_truth=False,quality_accepted=False,
        caution='Face target sets differ. Their own-target losses/residuals cannot establish improvement. Review originals and unchanged body/feet/hands protection.',
        renderer_sha256=sha(renderer_source))
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),frames=[r['frame_id'] for r in results]),ensure_ascii=False))

if __name__=='__main__':main()
