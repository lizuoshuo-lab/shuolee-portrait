"""Fixed-shape PnP feasibility only: never changes the trained checkpoint."""
import numpy as np
import cv2
from PIL import Image,ImageDraw
import fit_demo_flame_face123 as f
from smplify_score_contract123 import relative_confidence
from run_official_gaussians123 import read,save,sha,now

def main():
    out=f.OUT/'rigid-pose-diagnostic';assert not out.exists();out.mkdir()
    save(out/'plan.json',dict(updated=now(),scope='One fixed EPNP solve per frame; freeze final estimated face shape/expression/points and K. Change only rigid head transform for feasibility; never overwrite parameters.',
        objective='Test whether the existing estimated 3D face can better explain the held same-detector labels with a rigid pose seed.',
        confidence='Use the previously frozen FIT mask. EPNP is unweighted and is not an identical original training objective; improvement does not uniquely isolate optimizer versus weighting.',
        counts=59,optimizer_steps=0,source_sha256=sha(f.OUT/'epoch-30/face-parameters-and-points.npz')))
    with np.load(f.OUT/'epoch-30/face-parameters-and-points.npz',allow_pickle=False) as z:points=dict(z)
    with np.load(f.DATA/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    rows=read(f.DATA/'data/inputs.json')['records'];conf,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    p=obs['keypoints'][:,23:91];side=int(np.ceil(np.max(p.max(1)-p.min(1))+40));stats=[];transforms=[];predictions=[]
    for i,row in enumerate(rows):
        target=p[i];xyz=points['landmarks3d_camera'][i];K=points['K_integer_centers'][i]
        weights=conf[i,23:91].copy();weights[:17]=0;weights[f.HELD]=0
        if 25<=i<=31:weights[27:36]=0;weights[48:68]=0
        fit=np.flatnonzero(weights>0)
        ok,rv,tv=cv2.solvePnP(xyz[fit].astype(np.float64),target[fit].astype(np.float64),K,None,flags=cv2.SOLVEPNP_EPNP)
        assert ok;rotation=cv2.Rodrigues(rv)[0];new=xyz@rotation.T+tv.ravel()
        assert np.linalg.det(rotation)>.9999 and new[:,2].min()>.1
        pred=new@K.T;pred=pred[:,:2]/pred[:,2:];predictions.append(pred);transforms.append((rotation,tv.ravel()))
        olderr=np.linalg.norm(points['landmarks2d_native'][i,:68]-target,axis=1);err=np.linalg.norm(pred[:68]-target,axis=1)
        stats.append(dict(frame=row['frame'],pose_delta_deg=float(np.linalg.norm(rv)*180/np.pi),
            old_held_median=float(np.median(olderr[f.HELD])),new_held_median=float(np.median(err[f.HELD])),
            old_held_max=float(max(olderr[f.HELD])),new_held_max=float(max(err[f.HELD])),
            old_fit_median=float(np.median(olderr[fit])),new_fit_median=float(np.median(err[fit])),
            device_occlusion=25<=i<=31))
        xy=np.floor((target.min(0)+target.max(0))/2-side/2).astype(int)
        image=Image.open(row['image']).convert('RGB').crop((*xy,*(xy+side)));panels=[image.copy(),image.copy(),image.copy()]
        for panel,coords,color in [(panels[1],points['landmarks2d_native'][i,17:68],'#b632db'),(panels[2],pred[17:68],'#1177ff')]:
            draw=ImageDraw.Draw(panel)
            for x,y in coords-xy:draw.ellipse((x-1,y-1,x+1,y+1),outline=color,width=1)
        joined=Image.new('RGB',(side*3,side),'white')
        for j,img in enumerate(panels):joined.paste(img,(side*j,0))
        joined.save(out/(row['frame']+'_原图_当前投影_固定脸型PnP诊断.png'))
    np.savez_compressed(out/'rigid-head-correction.npz',rotation=np.stack([a for a,b in transforms]),translation=np.stack([b for a,b in transforms]),landmarks2d_native=np.stack(predictions))
    result=dict(updated=now(),state='readonly_feasibility_complete',frames=59,rows=stats,
        median_old_held=float(np.median([s['old_held_median'] for s in stats])),median_new_held=float(np.median([s['new_held_median'] for s in stats])),
        worst_new_held=float(max(s['new_held_max'] for s in stats)),improved_frame_held_medians=sum(s['new_held_median']<s['old_held_median'] for s in stats),
        head_pose_changed_for_diagnostic_only=True,K_changed=False,shared_shape_changed=False,training_checkpoint_changed=False,upstream_VHAP_training_run=False,
        caveat='Frozen estimated shape and same-detector labels. The point data are not measured 3D truth; no independent geometry acceptance.')
    save(out/'result.json',result);print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
