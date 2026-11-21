"""Native same-frame review of actual 022 GVHMR predictions."""
import ast

import numpy as np
from PIL import Image

from official_smplifyx_entry123 import sha
from review_full_sequence123 import joined, mesh_renderer, relocate
from run_leafwoman_gvhmr143 import ARCHIVE, CONTACTS, OFFICIAL, OUT, atomic, read


def main():
    plan=read(OUT/'plan.json');result=read(OUT/'result.json')
    assert result['real_prediction'] and result['plan_sha256']==sha(OUT/'plan.json')
    pred=OUT/'prediction-incam.npz'
    assert sha(pred)==result['prediction_sha256'] and sha(ARCHIVE)==plan['baseline_sha256']
    assert sha(CONTACTS)==plan['contacts_sha256'] and not (OUT/'native-review.json').exists()
    contacts={r['frame']:r for r in read(CONTACTS)['records']}
    tree=ast.parse((OFFICIAL/'smplifyx/utils.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if getattr(n,'name',None)=='smpl_to_openpose')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(OFFICIAL),'exec'),scope)
    bodymap=scope['smpl_to_openpose'](use_face_contour=False)[:25]
    renderer,renderer_path=mesh_renderer()
    visuals=[];metrics=[]
    with np.load(pred,allow_pickle=False) as new,np.load(ARCHIVE,allow_pickle=False) as old:
        assert np.array_equal(new['faces'],old['faces']) and np.array_equal(new['K_integer_centers'],old['K_integer_centers'])
        assert old['frame_ids'].tolist()==plan['comparison_roster']
        gv_frames=new['frame_ids'].tolist();K=old['K_integer_centers'];faces=old['faces']
        gv_joints=new['joints_camera'];old_obs=old['observed_native_keypoints'];old_uv=old['joints_projected_native']
        old_mesh,gv_mesh=old['vertices_camera'],new['vertices_camera']
        for i,frame in enumerate(plan['comparison_roster']):
            j=gv_frames.index(frame)
            q=gv_joints[j,bodymap[2:8]]@K.T
            assert (q[:,2]>0).all()
            observed=old_obs[i,2:8]
            active=observed[:,2]>0
            errors_old=np.linalg.norm(old_uv[i,2:8][active]-observed[active,:2],axis=1)
            errors_new=np.linalg.norm((q[:,:2]/q[:,2:])[active]-observed[active,:2],axis=1)
            metrics.append(dict(frame=frame,old_arm_errors_px=errors_old.tolist(),gvhmr_arm_errors_px=errors_new.tolist()))
            if frame not in ('frame_000007','frame_000020','frame_000033'):continue
            item=contacts[frame]['paths']['body']; path=relocate(item['path']); assert sha(path)==item['sha256']
            with Image.open(path) as im:raw=im.convert('RGB')
            b=np.asarray(item['crop_original_xyxy'])
            assert raw.size==tuple(b[2:]-b[:2])
            images=[raw]
            for v in (old_mesh[i:i+1],gv_mesh[j:j+1]):
                images.append(renderer(raw,b,dict(vertices_camera=v,K_integer_centers=K,faces=faces)))
            p=OUT/(frame+'-gvhmr-native.png');joined(images).save(p)
            visuals.append(dict(frame=frame,file=str(p),sha256=sha(p)))
    atomic(OUT/'native-review.json',dict(prediction_sha256=sha(pred),renderer_sha256=sha(renderer_path),frames=metrics,
        images=visuals,native_panels=['source','frozen_SMPLify-X','GVHMR_keypoint_only'],
        same_detector_points_as_model_input=True,visual_review_pending=True,quality_accepted=False))


if __name__=='__main__':
    main()
