"""Read-only stage-boundary decomposition and native-pixel mesh evidence."""
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import trace_smplifyx_stage19_123_v2 as control
from review_full_sequence123 import mesh_renderer, joined

b=control.base
OUT=b.OUT/'review'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    result=b.read(b.OUT/'result.json'); assert result['endpoint_parity_passed']
    data=b.load_npz(b.OUT/'trajectory.npz')
    args=b.arguments(); stages=[s for s in result['stages'] if s['kind']=='body']
    OUT.mkdir()
    sources=[Path(__file__),Path(b.entry.__file__),Path(args.official_root)/'smplifyx/fitting.py',
        Path(args.official_root)/'smplifyx/prior.py']
    inputs=[b.OUT/'result.json',b.OUT/'trajectory.npz',Path(args.assets_root)/'smplx/SMPLX_NEUTRAL.npz']
    plan=dict(updated=b.now(),state='prepared',body_forwards_max=10,cpu_seconds_max=120,device='cpu',optimizer_updates=0,
        original_loss_terms=True,native_crops_unchanged=True,no_text_on_images=True,
        source_hashes={str(p):b.sha(p) for p in sources},input_hashes={str(p):b.sha(p) for p in inputs})
    b.save(OUT/'plan.json',plan)
    started=time.monotonic()
    import torch
    import smplx
    torch.set_num_threads(1)
    utils,prior,fitting,_=b.entry.load_official(args.official_root)
    mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    body=smplx.SMPLX(str(Path(args.assets_root)/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
        use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
        joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).eval()
    loader=b.entry.module(Path(args.runtime_root)/'src/avatar_core/shuolee_vposer.py','_trace19_review_vposer')
    loader.MODEL=Path(args.assets_root)/'vposer/V02_05';loader.CODE=Path(args.runtime_root)/'third_party/human_body_prior'
    vposer,_,_=loader.load_vposer('cpu')
    cfg=b.entry.load_official_config(args.official_root)
    l2=lambda:prior.L2Prior()
    loss=fitting.create_loss(loss_type='smplify',rho=cfg['rho'],body_pose_prior=l2(),shape_prior=l2(),expr_prior=l2(),
        angle_prior=prior.SMPLifyAnglePrior(),jaw_prior=l2(),left_hand_prior=l2(),right_hand_prior=l2(),
        use_joints_conf=True,use_face=True,use_hands=True,interpenetration=False,dtype=torch.float32)
    K=torch.as_tensor(data['K_integer_centers'],dtype=torch.float32)
    target=torch.as_tensor(data['observed_native_keypoints'])
    terms=[]
    with torch.no_grad():
        for s in stages:
            for index in [s['before_state'],s['after_state']]:
                assert time.monotonic()-started<120
                prefix=f'state_{index:03d}'
                body.reset_params(**{name:data[prefix+'_parameter_'+name] for name,_ in body.named_parameters()})
                latent=torch.as_tensor(data[prefix+'_pose_embedding'])
                translation=torch.as_tensor(data[prefix+'_camera_translation'])
                geometry=body(body_pose=vposer.decode(latent)['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True)
                def camera(points):
                    q=(points+translation[:,None])@K.T
                    return q[...,:2]/q[...,2:]
                projection=camera(geometry.joints)
                vm=float((geometry.vertices+translation[:,None]-torch.as_tensor(data[prefix+'_vertices_camera'])).abs().max())
                px=float((projection-torch.as_tensor(data[prefix+'_joints_projected_native'])).abs().max())
                assert vm<2e-5 and px<.005,(index,vm,px)
                loss.reset_loss_weights(s['weights'])
                weights=torch.ones((1,118));weights[:,[1,9,12]]=0
                weights[:,25:67]=cfg['hand_joints_weights'][s['stage']]
                weights[:,67:]=cfg['face_joints_weights'][s['stage']]
                full=loss(geometry,camera=camera,gt_joints=target[...,:2],joints_conf=target[...,2],body_model_faces=body.faces_tensor.view(-1),
                    joint_weights=weights,pose_embedding=latent,use_vposer=True)
                item=dict(state=index,stage=s['stage'],label='before' if index==s['before_state'] else 'after',
                    latent_prior=float(latent.square().sum()*loss.body_pose_weight.square()),
                    shape_prior=float(loss.shape_prior(geometry.betas).sum()*loss.shape_weight.square()),
                    bending_prior=float(loss.angle_prior(geometry.full_pose[:,3:66]).sum()*loss.bending_prior_weight),
                    left_hand_prior=float(loss.left_hand_prior(geometry.left_hand_pose).sum()*loss.hand_prior_weight.square()),
                    right_hand_prior=float(loss.right_hand_prior(geometry.right_hand_pose).sum()*loss.hand_prior_weight.square()),
                    expression_prior=float(loss.expr_prior(geometry.expression).sum()*loss.expr_prior_weight.square()),
                    jaw_prior=float(loss.jaw_prior(geometry.jaw_pose*loss.jaw_prior_weight).sum()),
                    latent_squared_norm=float(latent.square().sum()),total=float(full),vertex_parity_m=vm,projection_parity_px=px)
                reference=next(r for r in result['evaluations'] if r['state']==index)
                assert abs(item['total']-reference['total'])<=max(.1,abs(reference['total'])*1e-5)
                terms.append(item)
    assert not torch.cuda.is_initialized()
    rows={r['frame']:r for r in b.read(b.common.BASE/'data/training-inputs.json')['records']}
    row=rows[b.FRAME];assert b.sha(row['image'])==row['source_sha256']
    raw=Image.open(row['image']).convert('RGB');assert list(raw.size)==row['wh']
    crops=b.read(b.common.BASE/'review'/(b.FRAME+'_saved-review.json'))
    mesh,renderer_source=mesh_renderer()
    record=Path('/mnt/d/00/记录/demo'); record.mkdir(exist_ok=True)
    files=[]
    for s in [stages[0],stages[4]]:
        states=[s['before_state'],s['after_state']]
        for region in ['body','face','left_hand','right_hand','feet']:
            crop=next(v['crop_integer_xyxy'] for v in crops['files'] if v['region']==region)
            image=raw.crop(crop); panels=[image]
            for index in states:
                prefix=f'state_{index:03d}'
                fit=dict(vertices_camera=data[prefix+'_vertices_camera'],faces=data['faces'],K_integer_centers=data['K_integer_centers'])
                panels.append(mesh(image,np.array(crop),fit))
            path=OUT/f'frame19_stage{s["stage"]+1}_{region}_raw_before_after.png'
            joined(panels).save(path)
            dest=record/f'SMPL-X_原版阶段追踪_frame_000019_阶段{s["stage"]+1}_{region}_原图_进入阶段_阶段结束.png'
            assert not dest.exists();dest.write_bytes(path.read_bytes())
            assert b.sha(path)==b.sha(dest)
            files.append(dict(path=str(path),record_path=str(dest),sha256=b.sha(path),stage=s['stage'],states=states,region=region,
                crop_integer_xyxy=crop,native_pixels=True,no_text_added=True,panels=['raw','before_stage','after_stage']))
    for path,digest in {**plan['source_hashes'],**plan['input_hashes']}.items():assert b.sha(path)==digest,path
    b.save(OUT/'analysis.json',dict(updated=b.now(),state='cpu_decomposition_and_native_images_ready',prior_terms=terms,files=files,
        elapsed_seconds=time.monotonic()-started,body_forwards=10,renderer_sha256=b.sha(renderer_source),
        actual_visual_review=False,quality_accepted=False,goal_complete=False,
        limits=['Depth-sorted CPU overlay for review, not exact rasterizer acceptance.', 'One frame and detector residuals, not all-sequence quality.']))
    print('STAGE19_CPU_REVIEW_READY',len(files),'images',flush=True)
    for row in terms: print(row['stage'],row['label'],row['latent_prior'],row['bending_prior'],row['left_hand_prior']+row['right_hand_prior'],flush=True)


if __name__=='__main__':main()
