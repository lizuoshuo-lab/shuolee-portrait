"""Read-only CPU audit of the actual grass fitter, not a new fit or camera."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, importlib.util, json, sys, time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/grass-exploratory/provenance-audit/mapping-audit'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
FROZEN=ROOT.parent/'runtime'
ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
MODEL=ASSETS/'smplx/SMPLX_NEUTRAL.npz'
REGDIR=ASSETS/'gvhmr/hmr4d/utils/body_model'
SOURCE=FROZEN/'src/shuolee_portrait/fitting.py'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):
    x=np.asarray(x).reshape(-1);x=x[np.isfinite(x)]
    return dict(n=int(len(x)),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max())) if len(x) else dict(n=0)
def inputs():
    return [ROOT/'data/inputs.json',ROOT/'data/observations.npz',ROOT/'inference/prediction-postprocessed.npz',ROOT/'fit/initialization.npz',ROOT/'fit/fitted.npz',ROOT/'fit/observation-weights.npz',SOURCE,MODEL,REGDIR/'smplx2smpl_sparse.pt',REGDIR/'smpl_coco17_J_regressor.pt']+list((O/'official').rglob('*.*'))
def register():
    assert not (O/'plan.json').exists(), 'Read the existing audit rather than repeat it.'
    save(O/'plan.json',dict(created=now(),purpose='Separate wrong semantic/index/coordinate bindings from incomplete fitting constraints and inconsistent shared-camera observations.',
        user_steering=['人体都是扭曲的肯定是3d到smpl的映射错了呀','我看smplify是能映射出正确的身体脸手来呀'],
        cpu_only=True,fit_steps=0,camera_changes=False,training_started=False,
        official_commit='68f8536707f43f4736cdd75a19b18ede886a4d53',
        checks=['Exact official COCO25 reordering into COCO17, feet, hands and static face51; do not relabel regressed joints as direct SMPL-X joints.',
                'SMPL-X forward of all94 saved per-frame prior parameters against actual127 incam_joints; root-centered world transform round trip.',
                'Full composed6890x10475 regressor versus official GVHMR algebra, row sums and coordinate equivariance.',
                'Reconstruct saved final model from pose,shape,hands,root,scale and translation; compare full10475vertices and joints.',
                'Per-frame prior versus shared-mean initialization versus final projected body; root/pose spread under fixed cameras.',
                'Compare optimized variables and staged fitting against official SMPLify-X; no numerical fit.'],
        selected_visual_slots=[0,31,62],selection='Fixed same diagnostic views, not selected by residual.',
        limitations=['COCO/OpenPose anatomical correspondences are not physical multi-view material identities.','Regressor/direct-joint differences do not by themselves prove a permutation bug.','No new official SMPLify-X execution or successful fit is claimed.'],
        inputs={str(p):sha(p) for p in inputs()}))

def run():
    assert (O/'plan.json').exists() and not (O/'result.json').exists()
    start=time.monotonic();plan=read(O/'plan.json')
    assert all(sha(Path(p))==s for p,s in plan['inputs'].items())
    import torch, smplx
    from smplx.joint_names import JOINT_NAMES
    from PIL import Image,ImageDraw
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    rows=read(ROOT/'data/inputs.json')['records'];n=len(rows)
    data=dict(np.load(ROOT/'data/observations.npz'));pred=dict(np.load(ROOT/'inference/prediction-postprocessed.npz'))
    fit=dict(np.load(ROOT/'fit/fitted.npz'));initial=dict(np.load(ROOT/'fit/initialization.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
    slots=np.array([r['official_prediction_slot'] for r in rows]);E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows])
    K[:,:2,2]-=.5
    t=lambda x:torch.as_tensor(x,dtype=torch.float32)
    model=smplx.SMPLX(str(MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=False,batch_size=1,use_face_contour=False)
    # Execute only the inspected pure mapping function, not any fitting entry.
    tree=ast.parse((O/'official/smplifyx/utils.py').read_text())
    node=next(q for q in tree.body if isinstance(q,ast.FunctionDef) and q.name=='smpl_to_openpose')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'official-smpl_to_openpose','exec'),scope)
    official=scope['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    # OpenPose25 index for the explicitly named COCO17 order, not same-number slots.
    coco_names=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
    coco_from25=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    direct_ids=official[:25][coco_from25]
    assert [JOINT_NAMES[i] for i in direct_ids]==coco_names
    foot_names=['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']
    foot_ids=[JOINT_NAMES.index(x) for x in foot_names]
    hand_names=['wrist','thumb1','thumb2','thumb3','thumb','index1','index2','index3','index','middle1','middle2','middle3','middle','ring1','ring2','ring3','ring','pinky1','pinky2','pinky3','pinky']
    hand_ids=[JOINT_NAMES.index(s+'_'+j) for s in ['left','right'] for j in hand_names]
    assert np.array_equal(foot_ids,official[19:25]) and np.array_equal(hand_ids,official[25:67])
    assert np.array_equal(official[67:],np.arange(76,127))
    M=torch.load(REGDIR/'smplx2smpl_sparse.pt',weights_only=True,map_location='cpu').to_dense()
    J=torch.load(REGDIR/'smpl_coco17_J_regressor.pt',weights_only=True,map_location='cpu');reg=J@M
    assert list(M.shape)==[6890,10475] and list(reg.shape)==[17,10475]
    def forward(params,batch=8):
        vs=[];js=[]
        total=max(len(v) for v in params.values())
        for a in range(0,total,batch):
            b=min(a+batch,total);args={k:t(v[a:b] if len(v)>1 else np.repeat(v,b-a,axis=0)) for k,v in params.items()}
            for key,dim in [('global_orient',3),('transl',3),('left_hand_pose',45),('right_hand_pose',45),('expression',10),('jaw_pose',3),('leye_pose',3),('reye_pose',3)]:
                if key not in args:args[key]=torch.zeros((b-a,dim))
            z=model(**args);vs.append(z.vertices.numpy());js.append(z.joints.numpy())
        return np.concatenate(vs),np.concatenate(js)
    params={k[len('incam_'):]:pred[k][slots] for k in ['incam_betas','incam_body_pose','incam_global_orient','incam_transl']}
    pv,pj=forward(params);prior_joint_error=np.abs(pj-pred['incam_joints'][slots])
    baseparams={k:v for k,v in params.items() if k not in ('global_orient','transl')};bv,bj=forward(baseparams)
    predR=Rotation.from_rotvec(params['global_orient']).as_matrix();pelvis=pred['incam_joints'][slots,0]
    # SMPL-X rotates about its shaped pelvis; global translation is not itself pelvis.
    pc=np.einsum('nij,nvj->nvi',predR,bv-bj[:,:1])+pelvis[:,None]
    A=np.concatenate([E[:,:3,:3],-pelvis[:,:,None]],-1).reshape(-1,4);rhs=-E[:,:3,3].reshape(-1)
    sol=least_squares(lambda x:A@x-rhs,np.linalg.lstsq(A,rhs,rcond=None)[0],loss='soft_l1',f_scale=.05);s0=float(sol.x[3])
    R0=E[:,:3,:3].transpose(0,2,1)@predR;t0=np.einsum('nji,nj->ni',E[:,:3,:3],s0*pelvis-E[:,:3,3])
    world=np.einsum('nij,nvj->nvi',R0,bv-bj[:,:1])*s0+t0[:,None]
    pc_back=(np.einsum('nij,nvj->nvi',E[:,:3,:3],world)+E[:,:3,3][:,None])/s0
    fparams=dict(body_pose=fit['body_pose'][:1],betas=fit['betas'],left_hand_pose=fit['left_hand_pose'][:1],right_hand_pose=fit['right_hand_pose'][:1])
    fv,fj=forward(fparams);R=fit['root_rotation'][:1];scale=float(fit['world_scale']);trans=fit['world_translation'][:1]
    fw=np.einsum('nij,nvj->nvi',R,fv-fj[:,:1])*scale+trans[:,None]
    jw=np.einsum('nij,nvj->nvi',R,fj-fj[:,:1])*scale+trans[:,None]
    tri=model.faces_tensor[model.lmk_faces_idx].numpy();bary=model.lmk_bary_coords.numpy()
    face=(fv[:,tri]*bary[None,:,:,None]).sum(2)
    native_co=np.einsum('jv,nvc->njc',reg.numpy(),fv)
    intermediate=torch.einsum('kv,nvc->nkc',M,t(fv));composition=J@intermediate
    co_world=(native_co-fj[:,:1])@R.transpose(0,2,1)*scale+trans[:,None]
    forward_world_reg=np.einsum('jv,nvc->njc',reg.numpy(),fw)
    def project_cam(x):
        q=np.einsum('nij,nvj->nvi',K,x);return q[...,:2]/q[...,2:]
    def project_world(x):return project_cam(np.einsum('nij,nvj->nvi',E[:,:3,:3],np.broadcast_to(x,(n,*x.shape[1:])))+E[:,:3,3][:,None])
    pco=np.einsum('jv,nvc->njc',reg.numpy(),pv)
    initco=np.einsum('jv,nvc->njc',reg.numpy(),initial['vertices_world'])
    stages={'per_frame_prior_regressed':project_cam(pco),'shared_mean_initialization_regressed':project_world(initco),'final_regressed':fit['projected_coco17'],
            'final_official_direct_no_refit':project_world(jw[:,direct_ids])}
    mask=weights['body']>.25;target=weights['body_target'];stage_err={}
    for name,uv in stages.items():
        d=np.linalg.norm(uv-target,axis=-1)
        stage_err[name]=dict(all=stats(d),reliable=stats(d[mask]),per_joint={nm:stats(d[mask[:,j],j]) for j,nm in enumerate(coco_names)})
    rot_mean=Rotation.from_matrix(R0).mean();rot_spread=(rot_mean.inv()*Rotation.from_matrix(R0)).magnitude()*180/np.pi
    pose=Rotation.from_rotvec(pred['incam_body_pose'][slots].reshape(-1,3)).as_matrix().reshape(n,21,3,3)
    spreads={}
    for j in range(21):
        rs=Rotation.from_matrix(pose[:,j]);spreads[JOINT_NAMES[j+1]]=stats((rs.mean().inv()*rs).magnitude()*180/np.pi)
    output=dict(created=now(),state='read_only_mapping_and_fitter_audit_complete',frames=n,seconds=time.monotonic()-start,
        official_source_commit=plan['official_commit'],actual_fit_backend='frozen_legacy_sofa_coarse',official_smplifyx_was_run=False,
        semantic=dict(coco17_names=coco_names,official_direct_smplx_indices=direct_ids.tolist(),legacy_body='GVHMR SMPL-X to SMPL vertex conversion then COCO17 regressor, not official direct SMPL-X joint selector',
            feet_same_as_official=True,hands_same_as_official=True,face_fixed51_same_as_official=True,face_barycentric_vs_native_joint_max=float(np.max(np.abs(face-fj[:,76:127]))),
            regressor_shape=list(reg.shape),regressor_row_sum_error_max=float((reg.sum(1)-1).abs().max()),
            composed_vs_two_step_max=float(np.max(np.abs(native_co-composition.numpy()))),
            translation_equivariance_max_world=float(np.max(np.abs(co_world-forward_world_reg))),
            direct_vs_regressed_model_units={nm:float(np.linalg.norm(native_co[0,j]-fj[0,direct_ids[j]])) for j,nm in enumerate(coco_names)},
            direct_vs_regressed_pixel_difference=stats(np.linalg.norm(stages['final_official_direct_no_refit']-stages['final_regressed'],axis=-1))),
        coordinate_checks=dict(saved_prior127_joint_max=float(prior_joint_error.max()),pelvis_centered_prior_vertices_max=float(np.abs(pc-pv).max()),camera_world_roundtrip_vertices_max=float(np.abs(pc_back-pv).max()),
            fitted_full_vertices_reconstruction_max_world=float(np.max(np.abs(fw-fit['vertices_world'][:1]))),fitted_full_joints_reconstruction_max_world=float(np.max(np.abs(jw-fit['joints_world'][:1]))),
            fitted_coco_world_max=float(np.max(np.abs(co_world-fit['coco17_world'][:1]))),all_root_rotation_det=stats(np.linalg.det(fit['root_rotation'])),initial_body_to_world_scale=s0),
        stage_body_errors_px=stage_err,world_root_rotation_spread_degrees=stats(rot_spread),per_joint_prior_rotation_spread_degrees=spreads,
        fitting_gaps=dict(optimized_variables=['shared_root_rotation','shared_translation','shared_scale','shared21body_pose_correction','10shape','hands_bounded_delta','vposer_auxiliary_latent'],
            expression_optimized=False,jaw_optimized=False,eye_pose_optimized=False,
            hands_axis_angle_delta_limit_rad=.25,hand_limit_is_componentwise_not_total_rotation=True,
            face_fixed51_weight_zero_slots=list(range(10))+list(range(19,31)),
            body_prior='Separate free axis-angle correction with soft distance to VPoser decode; not VPoser-decoded pose parameterization.',
            staged_schedule='Only root first150steps, then joint Adam; not official body/hand/face loss schedule.',
            anatomy_visibility='none',self_collision=False,
            framewise_prior_averaged_before_fit=True,unproven_static_consistency=True),
        limits=plan['limitations'],fit_steps=0,training_steps=0,source_inputs_unchanged=True)
    # Identical integer crop for all four panels; uniform resize only, no text.
    edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
    visuals=[]
    for slot in plan['selected_visual_slots']:
        row=rows[slot];image_path=Path(row.get('image',row.get('image_path','')));im=Image.open(image_path).convert('RGB')
        box=data['boxes'][slot];pad=.15*max(box[2]-box[0],box[3]-box[1]);x0,y0=max(0,int(box[0]-pad)),max(0,int(box[1]-pad));x1,y1=min(im.width,int(box[2]+pad)),min(im.height,int(box[3]+pad))
        crop=im.crop((x0,y0,x1,y1));panels=[crop.copy()]
        for key in ['per_frame_prior_regressed','shared_mean_initialization_regressed','final_regressed']:
            panel=crop.copy();draw=ImageDraw.Draw(panel);a=target[slot]-[x0,y0];b=stages[key][slot]-[x0,y0]
            for i,j in edges:
                draw.line([tuple(a[i]),tuple(a[j])],fill=(0,230,70),width=2)
                draw.line([tuple(b[i]),tuple(b[j])],fill=(255,0,200),width=2)
            for q in a:draw.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=(0,230,70))
            for q in b:draw.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),fill=(255,0,200))
            panels.append(panel)
        size=(round(crop.width*600/crop.height),600);canvas=Image.new('RGB',(size[0]*4,600),'white')
        for j,panel in enumerate(panels):canvas.paste(panel.resize(size,Image.Resampling.LANCZOS),(j*size[0],0))
        path=O/f'slot{slot:03d}_raw_prior_sharedinit_final.png';canvas.save(path)
        visuals.append(dict(slot=slot,path=str(path),sha256=sha(path),columns=['原图','逐帧先验','共用姿态均值初始化','最终粗拟合'],green='same2Dbodytarget',magenta='sameCOCO17regression_of_each3Dstage',crop=[x0,y0,x1,y1],uniform_resize_to=size))
    output['visuals']=visuals
    assert all(sha(Path(p))==s for p,s in plan['inputs'].items()),'Source mutation!'
    save(O/'result.json',output);print(json.dumps(output,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.run:run()
