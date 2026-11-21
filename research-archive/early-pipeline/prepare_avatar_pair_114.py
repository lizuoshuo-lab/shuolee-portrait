from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');B=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee';G=LAB/'repos/GaussianAvatars_SMPLX-shuolee'
PAIR=B/'current/avatar-repository-pair-114';SOURCE=B/'current/person-static-fullscene-camera-fresh-075'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not PAIR.exists()
PAIR.mkdir(); now=datetime.datetime.now().astimezone().isoformat()
repos={'gaussianavatar':{'url':'https://github.com/aipixel/GaussianAvatar','commit':'d981c62238ef64e89dcc04719d2ebbb4758b080a','implementation':'Official POP_no_unet stage1 network and canonical LBS, adapted to one static body and moving real cameras.'},'smplx':{'url':'https://github.com/jmaraval/GaussianAvatars_SMPLX','commit':'e44c1a6c5fbb7f76089f14982eba98d94e553214','implementation':'Published GaussianAvatars mesh binding/render/optimizer plus explicit local SMPL-X adapter. Upstream still lacks train_smplx.py and scene/smplx_gaussian_model.py.'}}
for arm in repos:
 r=PAIR/arm
 for d in ['data','inference','source-final']: (r/d).mkdir(parents=True,exist_ok=True)
 for name in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/time-camera-inputs.npz','inference/vitpose-keypoints.pt','camera-source.json']:
  shutil.copy2(SOURCE/name,r/name)
 save(r/'source-inputs.json',dict(created=now,from_scratch=True,raw_inputs_source=str(SOURCE),old_subject_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,old_motion_loaded=False,representation='shared_static_world',real_frames=175,initial_points=200000,steps=18000,repository=repos[arm]))
protocol=dict(created=now,round='114',user_confirmed_methods=repos,state='fresh_preparation',
 objective='Direct comparison of the two user-named human repositories, not two camera strategies.',
 diagnosis='113 reduced its estimated geometric proxy but did not visibly remove duplicate nose/mouth/glasses/body edges. Test structural neural-offset and mesh-bound avatar representations as two distinct methods.',
 same_conditions=dict(images=175,cameras='074 registered full-frame cameras; fixed for both',steps=18000,height=1536,seed=6060,initial_points=200000,source_inputs_sha256=sha(SOURCE/'data/training-inputs.json'),foreground_only_RGB=True,soft_mask_alpha_weight=.1,static_shared_body=True,prior_fit_steps=1200),
 fresh='Both arms independently rerun shared static SMPL-X fitting from original 2D/official prediction observations. Fresh Gaussian or POP network/feature parameters and empty optimizers; no prior learned fit or Gaussian checkpoint.',
 method_differences=['GaussianAvatar retains official stage1 neural UV geometry/RGB and isotropic opacity-one Gaussians; no dynamic stage2.','GaussianAvatars_SMPLX retains mesh-local offsets, anisotropic covariance, opacity and SH; local SMPLX model missing upstream is explicitly labeled.','Dataset/resolution/foreground and static adapters are local; this is not full official benchmark reproduction.'],
 review=dict(raw_frames=[1,34,70,100,136,175],regions=['nose-mouth-eyes-glasses','shoulder-sleeve','collar-hem','legs-shoes'],all_175_boundary_statistics=True,same_fixed_camera=True,same_crop_background_exposure=True,novel_view_path='same150frame path as prior113, no ground truth claim',baseline='113 only as frozen review reference'),
 stop=['nonfinite values or gradients','camera/input mismatch','insufficient native projection/gradient/roundtrip validation','catastrophic loss of full person','18k maximum; no automatic hyperparameter sweep'],
 exports='World-coordinate standard and +Y PLY, full covariance and SH, native readback and actual OpenFile Gaussian/point UI before each latest publication.',
 quality_accepted=False,goal_complete=False)
save(PAIR/'protocol.json',protocol)
# Derive a shared STATIC fitter from the audited all-observation fitter. No
# previous fitted parameter is read. All 175 views see one identical mesh.
src=LAB/'repos/GVHMR-shuolee/shuolee/fit_fresh_sequence.py';text=src.read_text(encoding='utf-8-sig')
def replace(a,b):
 global text
 assert a in text,a[:80]
 text=text.replace(a,b,1)
replace('batch_size=n,use_face_contour=False','batch_size=1,use_face_contour=False')
replace("rv,tv,sv=param((n,3)),param((n,3)),param(())","rv,tv,sv=param((1,3)),param((1,3)),param(())")
replace("shared,temporal,bd=param((1,21,3)),param((n,21,3)),param((1,10))","shared,temporal,bd=param((1,21,3)),torch.zeros((1,21,3),device='cuda'),param((1,10))")
replace('hd=param((n,2,45))','hd=param((1,2,45))')
replace('latent=torch.nn.Parameter(z0.clone())',"latent=torch.nn.Parameter(z0.mean(0,keepdim=True).clone())\n    R0g=g(Rotation.from_matrix(R0).mean().as_matrix())[None]\n    t0g=g(np.median(t0,axis=0))[None]\n    pose0=g(np.stack([Rotation.from_matrix(pose0[:,j].cpu().numpy()).mean().as_matrix() for j in range(21)]))[None]\n    left0=left0.median(0,keepdim=True).values;right0=right0.median(0,keepdim=True).values")
replace("[shared,temporal,bd,hd,latent]","[shared,bd,hd,latent]")
replace('pose=matrix_to_axis_angle(pr).reshape(n,63)','pose=matrix_to_axis_angle(pr).reshape(1,63)')
replace('betas=beta.expand(n,-1)','betas=beta')
replace('world=lambda x:(x-p)@R.transpose(1,2)*s+t[:,None]','world=lambda x:((x-p)@R.transpose(1,2)*s+t[:,None]).expand(n,-1,-1)')
replace('R=R,t=t,s=s,body=pose,betas=beta,pr=pr,lh=lh,rh=rh,pelvis=p,corr=corr','R=R.expand(n,-1,-1),t=t.expand(n,-1),s=s,body=pose.expand(n,-1),betas=beta,pr=pr.expand(n,-1,-1,-1),lh=lh.expand(n,-1),rh=rh.expand(n,-1),pelvis=p.expand(n,-1,-1),corr=corr')
replace("reshape(n,21,3,3)","reshape(1,21,3,3).expand(n,-1,-1,-1)")
replace("+.5*contact+", "+0*contact+")
replace("deformation='SO(3) updates without per-subject hard rotation bounds; soft pretrained and VPoser priors'", "deformation='One shared static pose/shape/root/translation for every camera; no per-frame parameters. Old floor term disabled because no fresh floor-world mapping is assumed'")
replace("state='fresh_all_frame_fit_visual_review_pending'", "state='fresh_shared_static_fit_visual_review_pending'")
text='# Shared-static adapter for the two repository comparison. Derived from '+str(src)+'\n'+text
(G/'fit_avatar_pair_static_114.py').write_text(text,encoding='utf-8')
save(PAIR/'source-audit.json',dict(repositories=repos,source_fitter_sha256=sha(src),static_fitter_sha256=sha(G/'fit_avatar_pair_static_114.py'),old_gaussians_read=False,training_started=False))
plan=B/'long-term-plan.md';old=plan.read_text(encoding='utf-8-sig')
plan.write_text('<!-- AVATAR_REPOSITORY_PAIR_114_CURRENT -->\n\n用户已确认比较 aipixel/GaussianAvatar 全身路线与 jmaraval/GaussianAvatars_SMPLX。分别从原始观测重新拟合静态人体并从头训练，同175图像、20万初始点、18k步和固定相机验收。SMPLX缺失的公开入口使用明示的本地适配，不称官方完整复现。面部和全身重影、未知视角、兼容PLY均验收；每轮更新右侧查看器。113已完成，重影仍在，质量未通过。以下为历史记录。\n\n'+old,encoding='utf-8')
state=read(B/'work-status.json');state.update(updated=now,local_training_running=False,goal_accepted=False)
state['next_bounded_work']=dict(scope=protocol['objective'],repositories=repos,run=str(PAIR),state='source_audited_preparing_fresh_static_fits')
save(B/'work-status.json',state)
print(json.dumps(dict(pair=str(PAIR),prepared=True,methods=list(repos))))
