"""Frozen-state CPU shape/head-input audit. No detector, fitter or candidate."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import official_smplifyx_entry123 as entry
from run_official_gaussians123 import ASSETS, read,save,sha,now

HERE=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=REC/'cross-subject-sh0-124'
CONTROL=ROOT/'face-absence-input124/indoor-transition-control'
OUT=ROOT/'shared-shape-head5-audit124'
OFFICIAL=REC/'smplifyx-reference-control/official'
HEAD=np.array([0,15,16,17,18])


def snapshots():
    files={s+'_reference':ROOT/(s+'-input/reference-fit/winner-camera-space.npz') for s in ['winterhat','sofa','indoor']}
    files.update({p.parent.parent.name+'_'+p.parent.name:p for p in sorted(CONTROL.glob('*/*/winner-camera-space.npz'))})
    assert len(files)==9
    return files


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    assert read(CONTROL/'conclusion.json')['actual_visual_review']
    paths=list(snapshots().values())+[Path(__file__),Path(entry.__file__),ASSETS/'smplx/SMPLX_NEUTRAL.npz',
        CONTROL/'conclusion.json',CONTROL/'plan.json',ROOT/'face-absence-input124/indoor-diagnostic-input.npz',
        ROOT/'indoor-input/data/inputs.json']+[OFFICIAL/k for k in entry.PINNED]
    import smplx
    paths += [Path(smplx.__file__).parent/'body_models.py',Path(smplx.__file__).parent/'lbs.py']
    for row in read(ROOT/'indoor-input/data/inputs.json')['records']:
        if row['frame'] in [f'frame_{i:06d}' for i in [38,39,40,41,53,61,105]]:paths.append(Path(row['image']))
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered_read_only',
        hypothesis='Separate shared-beta geometry, per-frame expression and still-active COCO head5 observations in saved reference and face51-abstention outputs. Do not infer actual shape from neutral-template preference.',
        frozen_snapshots={k:str(v) for k,v in snapshots().items()},
        budget=dict(cpu_seconds=120,optimizer_steps=0,new_fits=0,new_detectors=0,gpu_calls=0),
        controls=['Reconstruct all9 snapshots with original model/mapping and saved parameters; <=0.003px joint projection and5e-6m vertex parity.',
            'For3 reference betas, compare rest-shape geometry at zero expression, common fixed camera and zero shape only as a counterfactual.',
            'For indoor105 and original/abstention40, zero only expression or only beta at the SAME saved pose/camera; never refit or promote these geometries.',
            'Report all111 head5 confidences and face51 group membership from existing frozen archive, including all27 no-face frames.',
            'Compute original final-stage robust data gradients with respect to beta at saved105, grouped by head5/otherbody/hands/innerface; compare shape prior. Sensitivity only, not a new optimizer.'],
        guards=['Original source, data, K, beta, poses, masks, confidence and models remain unchanged.',
            'Neutral beta is not physical truth. Projection changes do not prove new shape quality or causal uniqueness.',
            'Frozen beta has no optimizer updates in continuation; its derivative is a diagnostic sensitivity only.',
            'No hidden-leg, gravity or real-camera trajectory claim.'],
        visual_frames=[38,39,40,41,53,61,105],frozen={str(p):sha(p) for p in paths}))
    print('SHAPE_HEAD5_AUDIT_REGISTERED',flush=True)


def stats(x):
    x=np.asarray(x);return dict(min=float(x.min()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'progress.json').exists()
    plan=read(OUT/'plan.json')
    for path,digest in plan['frozen'].items():assert sha(path)==digest,path
    save(OUT/'progress.json',dict(updated=now(),state='cpu_forward_only',pid=os.getpid()))
    start=time.monotonic()
    try:
        import torch,smplx
        torch.set_num_threads(1)
        utils,_,_,_=entry.load_official(OFFICIAL)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        body=smplx.SMPLX(str(ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float64).cpu()
        def tensor(x,grad=False):return torch.tensor(x,dtype=torch.float64,requires_grad=grad)
        def forward(a,beta=None,expr=None):
            values={k:tensor(a[k]) for k in ('global_orient','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose')}
            values.update(body_pose=tensor(a['body_pose_axis_angle']),betas=tensor(a['betas']) if beta is None else beta,
                expression=tensor(a['expression']) if expr is None else expr,return_verts=True,return_full_pose=True)
            return body(**values)
        def camera(out,a):
            v=out.vertices+tensor(a['camera_translation'])[:,None]
            j=out.joints+tensor(a['camera_translation'])[:,None]
            q=j@tensor(a['K_integer_centers']).T
            return v,j,q[...,:2]/q[...,2:]
        arrays={};checks={};results={}
        for label,path in snapshots().items():
            with np.load(path,allow_pickle=False) as z:a=dict(z)
            arrays[label]=a
            with torch.no_grad():v,j,p=camera(forward(a),a)
            vd=float(np.max(np.abs(v.numpy()-a['vertices_camera'])));pd=float(np.max(np.abs(p.numpy()-a['joints_projected_native'])))
            checks[label]=dict(vertex_delta_m=vd,projection_delta_px=pd,passed=vd<=5e-6 and pd<=.003)
            assert checks[label]['passed'],(label,checks[label])
            results[label]=dict(beta=a['betas'].tolist(),beta_norm=float(np.linalg.norm(a['betas'])),
                expression_norm=float(np.linalg.norm(a['expression'])),jaw_radians=a['jaw_pose'].tolist())
        head_vertices=body.lbs_weights.argmax(-1).numpy()==15
        assert head_vertices.sum()>100
        neutral=np.zeros((1,10));rest={};shape_stats={}
        for label in ['winterhat_reference','sofa_reference','indoor_reference','neutral']:
            beta=neutral if label=='neutral' else arrays[label]['betas']
            a=arrays['indoor_reference'];rest_a={k:np.zeros_like(a[k]) for k in ['global_orient','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','body_pose_axis_angle','expression']}
            rest_a['betas']=beta
            with torch.no_grad():v=forward(rest_a).vertices[0].numpy()
            rest[label]=v
        for label,v in rest.items():
            delta=np.linalg.norm(v-rest['neutral'],axis=-1)
            shape_stats[label]=dict(bounds_m=np.ptp(v,axis=0).tolist(),displacement_from_neutral_m=stats(delta),head_displacement_m=stats(delta[head_vertices]))
        variants={};effects={}
        for label in ['indoor_reference','original_frame_000040','face_absence_frame_000040']:
            a=arrays[label];variants[label]={};effects[label]={}
            for kind,bb,ee in [('saved',None,None),('zero_expression',None,tensor(neutral)),('zero_beta',tensor(neutral),None)]:
                with torch.no_grad():v,j,p=camera(forward(a,beta=bb,expr=ee),a)
                variants[label][kind]=v.numpy()
                dv=np.linalg.norm(v.numpy()[0]-a['vertices_camera'][0],axis=-1)
                effects[label][kind]=dict(vertex_m=stats(dv),head_vertex_m=stats(dv[head_vertices]),joint_projection_px=stats(np.linalg.norm(p.numpy()[0]-a['joints_projected_native'][0],axis=-1)))
        src=ROOT/'face-absence-input124/indoor-diagnostic-input.npz'
        with np.load(src,allow_pickle=False) as z:d=dict(z)
        rows=read(ROOT/'indoor-input/data/inputs.json')['records'];assert d['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
        head_records=[]
        for i,row in enumerate(rows):
            original=d['original_mapped'][i];candidate=d['candidate_mapped'][i]
            assert np.array_equal(original[HEAD],candidate[HEAD])
            head_records.append(dict(frame=row['frame'],associated_count=int(d['associated_count'][i]),
                body_head5=candidate[HEAD].tolist(),inner_face51_nonzero=int(np.count_nonzero(candidate[67:,2])),
                body_head5_weight_sum=float(np.square(candidate[HEAD,2]).sum())))
        a=arrays['indoor_reference'];slot=[r['frame'] for r in rows].index('frame_000105');target=d['original_mapped'][slot:slot+1]
        beta=tensor(a['betas'],True);_,_,projection=camera(forward(a,beta=beta),a)
        error=tensor(target[...,:2])-projection;robust=100.**2*error.square()/(100.**2+error.square())
        w=tensor(target[...,2]);jw=torch.ones_like(w);jw[:,[1,9,12]]=0;jw[:,25:]=2
        weighted=(jw*w).square()[...,None]*robust*(1000/rows[slot]['wh'][1])**2
        groups={'head5':HEAD,'other_body_feet':np.setdiff1d(np.arange(25),HEAD),'hands42':np.arange(25,67),'innerface51':np.arange(67,118)}
        grads={};terms={}
        for name,ids in groups.items():
            val=weighted[:,ids].sum();g=torch.autograd.grad(val,beta,retain_graph=True)[0]
            terms[name]=float(val.detach());grads[name]=g.detach().numpy()
        prior=25*beta.square().sum();terms['shape_prior']=float(prior.detach());grads['shape_prior']=torch.autograd.grad(prior,beta)[0].detach().numpy()
        gradient={k:dict(gradient=v.tolist(),norm=float(np.linalg.norm(v)),loss=terms[k]) for k,v in grads.items()}
        gradient['sum']=dict(gradient=sum(grads.values()).tolist(),norm=float(np.linalg.norm(sum(grads.values()))))
        payload={'faces':arrays['indoor_reference']['faces'],'head_vertex_mask':head_vertices}
        payload.update({'rest_'+k:v for k,v in rest.items()})
        for label,vv in variants.items():
            for kind,v in vv.items():payload[label+'__'+kind]=v
        np.savez_compressed(OUT/'diagnostic-geometries.npz',**payload)
        for path,digest in plan['frozen'].items():assert sha(path)==digest,path
        seconds=time.monotonic()-start;assert seconds<120
        save(OUT/'result.json',dict(updated=now(),state='complete_pending_visual_review',seconds=seconds,
            replay_checks=checks,parameters=results,rest_shape_statistics=shape_stats,counterfactual_effects=effects,
            head5_records=head_records,reference105_beta_data_gradient=gradient,head_region='Vertices with maximum LBS weight at original SMPL-X head joint15, not an anatomical face landmark subset.',
            original_files_unchanged=True,optimizer_steps=0,new_fit_calls=0,actual_visual_review=False,quality_accepted=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None))
        print('SHAPE_HEAD5_AUDIT_COMPLETE',seconds,shape_stats,flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc)))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    (prepare if args.mode=='prepare' else run)()
