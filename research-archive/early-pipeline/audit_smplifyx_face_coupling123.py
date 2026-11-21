"""Bounded CPU attribution through the existing SMPL-X forward model.

Saved parameter-block counterfactuals are diagnostic only. No optimizer,
candidate selection, fitted-model export or training-file modification.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import replay_demo_limb_losses123 as base
import official_smplifyx_entry123 as entry
from review_smplifyx_orientations123 import metrics, load
from review_full_sequence123 import mesh_renderer, joined
from audit_smplifyx_data_terms123 import image_terms

BODY=base.ROOT.parent
RUN=BODY/'body-weight-guarded-sequence'
PLAN=RUN/'review/face-coupling-plan.json'
OUT=RUN/'face-coupling-audit'
read,save,sha=base.read,base.save,base.sha
VARIANTS=('restore_root','restore_body','restore_spine','restore_head_neck','restore_face','restore_hands')
FACE=('expression','jaw_pose','leye_pose','reye_pose')
HAND=('left_hand_pose','right_hand_pose')


def now():return datetime.now(timezone.utc).isoformat()


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not PLAN.exists() and not OUT.exists(), 'Completed or active diagnostics must be preserved.'
    selection=read(RUN/'selection-audit.json')
    closeout=read(RUN/'review/closeout.json')
    assert selection['checked_frames']==59 and closeout['state']=='rejected_face_protection'
    frames=[row['frame'] for row in selection['rows']]
    selected=[row['frame'] for row in selection['rows'] if row['selected_index']==0]
    assert frames==[f'frame_{i:06d}' for i in range(1,60)] and len(selected)==13
    import smplx.body_models,smplx.lbs,smplx.utils,smplx.vertex_joint_selector,smplx.joint_names
    assert smplx.joint_names.JOINT_NAMES[12]=='neck' and smplx.joint_names.JOINT_NAMES[15]=='head'
    sources=[Path(__file__),Path(base.__file__),Path(entry.__file__),
        Path(metrics.__code__.co_filename),Path(image_terms.__code__.co_filename),mesh_renderer()[1]]
    sources += [Path(mod.__file__) for mod in (smplx.body_models,smplx.lbs,smplx.utils,smplx.vertex_joint_selector,smplx.joint_names)]
    sources += [base.OFFICIAL/relative for relative in entry.PINNED]
    inputs=[RUN/'selection-audit.json',RUN/'review/closeout.json',RUN/'review/criteria-audit.json',
        RUN/'review/visual-review.json',base.ROOT/'fit/fitted.npz',base.ROOT/'data/training-inputs.json',
        base.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    for frame in frames:
        inputs += [root/'frames'/frame/'winner-camera-space.npz' for root in (base.ROOT,RUN)]
    failures=sorted({row['frame'] for row in closeout['protection_failures']})
    records={row['frame']:row for row in read(base.ROOT/'data/training-inputs.json')['records']}
    for frame in failures:
        inputs += [Path(records[frame]['image']),base.ROOT/'review'/(frame+'_saved-review.json')]
    save(PLAN,dict(updated=now(),state='prepared',frames=frames,selected_frames=selected,
        visualization_frames=failures,visualization_rule='All and only failures of the already fixed per-frame face protection; not a fitting or training subset.',
        hypothesis='Attribute changed face projection to root, body-chain or local face parameters while preserving fixed camera/beta/observations. A lower original total need not preserve per-region median.',
        variants=list(VARIANTS),body_pose_rows=dict(spine=[2,5,8],head_neck=[11,14]),
        joint_names=dict(spine=['spine1','spine2','spine3'],head_neck=['neck','head']),
        face_fields=list(FACE),hand_fields=list(HAND),
        forward_budget=2*len(frames)+len(selected)*len(VARIANTS),cpu_seconds_max=180,
        parity_limits=dict(vertices_max_m=2e-5,projection_max_px=.01),
        criteria=['Reconstruct all59 baseline and selected saved meshes and projections before drawing attribution conclusions.',
            'For all13 new selected fits, restore each of six predeclared blocks independently from the same-frame original fit; report every result.',
            'Keep fixed K, beta, camera translation, observation mapping and confidence exactly unchanged.',
            'Report original weighted robust face term and unweighted native-pixel median separately.',
            'Parameter restorations are conditional interventions, not a fitted candidate or unique physical cause.',
            'No new optimizer, camera estimate, image warp, threshold sweep, frame splice or Gaussian/VHAP start.'],
        source_hashes={str(p.resolve()):sha(p) for p in sources},input_hashes={str(p):sha(p) for p in inputs},
        no_gpu=True,optimizer_updates=0,quality_accepted=False,goal_complete=False))
    print('PREPARED_FACE_ATTRIBUTION',len(frames),len(selected),flush=True)


def verify():
    p=read(PLAN)
    for path,digest in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(path)==digest,path
    return p


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=verify();assert not OUT.exists(), 'Never rerun a completed or failed diagnostic in place.'
    OUT.mkdir();save(OUT/'progress.json',dict(state='cpu_saved_parameter_attribution',pid=os.getpid(),frames=0))
    started=time.monotonic()
    def deadline(signum,frame):raise TimeoutError('Preregistered CPU diagnostic budget exceeded.')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(p['cpu_seconds_max'])
    try:
        import torch,smplx,smplx.joint_names
        from PIL import Image
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert not torch.cuda.is_initialized()
        entry.adapted_entry_source(base.OFFICIAL)
        utils=entry.module(base.OFFICIAL/'smplifyx/utils.py','_face_attribution123_utils')
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        model=smplx.SMPLX(str(base.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).cpu().eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        parameter_names=[name for name,_ in model.named_parameters()]
        for name,rows in p['body_pose_rows'].items():
            assert [smplx.joint_names.JOINT_NAMES[i+1] for i in rows]==p['joint_names'][name]
        aggregate=load(base.ROOT/'fit/fitted.npz');ids=list(aggregate['frame_ids'].astype(str))
        records={row['frame']:row for row in read(base.ROOT/'data/training-inputs.json')['records']}
        renderer,_=mesh_renderer();parity=[];rows=[];images=[];forwards=0
        diagnostic_arrays={}
        @torch.no_grad()
        def forward(saved):
            nonlocal forwards
            assert time.monotonic()-started<p['cpu_seconds_max']
            model.reset_params(**{name:saved[name] for name in parameter_names})
            result=model(body_pose=torch.from_numpy(saved['body_pose_axis_angle']),return_verts=True,return_full_pose=True)
            t=torch.from_numpy(saved['camera_translation'])[:,None]
            v=(result.vertices+t).numpy();j=(result.joints+t).numpy()
            K=torch.from_numpy(saved['K_compute_float32']);q=(result.joints+t)@K.T
            uv=(q[...,:2]/q[...,2:]).numpy();forwards+=1
            return dict(vertices_camera=v,joints_camera=j,joints_projected_native=uv,faces=saved['faces'],K_integer_centers=saved['K_integer_centers'])
        for n,frame in enumerate(p['frames']):
            original=load(base.ROOT/'frames'/frame/'winner-camera-space.npz');selected=load(RUN/'frames'/frame/'winner-camera-space.npz')
            for key in ('K_integer_centers','K_compute_float32','camera_translation','betas','faces'):
                assert np.array_equal(original[key],selected[key]),(frame,key)
            target=aggregate['observed_native_keypoints'][ids.index(frame)]
            evaluated={};summary={}
            for label,saved in (('baseline',original),('selected',selected)):
                actual=forward(saved);evaluated[label]=actual
                vd=float(np.max(np.abs(actual['vertices_camera']-saved['vertices_camera'])))
                ud=float(np.max(np.abs(actual['joints_projected_native']-saved['joints_projected_native'])))
                assert vd<p['parity_limits']['vertices_max_m'] and ud<p['parity_limits']['projection_max_px'],(frame,label,vd,ud)
                parity.append(dict(frame=frame,arm=label,vertex_max_m=vd,projection_max_px=ud))
            if frame in p['selected_frames']:
                for variant in p['variants']:
                    changed={k:v.copy() for k,v in selected.items()}
                    if variant=='restore_root':changed['global_orient']=original['global_orient']
                    elif variant=='restore_body':changed['body_pose_axis_angle']=original['body_pose_axis_angle']
                    elif variant in ('restore_spine','restore_head_neck'):
                        block=p['body_pose_rows'][variant.removeprefix('restore_')]
                        changed['body_pose_axis_angle'].reshape(21,3)[block]=original['body_pose_axis_angle'].reshape(21,3)[block]
                    else:
                        for field in (FACE if variant=='restore_face' else HAND):changed[field]=original[field]
                    evaluated[variant]=forward(changed)
            for label,actual in evaluated.items():
                group=metrics(actual,target)
                summary[label]=dict(groups=group['groups'],left_elbow_px=group['left_elbow_px'],right_elbow_px=group['right_elbow_px'],
                    original_data_terms=image_terms(actual['joints_projected_native'],target,records[frame]['wh'][1]),
                    face_errors_px=[x['error_px'] for x in group['per_joint'][67:]],
                    face_projection_delta_from_selected_px=float(np.max(np.linalg.norm(actual['joints_projected_native'][:,67:]-evaluated['selected']['joints_projected_native'][:,67:],axis=-1))))
                diagnostic_arrays[frame+'__'+label]=actual['joints_projected_native']
            rows.append(dict(frame=frame,new_selected=frame in p['selected_frames'],variants=summary,
                saved_parameters_changed={name:float(np.max(np.abs(selected[name]-original[name]))) for name in parameter_names+['body_pose_axis_angle']},
                beta_camera_observations_unchanged=True))
            if frame in p['visualization_frames']:
                raw=Image.open(records[frame]['image']).convert('RGB')
                crop=next(x['crop_integer_xyxy'] for x in read(base.ROOT/'review'/(frame+'_saved-review.json'))['files'] if x['region']=='face')
                labels=['baseline','selected']+list(p['variants'])
                native_crop=raw.crop(crop)
                panels=[native_crop]+[renderer(native_crop,np.array(crop),evaluated[label]) for label in labels]
                path=OUT/(frame+'_face_raw_base_selected_root_body_spine_headneck_face_hands.png')
                joined(panels).save(path)
                destination=Path('/mnt/d/00/记录/demo')/('分析_面部参数耦合_'+path.name)
                assert not destination.exists();destination.write_bytes(path.read_bytes())
                images.append(dict(frame=frame,path=str(path),record_path=str(destination),sha256=sha(path),crop_integer_xyxy=crop,
                    columns=['raw']+labels,no_resampling=True,no_text_added=True,diagnostic_only=True))
            if n%10==0 or n==58:save(OUT/'progress.json',dict(state='cpu_saved_parameter_attribution',pid=os.getpid(),frames=n+1,forwards=forwards))
        assert forwards==p['forward_budget'] and not torch.cuda.is_initialized()
        verify()
        np.savez_compressed(OUT/'diagnostic-projections-only.npz',**diagnostic_arrays)
        result=dict(updated=now(),state='complete_pending_visual_review',frames=59,new_selected_frames=len(p['selected_frames']),
            body_forwards=forwards,elapsed_cpu_seconds=time.monotonic()-started,parity=parity,rows=rows,images=images,
            max_vertex_parity_m=max(x['vertex_max_m'] for x in parity),max_projection_parity_px=max(x['projection_max_px'] for x in parity),
            plan_sha256=sha(PLAN),no_gpu=True,optimizer_updates=0,no_fit_or_candidate_changes=True,
            no_training_admission=True,quality_accepted=False,goal_complete=False,
            limitations=['Counterfactual parameter replacements diagnose conditional numerical effects only.',
                'Original detector residuals are not independent material truth.',
                'No counterfactual is a fitted result or eligible for model export.'])
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(state='complete',pid=None,frames=59,forwards=forwards))
        for row in rows:
            if row['frame'] in p['visualization_frames']:
                print(row['frame'],{k:round(v['groups']['face51']['median'],5) for k,v in row['variants'].items()},flush=True)
        print('COMPLETE_CPU_ATTRIBUTION',forwards,result['elapsed_cpu_seconds'],result['max_projection_parity_px'],flush=True)
    except BaseException as error:
        save(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_fit_modified=True))
        save(OUT/'progress.json',dict(state='failed_preserved',pid=None));raise
    finally:signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
