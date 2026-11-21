"""Read-only evidence from the temporal candidate's explicit review pause."""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now
from review_full_sequence123 import mesh_renderer,joined,crop_box
from smplify_score_contract123 import relative_confidence
from official_smplifyx_entry123 import COCO

def main(final=False, hand_control=False, continuation=False, midpoint=False, core_subject=None, snapshot=False, core_variant=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    cv2.setNumThreads(1)
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    run=root/(core_subject+'-temporal' if core_subject else ('hand-observation-coherence' if hand_control else ('winterhat-continuation' if continuation else 'winterhat-temporal')))
    if core_variant:
        assert core_subject=='sofa'
        run=root/'sofa-temporal-warmstart-v2'
    fit_folder='control-fit' if hand_control else 'official-fit'
    if snapshot:
        assert core_subject and not (final or hand_control or continuation or midpoint)
        progress=read(run/'temporal-progress.json')
        assert progress['state']=='fitting', 'Use final review after completion.'
        # Review every newly completed frame, not a selected best subset. Read
        # only committed frame receipts; the running fitter is never changed.
        earlier=set(read(run/'review-request.json')['frames'])
        done=[r['frame'] for r in progress['records']]
        request=dict(frames=[n for n in done if n not in earlier],last_frame=None)
        assert request['frames']
    elif hand_control:
        progress=read(run/'control-progress.json')
        assert progress['state']=='completed' and progress['frames']==12, 'Control must finish before review.'
        request=dict(frames=read(run/'control-plan.json')['preregistered_control_frames'],last_frame=None)
    elif final:
        progress=read(run/'temporal-progress.json')
        assert progress['state']=='completed', 'Do not review an incomplete final fit.'
        completed=read(run/'official-fit/fit/official-fit-result.json')
        assert completed['state']=='complete', 'Do not review an incomplete final fit.'
        request=dict(frames=read(run/'plan.json')['ordered_fit_frames'],last_frame=None)
        assert progress['frames']==completed['frames']==len(request['frames'])==read(run/'plan.json')['source_roster']
    else:
        request=read(run/('review-request-064.json' if midpoint else 'review-request.json'))
    out=run/(f'progress-review-{len(done):03d}' if snapshot else ('final-review' if (final or hand_control) else ('midpoint-review' if midpoint else 'checkpoint-review')));out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(), 'Inspect the existing review instead of recomputing.'
    src=root/((core_subject or 'winterhat')+'-input');rows=read(src/'data/inputs.json')['records'];byid={r['frame']:r for r in rows}
    with np.load(src/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    confidence,_=relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    targets=np.zeros((len(rows),118,3),np.float32)
    dst=np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    targets[:,dst,:2]=obs['keypoints'][:,source];targets[:,dst,2]=confidence[:,source]
    baseline_path=root/'winterhat-temporal/official-fit/fit/fitted.npz' if (hand_control or continuation) else src/'official-fit/fit/fitted.npz'
    baseline_parameters={}
    if core_subject:
        from smplify_observation_validity124 import mask_outside_raster
        for i,row in enumerate(rows): targets[i:i+1]=mask_outside_raster(targets[i:i+1],row['wh'])[0]
        baseline=dict(frame_ids=np.array([],dtype=str))
        if core_variant:
            old=[]
            for name in request['frames']:
                path=root/'sofa-temporal/official-fit/frames'/name/'winner-camera-space.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as z:
                        baseline_parameters[name]=dict(z)
                        old.append((name,z['vertices_camera'][0],z['joints_projected_native'][0]))
            if old:baseline=dict(frame_ids=np.array([x[0] for x in old]),vertices_camera=np.stack([x[1] for x in old]),joints_projected_native=np.stack([x[2] for x in old]))
    else:
        with np.load(baseline_path,allow_pickle=False) as z:baseline=dict(z)
    b_ids=baseline['frame_ids'].astype(str).tolist()
    render,_=mesh_renderer();records=[];pictures=[];fits={}
    record=Path('/mnt/d/00/记录')/('006' if core_subject=='sofa' else '001')
    for name in sorted(request['frames']):
        file=run/fit_folder/'frames'/name/'winner-camera-space.npz'
        if name==request['last_frame'] and not file.exists():file=run/('review-pause-pose-064.npz' if midpoint else 'review-pause-pose.npz')
        with np.load(file,allow_pickle=False) as z:f=dict(z)
        fits[name]=f
        row=byid[name];i=[r['frame'] for r in rows].index(name)
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        box=crop_box(mask.getbbox(),raw.size,30);K=np.array(row['K_integer_centers']);v=f['vertices_camera'][0]
        images={}
        for region,bb in [('body',box),('face',crop_box(row['face_box_xyxy'],raw.size,35)),('feet',crop_box([box[0],box[1]+.72*(box[3]-box[1]),box[2],box[3]],raw.size))]:
            parts=[raw.crop(bb)]
            if name in b_ids:parts.append(render(raw.crop(bb),np.array(bb),dict(vertices_camera=baseline['vertices_camera'][b_ids.index(name):b_ids.index(name)+1],faces=f['faces'],K_integer_centers=K)))
            parts.append(render(raw.crop(bb),np.array(bb),dict(vertices_camera=v[None],faces=f['faces'],K_integer_centers=K)))
            prefix='手部可靠度124' if hand_control else ('相邻初值124' if continuation else '连续帧124')
            if core_subject:prefix='原版连续帧124'
            previous='时间约束_' if (hand_control or continuation) else '独立拟合_'
            current='手部观测对照' if hand_control else ('相邻初值时间约束' if continuation else '时间约束')
            if core_subject:current='同运行相邻初值与时间约束'
            if core_variant:prefix,previous,current='原版精修连续帧124','五阶段时间项_','原版精修时间项'
            path=record/f'smpl-x_{prefix}_{name}_{region}_原图_{previous if name in b_ids else ""}{current}.png'
            panel=joined(parts)
            if path.exists():
                assert np.array_equal(np.asarray(Image.open(path).convert('RGB')),np.asarray(panel)), 'Preserve the earlier visual checkpoint: '+str(path)
            else:
                panel.save(path)
            images[region]=str(path)
        # The original photograph beneath a translucent mesh can conceal a
        # misplaced limb. Also show the same native projection on black.
        solid=render(Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'black'),np.array(box),
                     dict(vertices_camera=v[None],faces=f['faces'],K_integer_centers=K))
        solid_path=record/f'smpl-x_{prefix}_{name}_body_原图_同相机实体投影.png'
        solid_panel=joined([raw.crop(box),solid])
        if solid_path.exists():
            assert np.array_equal(np.asarray(Image.open(solid_path).convert('RGB')),np.asarray(solid_panel))
        else:
            solid_panel.save(solid_path)
        images['body_solid']=str(solid_path)
        side=[];center=(v.min(0)+v.max(0))/2;extent=float(np.max(np.ptp(v,axis=0)))
        for deg in (90,-90):
            a=np.deg2rad(deg);rot=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            vv=(v-center)@rot.T;vv[:,2]+=3*extent
            side.append(render(Image.new('RGB',(340,480),'black'),np.array([0,0,340,480]),dict(vertices_camera=vv[None],faces=f['faces'],K_integer_centers=np.array([[1150,0,170],[0,1150,240],[0,0,1]]))))
        n=Image.open(images['body']);n.thumbnail((450,480));solid.thumbnail((340,480));panel=Image.new('RGB',(1470,480),'black');panel.paste(n,(0,0));panel.paste(solid,(450,0));panel.paste(side[0],(790,0));panel.paste(side[1],(1130,0));pictures.append(panel)
        err=np.linalg.norm(f['joints_projected_native'][0]-targets[i,:,:2],axis=-1)
        j=f['joints_camera'][0]
        result=dict(frame=name,images=images,right_arm_error_px=err[[2,3,4]].tolist(),left_arm_error_px=err[[5,6,7]].tolist(),
            right_arm_confidence=targets[i,[2,3,4],2].tolist(),
            right_wrist_minus_shoulder_model_m=(j[4]-j[2]).tolist(),
            all_118_error_px=err.tolist(),face_inner51_error_px=err[67:118].tolist(),has_old_same_frame=name in b_ids,
            observation_confidence=targets[i,:,2].tolist(),disabled_original_joint_indices=[1,9,12],
            errors_are_against_detector_observations_not_3d_truth=True)
        if name in b_ids:
            k=b_ids.index(name);old=np.linalg.norm(baseline['joints_projected_native'][k]-targets[i,:,:2],axis=-1)
            result.update(baseline_right_arm_error_px=old[[2,3,4]].tolist(),baseline_all_118_error_px=old.tolist(),
                baseline_vertex_change_max_model_m=float(np.linalg.norm(v-baseline['vertices_camera'][k],axis=-1).max()))
        if name in baseline_parameters:
            previous=baseline_parameters[name]
            contract={key:np.array_equal(f[key],previous[key]) for key in ('K_integer_centers','camera_translation','betas','faces')}
            assert all(contract.values()),(name,contract)
            receipt=read(run/fit_folder/'frames'/name/'result.json') if file.name!='review-pause-pose.npz' else None
            result['controlled_input_parity']=contract
            result['original_stage_indices']=receipt['original_stage_indices'] if receipt else None
        records.append(result)
    pages=[]
    for begin in range(0,len(pictures),4):
        page=Image.new('RGB',(1470,480*min(4,len(pictures)-begin)),'black')
        for j,pic in enumerate(pictures[begin:begin+4]):page.paste(pic,(0,j*480))
        path=out/f'page-{begin//4+1:02d}.png';page.save(path);pages.append(str(path))
    comparisons=[Image.open(r['images']['body']).convert('RGB') for r in records if r['has_old_same_frame']]
    if not comparisons:comparisons=[Image.open(r['images']['body']).convert('RGB') for r in records[:4]]
    for p in comparisons:p.thumbnail((900,700))
    summary=Image.new('RGB',(max(x.width for x in comparisons),sum(x.height for x in comparisons)),'white');y=0
    for p in comparisons:summary.paste(p,(0,y));y+=p.height
    summary.save(out/'selected-comparison.png')
    transitions=[];ids=sorted(fits)
    for a,b in ([] if hand_control else zip(ids[:-1],ids[1:])):
        # A progress snapshot can exclude an earlier reviewed interval. Such a
        # gap is not one adjacent-frame transition and must not be reported as it.
        if int(b[-6:])-int(a[-6:])!=1:continue
        fa,fb=fits[a],fits[b]
        ra=np.stack([cv2.Rodrigues(x)[0] for x in fa['body_pose_axis_angle'].reshape(-1,3)])
        rb=np.stack([cv2.Rodrigues(x)[0] for x in fb['body_pose_axis_angle'].reshape(-1,3)])
        angle=np.degrees(np.arccos(np.clip((np.trace(ra.transpose(0,2,1)@rb,axis1=1,axis2=2)-1)/2,-1,1)))
        ja,jb=fa['joints_camera'][0],fb['joints_camera'][0]
        transitions.append(dict(previous=a,current=b,max_local_body_angle_deg=float(angle.max()),
            max_local_body_joint_index=int(np.argmax(angle)),
            dt_seconds=float(byid[b]['time_seconds']-byid[a]['time_seconds']),
            right_relative_wrist_motion_model_m=float(np.linalg.norm((jb[4]-jb[2])-(ja[4]-ja[2])))))
    save(out/'result.json',dict(updated=now(),subject=core_subject or 'winterhat',frames=len(records),records=records,transitions=transitions,pages=pages,
        actual_visual_review=False,synthetic_views_not_truth=True,solid_front_same_native_projection=True,full_candidate_quality_accepted=False,continuation_candidate=continuation,
        mode='completed_frame_snapshot' if snapshot else ('isolated_hand_input_control' if hand_control else ('full_sequence' if final else 'checkpoint')),core_stage_entry_control=core_variant,gaussian_admitted=False,
        raw_disabled_joint_errors_not_used_as_quality_aggregate=True))
    print('TEMPORAL_CHECKPOINT_REVIEW_READY',len(records),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group();g.add_argument('--final',action='store_true');g.add_argument('--hand-control',action='store_true');g.add_argument('--snapshot',action='store_true');parser.add_argument('--continuation',action='store_true');parser.add_argument('--midpoint',action='store_true');parser.add_argument('--core-subject',choices=['sofa']);parser.add_argument('--core-variant',action='store_true');args=parser.parse_args()
    assert not args.midpoint or (args.continuation and not args.final and not args.hand_control)
    assert not (args.continuation and args.hand_control)
    assert not args.core_subject or not (args.continuation or args.hand_control or args.midpoint)
    main(final=args.final,hand_control=args.hand_control,continuation=args.continuation,midpoint=args.midpoint,core_subject=args.core_subject,snapshot=args.snapshot,core_variant=args.core_variant)
