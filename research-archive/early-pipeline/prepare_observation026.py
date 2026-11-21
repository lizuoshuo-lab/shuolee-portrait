"""Prepare a bounded, fresh detector-observation ablation; never edit old fits."""
from pathlib import Path
import json

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
S=LAB/'repos/GVHMR-shuolee/shuolee'
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
R=BASE/'current/smplx-observation-uncertainty-fresh-026'

def replace_once(s,a,b):
    assert s.count(a)==1, (a[:100],s.count(a))
    return s.replace(a,b)

s=(S/'refine_body_motion_fresh.py').read_text(encoding='utf-8')
s=replace_once(s,"smplx-gvhmr-body-motion-fresh-024","smplx-observation-uncertainty-fresh-026")
s=replace_once(s,"choices=['control','body-texture']","choices=['rtmw','mean','uncertainty']")
s=replace_once(s,"    bw,fw=g(data['body_weights']),g(data['foot_weights']);fi=data['foot_indices']", """    bw,fw=g(data['body_weights']),g(data['foot_weights']);fi=data['foot_indices']
    # Only image detector predictions, never personalized fitted state, are inputs.
    raw_vit=torch.load(PRED/'vitpose-keypoints.pt',weights_only=True,map_location='cpu').numpy()
    vit=raw_vit[data['source_slots'],:,:2]
    pair=load(OUT.parent/'audit/detector-pairs.npz')
    assert np.array_equal(vit,pair['vitpose'])
    assert np.array_equal(data['original_keypoints'][:,:17],pair['rtmw'])
    assert np.array_equal(data['body_weights'],pair['weights'])
    assert np.array_equal(data['frame_ids'],pair['frame_ids'])
    rtm=data['original_keypoints'][:,:17]
    mean=(rtm+vit)/2
    difference=rtm-vit
    norm=np.linalg.norm(difference,axis=-1,keepdims=True)
    direction=difference/np.maximum(norm,1e-12)
    attenuation=(1+norm**2/400)**-.5
    identity=np.broadcast_to(np.eye(2),(*difference.shape[:2],2,2)).copy()
    whitener=identity+(attenuation-1)[...,None]*direction[...,None]*direction[...,None,:]
    covariance=identity+difference[...,None]*difference[...,None,:]/400
    assert np.all(np.linalg.eigvalsh(whitener)>0)
    whiten_error=float(np.max(abs(whitener@covariance@whitener-identity)))
    assert whiten_error<1e-5,whiten_error
    target=g(rtm if args.arm=='rtmw' else mean)
    W=g(whitener if args.arm=='uncertainty' else identity)
    np.savez_compressed(OUT/'observation-contract.npz',frame_ids=data['frame_ids'],
        rtmw=rtm,vitpose=vit,mean=mean,target=target.cpu().numpy(),
        whitener=W.cpu().numpy(),weights=data['body_weights'])
    c.save(OUT/'observation-contract.json',dict(arm=args.arm,detector_predictions_not_GT=True,
        mean_is_unweighted_average=True,disagreement_covariance='I + dd^T / 20^2',
        whitening_max_error=whiten_error,uncertainty_is_heuristic_not_calibrated=True,
        visibility_and_confidence_unchanged=True,foot_and_hand_targets_unchanged=True,
        vitpose_sha256=c.sha(PRED/'vitpose-keypoints.pt')))
""")
s=replace_once(s,"        body=wloss(st['uv'],kp[:,:17],bw,20)","""        if args.arm=='rtmw':
            # Preserve the exact old objective operations for the matched control.
            body=wloss(st['uv'],kp[:,:17],bw,20)
        else:
            residual=torch.einsum('fjab,fjb->fja',W,st['uv']-target)/20
            body=(robust(residual).sum(-1)*bw).sum()/bw.sum().clamp_min(1)
""")
s=replace_once(s,"        if args.arm=='body-texture':total+=.3*texture_ramp*body_motion","        total+=.3*texture_ramp*body_motion")
s=replace_once(s,"matched_reference='024 fresh control reproduces023; body-texture changes only added torso/leg relative-motion loss',","matched_reference='rtmw replays024 body-texture; mean changes target; uncertainty changes only residual whitening versus mean',")
s=replace_once(s,"arm=args.arm,body_texture_weight=.3 if args.arm=='body-texture' else 0.,","arm=args.arm,body_texture_weight=.3,observation_protocol='observation-contract.json',")
s=replace_once(s,"    started=time.monotonic();history=[]", """    with torch.no_grad():
        start=forward()
        np.savez_compressed(OUT/'starting-state.npz',vertices_world=start['v'].cpu().numpy(),
            body_pose=start['body'].cpu().numpy(),root_rotation=start['R'].cpu().numpy(),
            translation=start['t'].cpu().numpy(),world_scale=start['s'].cpu().numpy(),
            betas=start['betas'].cpu().numpy())
    started=time.monotonic();history=[]""")
target=S/'refine_observation_uncertainty_fresh.py'
assert not target.exists()
compile(s,str(target),'exec')
target.write_text(s,encoding='utf-8')

r=(S/'review_body_motion_fresh.py').read_text(encoding='utf-8')
r=replace_once(r,"smplx-gvhmr-body-motion-fresh-024","smplx-observation-uncertainty-fresh-026")
r=replace_once(r,"LABELS={'control':'控制：足部约束','body-texture':'新增躯干/腿部纹理约束'}", "LABELS={'rtmw':'原观测','mean':'两检测器均值','uncertainty':'均值与方向不确定性'}")
start=r.index("    for key in initial['control']:")
end=r.index("    for arm in fits:\n",start)
r=r[:start]+"""    import hashlib
    for arm in ['mean','uncertainty']:
        for key in initial['rtmw']:
            assert np.array_equal(initial['rtmw'][key],initial[arm][key]),(arm,key)
    data=initial['rtmw'];E,K=data['w2c'],data['K'];N=len(data['frame_ids'])
    starts={arm:load(ROOT/arm/'starting-state.npz') for arm in LABELS}
    assert all(np.array_equal(starts['rtmw'][k],starts[a][k]) for a in ['mean','uncertainty'] for k in starts['rtmw'])
    cleanup=c.read(c.BASE/'current/smplx-gvhmr-shared-world-fresh-025/cleanup-receipt.json')
    reference=next(x for x in cleanup['files'] if x['path'].replace('\\\\','/').endswith('smplx-gvhmr-body-motion-fresh-024/body-texture/fitted.npz'))
    exact_replay={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==reference['arrays'][k]['sha256'] for k,v in fits['rtmw'].items()}
    assert all(exact_replay.values()),exact_replay
"""+r[end:]
r=replace_once(r,"    bo=load(ROOT/'control/body-texture-observations.npz')\n    other=load(ROOT/'body-texture/body-texture-observations.npz')\n    assert all(np.array_equal(bo[k],other[k]) for k in bo)","""    bo=load(ROOT/'rtmw/body-texture-observations.npz')
    for arm in ['mean','uncertainty']:
        other=load(ROOT/arm/'body-texture-observations.npz')
        assert all(np.array_equal(bo[k],other[k]) for k in bo)""")
r=replace_once(r,"ROOT/'control/shoe-observations.npz'","ROOT/'rtmw/shoe-observations.npz'")
r=replace_once(r,"        stats[arm]=dict(visible_body_error_px=quant(errors[gate]),", """        contract=load(ROOT/arm/'observation-contract.npz')
        detector_errors={name:quant(np.linalg.norm(fit['projected_coco17']-contract[name],axis=-1)[gate]) for name in ['rtmw','vitpose','mean']}
        stats[arm]=dict(visible_body_error_px=quant(errors[gate]),errors_to_detectors_not_GT=detector_errors,""")
# Keep selected evidence and all-view contact sheets; avoid generating duplicate videos.
vstart=r.index("    video_receipts=[]")
vend=r.index("    result=dict(",vstart)
r=r[:vstart]+"""    time_files=[]
    for name in ['front','left90']:
        cam=next(x for x in cams if x['view']==name);W,H=cam['width'],cam['height'];ex=np.array(cam['w2c']);intr=np.array(cam['K'])
        snapshots=[]
        for t in selected:
            panels=[]
            for arm,fit in fits.items():
                camera=fit['vertices_world'][t]@ex[:3,:3].T+ex[:3,3]
                im,_=mesh_image(camera,faces,intr,W,H,ctx);panels.append(art.title(rgb(im),LABELS[arm]+f' | {t:02d}'))
            frame=art.combine(panels);frame.thumbnail((1170,620),Image.Resampling.LANCZOS);snapshots.append(frame)
        sheet=Image.new('RGB',(2340,1860),'white')
        for i,im in enumerate(snapshots):sheet.paste(im,((i%2)*1170,(i//2)*620))
        name=f'fixed-{name}-time-sheet.jpg';sheet.save(OUT/name,quality=95);time_files.append(name)
"""+r[vend:]
r=replace_once(r,"control023_replay_max_abs=baseline_max,","control024_array_hash_equal=exact_replay,")
r=replace_once(r,"new_body_updates=2400","new_body_updates=3600")
r=replace_once(r,"videos=video_receipts,","selected_time_sheets=time_files,")
r=replace_once(r,"print(json.dumps(dict(control023_max_abs=max(baseline_max.values()),statistics=stats),ensure_ascii=False),flush=True)","print(json.dumps(dict(control024_exact_replay=all(exact_replay.values()),statistics=stats),ensure_ascii=False),flush=True)")
target=S/'review_observation_uncertainty_fresh.py'
assert not target.exists()
compile(r,str(target),'exec')
target.write_text(r,encoding='utf-8')

protocol=dict(phase='026',state='prepared',arms=['rtmw','mean','uncertainty'],steps_per_arm=1200,
    from_scratch=True,old_person_fit_loaded=False,
    hypothesis='Detector disagreement may bias 3D articulation; average targets and then attenuate only disagreement directions.',
    independent_comparisons=['rtmw versus mean: target averaging','mean versus uncertainty: direction-specific weighting'],
    unchanged=['raw GVHMR initialization','59 calibrated cameras','body visibility/confidence','shoe and torso feature losses','local hand observations','pose and shape prior','learning rates and schedule'],
    acceptance='Same-camera image overlays and fixed-world side geometry must improve; training loss or detector residual alone cannot establish geometry accuracy.',
    independent_GT=False,new_gaussian_updates=0)
assert not (R/'experiment-protocol.json').exists()
(R/'experiment-protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared fresh three-arm observation ablation and review.')
