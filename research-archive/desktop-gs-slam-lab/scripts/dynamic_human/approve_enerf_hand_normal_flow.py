from pathlib import Path
import json,hashlib,numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';N=O/'normal-flow'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=read(N/'preparation.json');checks=[]
for row in r['records']:
    p=N/row['filename'];assert sha(p)==row['sha256'];z=np.load(p);n=len(z['xy']);assert n==row['accepted'];assert np.all(z['fb']<.5) and np.all(z['color_delta']<=18) and np.all(z['normal_dis_agreement']<=1) and np.all(z['source_gradient']>=5) and np.all(z['target_gradient']>=3) and np.all(z['normal_cosine']>=.5)
    assert np.allclose(np.linalg.norm(z['normal'],axis=1),1,atol=1e-6);assert np.allclose((z['flow']*z['normal']).sum(1),z['normal_target'],atol=1e-6);assert np.allclose(z['xy']+z['flow'],z['destination'],atol=1e-4)
    assert row['source_frame']%2==0 and row['target_frame']%2==0 and row['camera'] not in ['00','06','12','14','16'];checks.append(dict(filename=row['filename'],sha256=sha(p),count=n))
cams=sorted({r['camera'] for r in r['records']});assert len(cams)==13 and sum(x['count'] for x in checks)==1527
save(N/'visual-review.json',dict(status='all13sheets52source_target_native_pairs_actually_reviewed',training_approved=True,preparation_sha256=sha(N/'preparation.json'),files=[dict(path=f'camera{c}-review.png',sha256=sha(N/f'camera{c}-review.png')) for c in cams],
    observed='Acceptedpoints are onvisiblebackofhand/wristandfewfingercreases. Excluded14/16toycontamination. Frontalopeningpalm04has only2points; wrist/handbackdominates. This doesnotconstitutecompletefingercoverage.',
    scope='Authorize bounded paired normalcomponentloss experiment only, not groundtruth handmotion or clarity acceptance.',goal_complete=False))
save(N/'input-audit.json',dict(status='52candidate_arrays_and_normal_component_gates_verified',records=checks,total=1527,goal_complete=False))
save(O/'training-protocol.json',dict(status='fixed_before_training_and_model_motion_errors',parent='079/local_bspline/resume.pt',branches=['photo_control','normal_flow'],additional_steps_each=1500,
    optimized='Existing residualgraphcoefficients onlight-personnodes supporting visiblehand. Sameactive set bothbranches; Gaussian appearance,canonicalgeometry,opacity,covariance,density,bodymotion,camerasandclocks frozen.',
    sampling='Same105trainingimage randomschedule,seed8301; samebackgroundseed8302. One normalflowpair perupdate,sharedschedule seed8303. FreshAdam lr0.0003,decayto0.0001.',
    loss='OriginalRGB+SSIM+alpha+graphregularizers. Candidate adds0.05meanSmoothL1(normal_error/2px); controlcoefficient0.',
    flow='Sourcevisibility weights frozenfromparent: normalizedsourcealpha>=0.95,lightpersoncontribution>=0.9. RasterweightedprojectedGaussiancenterdisplacement; approximatesmaterialmotion withinfinitefootprints.',
    topology='Active nodes fromparentGaussians projecting intoreviewed sourcehandmasks and within0.03scene_scale ofparentexpecteddepth,lightpersononly; others exactfrozen.',
    evaluation='Fixedfinal1500additionalupdates;120originalnativecontexts=90knownodd+18otherodd+12traindiagnostic,unchangedmattesandfixedhand/shoewindows. Noheldouttuning.',goal_complete=False))
print('APPROVED_BOUNDED_NORMAL_FLOW_CONTROL',1527)
