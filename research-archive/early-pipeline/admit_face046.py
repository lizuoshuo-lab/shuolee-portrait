from pathlib import Path
import numpy as np,json,hashlib,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/smplx-native-face-observations-046'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'admission.json').exists()
d=np.load(R/'face-observations.npz');contract=read(R/'mapping-contract.json')
assert contract['native_barycentric_joint_error_model_m']<1e-6 and d['valid'].sum()==55
mask=np.zeros((59,68),bool)
for t in range(59):
    if not d['valid'][t]:continue
    indices=list(range(29,36))+list(range(48,60)) if 12<=t<=34 else [30,33,51,57]
    mask[t,indices]=True
diag=np.array([11,23,33,45,58]);train=mask.copy();train[diag]=False
np.savez_compressed(R/'reviewed-support.npz',selected_mask=mask,train_mask=train,diagnostic_frames=diag)
record=dict(created=datetime.datetime.now().astimezone().isoformat(),state='limited_nose_mouth_pair_preregistered',
    limited_experiment_admitted=True,full_landmark_set_accepted=False,quality_accepted=False,
    observations_sha256=sha(R/'face-observations.npz'),support_sha256=sha(R/'reviewed-support.npz'),
    mapping_contract_sha256=sha(R/'mapping-contract.json'),
    reviewed_raw_images=[f'review/all-faces-{i:02d}.jpg' for i in [0,12,24,36,48]],
    reviewed_baseline_detail=['review/baseline-000023.jpg','review/baseline-000045.jpg'],
    all59_raw_head_images_reviewed=True,invalid_frames=np.flatnonzero(~d['valid']).tolist(),
    support_rule='Near-frontal12..34: FAN29..35 and48..59; other valid frames:30,33,51,57; no contours, brows or eyes',
    rationale='Brow/eye observations often coincide with hair or glasses; near-frontal nose and mouth have direct image support. Profiles retain only nose/lip midline estimates. This conservative manual rule is specific to this video, not a generic visibility model.',
    score_is_correctness_probability=False,hand_authored_landmark_coordinates=False,
    train_frames=int(train.any(-1).sum()),train_points=int(train.sum()),diagnostic_frames=diag.tolist(),diagnostic_points=int(mask[diag].sum()),
    diagnostic_scope='Excluded only from face loss. The same images still participate in body, foot, hand and prior constraints. Not independent GT or unseen-view testing.',
    pixel_contract='FAN directly receives original full image and translated SFD box, no outer crop resize inverse. Official API integer output retained. Direct vs translated integer-crop replay is0px on4 selected frames,3px on frame11; cannot claim perfect crop invariance or subpixel accuracy.',
    preregistered_pair=dict(arms=['control','face_surface'],steps_each=1200,original_NLF_start=True,zero_shape=True,
        new_Adam=True,old_personalized_initialization=False,face_weight=.2,face_scale_px=8.,ramp=[300,500],
        per_frame_normalization=True,base_support='045 visible ears only, COCO wrists retained',
        all_other_constraints_equal=True,acceptance='Same-source meshes and fixed cameras must show real head/neck improvement without body degradation; face residual alone insufficient'),
    no_new_download_bytes=0,body_updates=0,gaussian_updates=0)
(R/'admission.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print({k:record[k] for k in ['train_frames','train_points','diagnostic_points','invalid_frames']})
