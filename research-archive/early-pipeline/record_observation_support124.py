"""Record actual image reviews; no fitting/detection or historical edits."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib

ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def windows(p):
    return Path(p[5].upper()+':/'+p[7:]) if p.startswith('/mnt/') else Path(p)

stamp=datetime.now(timezone.utc).isoformat()
out=ROOT/'observation-confidence-audit';result=read(out/'result.json')
old=ROOT.parent/'smplifyx-reference-control/official-first-cycle/local-face-input-control/decision.json'
images=[Path('D:/00/记录/008')/f'特征点_124_frame_{n:06d}_face_原图_原生二维观测.png' for n in (53,61,105)]
images += [Path('D:/00/记录/006/特征点_124_frame_000026_body_原图_原生二维观测.png'),
           Path('D:/00/记录/006/smpl-x_唯一手腕124_frame000016_body_原图_原输入_唯一手腕目标_原生像素.png')]
cases=[]
for n in (53,61,105):
    r=next(r for r in result['subjects']['indoor']['records'] if r['frame']==f'frame_{n:06d}')
    w=r['stage4_effective_weight_sum_without_common_data_factor']
    cases.append(dict(frame=n,face_conf_median=r['groups']['face']['relative_confidence']['median'],
                      face_weight=w['face'],body_feet_weight=w['body_feet'],ratio=w['face']/w['body_feet']))
save(out/'visual-review.json',dict(updated=stamp,actual_visual_review=True,
    images=[dict(path=str(p),sha256=sha(p)) for p in images],
    findings=['Indoor53/61 are posterior-head crops; eye/nose/mouth detections fall on hair/ear instead of visible facial material.',
              'Indoor105 visibly supports face features. Detector dots still do not provide pixel-accurate ground truth.',
              'Sofa26 has a visible face and both wrists, but two hands overlap. Hidden finger targets cannot be certified from this single image.',
              'Sofa16 shoulders/upper arms remain misaligned in the two saved fits. Shared relative confidence is not an independent true skeletal-joint measurement.'],
    cases=cases,landmark_coordinates_not_hand_corrected=True,quality_accepted=False))
save(out/'conclusion.json',dict(updated=stamp,actual_visual_review=True,
    summary='373帧只读核验确认：RTMW响应峰值经逐点整段归一化后只是相对权重。室内53/61帧原图为后脑，但51个脸部点仍有效；其末阶段权重总量分别为身体脚点的2.69/3.50倍。现有288×384全身检测的原生SimCC步长约1.32—2.23px，但该间距不等于实测定位误差。未发现当前RGB通道或索引转换错误；这也不证明物理相机和对应正确。',
    cases=cases,new_fit_calls=0,new_detector_calls=0,source_hashes_verified=True,
    historical_controls_preserved=dict(fan_decision=str(old),fan_decision_sha256=sha(old),fan_not_adopted=True,
        simcc_conclusion=str(ROOT.parent/'simcc-decoder-conclusion.json'),repeat_old_decoder_or_FAN_control=False),
    limits=['Sofa has a visible face, so missing-face support alone cannot explain its shoulder drift.',
            'Original OpenPose detector confidence is not assumed a calibrated probability either.',
            'Response norms/weight sums do not directly prescribe new weights or prove unique physical cause.'],
    next_action='Use the already cached SFD detector, unchanged defaults and native full images, to test independent face-presence support. Do not re-run older FAN/continuous-SimCC controls or adjust relative-score thresholds. Preserve all original observations and require actual review before integration.',
    quality_accepted=False,gaussian_admitted=False))
sfd=ROOT/'sfd-face-presence124';d=read(sfd/'result.json')
counts={}
for subject in ('sofa','indoor'):
    rows=[r for r in d['records'] if r['subject']==subject]
    counts[subject]=dict(frames=len(rows),single=sum(len(r['associated_indices'])==1 for r in rows),
        missing=sum(len(r['associated_indices'])==0 for r in rows),multiple=sum(len(r['associated_indices'])>1 for r in rows))
save(sfd/'visual-review.json',dict(updated=stamp,actual_visual_review=True,
    all24_faces_reviewed=True,pages=[dict(path=str(windows(p)),sha256=sha(windows(p))) for p in d['pages']],
    enlarged_native_cases=['sofa007','sofa022','indoor071'],counts=counts,
    findings=['All12 sofa frames contain an actual visible face detection, including Haar-missed15/53 and misplaced50.',
              'Sofa7 has an extra ear box and sofa22 an extra hair box. The preregistered exact-one association correctly abstains; these frames were not silently chosen by score.',
              'Indoor43/53/61 have no associated face box; native images show posterior or near-posterior head with no complete visible eyes/nose/mouth set.',
              'All9 remaining indoor frames have one plausible associated face box, including partial profiles31/71/72 previously missed by Haar.',
              'Small outside-person detections remain recorded; presence does not certify hidden-eye/finger landmarks or correct body geometry.'],
    allow_same_detector_roster_expansion=True,
    expansion_only='Same detector/threshold/association on remaining143 sofa+indoor frames; reuse the24 hashed cached results, no repeat forwards. Review all167 before an input-only fit. No face coordinates, raw confidence, masks, cameras or historical fits change.',
    gate_admitted=False,quality_accepted=False,gaussian_admitted=False))
print(json.dumps(dict(confidence_cases=cases,sfd_counts=counts),ensure_ascii=False))
