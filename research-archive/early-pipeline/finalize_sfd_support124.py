"""Record the actual full-roster native-image review, without a fitting pass."""
import json
from datetime import datetime, timezone
from pathlib import Path
from record_observation_support124_helpers import read, save_new, sha, windows

ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
out=ROOT/'sfd-face-presence124/full167'
result=read(out/'result.json')
association=read(out/'association-review/result.json')
assert result['source_hashes_verified'] and result['new_network_forwards']==143
assert len(association['cases'])==16
counts={}
for subject in ('sofa','indoor'):
    rows=[r for r in result['records'] if r['subject']==subject]
    counts[subject]=dict(frames=len(rows),single=sum(len(r['associated_indices'])==1 for r in rows),
        no_associated_box=[r['frame'] for r in rows if not r['associated_indices']],
        ambiguous=[r['frame'] for r in rows if len(r['associated_indices'])>1])
pages=[windows(p['path']) for p in result['pages']]
pages += [Path(p) for p in association['pages']]
enlarged=[Path('D:/00/记录/008')/f'特征点_124_SFD_frame_{n:06d}_原图_整图人脸检测.png' for n in (37,39,64,68)]
stamp=datetime.now(timezone.utc).isoformat()
summary='连续帧约束已在原版SMPLify-X中实际生效，但错误脸手观测仍能拉偏姿态。373帧置信度审计后，使用本机已有SFD完成沙发56＋室内111帧检查；27张室内侧后/后脑图没有关联人脸框，全部与原图核对。另有16帧多框歧义，新增误框包括耳发、手腕、吊灯和背景玩偶。单个人脸框也不能证明51点全部可见：室内39/64等侧后方仍有框。检测结果只可支持局部输入缺失诊断，不能直接作为完整可见性或身体质量准入。'
save_new(out/'visual-review.json',dict(updated=stamp,actual_visual_review=True,
    all167_face_panels_reviewed=True,all16_ambiguous_full_frames_and_box_crops_reviewed=True,
    reviewed_pages=[dict(path=str(p),sha256=sha(p)) for p in pages],
    enlarged_native_cases=[dict(path=str(p),sha256=sha(p)) for p in enlarged],counts=counts,
    findings=[
        'All56 sofa frames contain a visible face. Eight exact-one association abstentions come from extra hair/ear or lamp-reflection boxes, not absent faces.',
        'Indoor40-63 and65-67 have no associated detection. Actual native images show rear/near-rear views without a complete visible eye/nose/mouth landmark set.',
        'Indoor37/39/64/68 retain a narrow side-face/ear box although most facial landmarks are self-occluded. A positive SFD result is not per-landmark visibility.',
        'Indoor75/76 add a hand/wrist detection;75 has wrist confidence0.938 above the real face0.905. Selecting the highest detector score would be wrong.',
        'Indoor81/83/84/85/86/94 include background toy faces inside the tracked-person rectangle. A person bounding box is not a physical instance mask.',
        'All original coordinates, scores, cameras, masks and historical fits remain unchanged. No neural rerun was used for review.'
    ],input_gate_admitted=False,per_landmark_visibility_proven=False,
    quality_accepted=False,gaussian_admitted=False))
save_new(out/'conclusion.json',dict(updated=stamp,summary=summary,counts=counts,
    source_result_sha256=sha(out/'result.json'),visual_review_sha256=sha(out/'visual-review.json'),
    detector_calls_total=167,reused_cached_calls=24,new_roster_calls=143,roster_seconds=result['seconds'],
    fit_calls=0,actual_visual_review=True,quality_accepted=False,gaussian_admitted=False,
    next_action='Prepare only an opt-in negative-evidence adapter: for a hashed frame with zero associated SFD boxes, abstain from its original51 inner-face data targets, while retaining all raw coordinates and all body/hand/foot targets. Positive and ambiguous boxes must never increase confidence or certify visibility. Validate the original118-anchor mapping and unchanged inputs on all167 records before registering any bounded matched fit. Do not deploy a sole SFD visibility gate, select highest scores, sweep thresholds, repeat detectors or restart failed sequences. Sofa face presence does not explain its shoulder failure; indoor lower-body occlusion remains separate.'))
print(json.dumps(dict(summary=summary,counts=counts),ensure_ascii=False))
