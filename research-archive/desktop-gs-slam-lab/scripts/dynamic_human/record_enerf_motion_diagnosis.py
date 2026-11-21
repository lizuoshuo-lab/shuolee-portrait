from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
plan=read(R/'unique-patches/visual-review-plan.json');assert len(plan['records'])==19
review=dict(status='all19_unique_candidate_review_panels_actually_viewed',
    files=[dict(path=r['file'],sha256=hashlib.sha256((R/r['file']).read_bytes()).hexdigest()) for r in plan['records']],
    additional_reviewed=['review/largest-model-disagreement.png','review/camera07-frame000744-coverage.png'],
    findings_zh=[
        '八张固定覆盖图中的对应主要来自衣服图案、下摆、头部、鞋边，裸露手指覆盖不足；这些是照片上的标注，不是新的重建图。',
        '十条通过唯一性筛选且落在手部框中的观测全部查看，其中八条位于白衣人物衣服下摆或相邻裤子；剩下两条位于黑衣人物手掌附近，不能认作独立指尖真值。',
        '原最大偏差候选在前臂边界上沿边滑动。局部相关峰值虽为0.9959，但第二个分离峰几乎等高，唯一性检查将其剔除。',
        '唯一性筛选后的最大模型差异位于黑衣人物胸前图案。单一图块对应和投影中心运动仍不足以认证真实表面运动，不能把该16.67像素差异直接解释为几何真值误差。'],
    hand_box_true_fingertip_labels=0,
    decision='Do not add these sparse tracks as finger supervision. Preserve them as texture-biased motion diagnostics; no Gaussian or body model updated.',
    next='Evaluate increased native photometric loss on explicit hand/foot image windows, without treating predicted joints or photo feature correspondences as anatomical truth.',
    goal_complete=False)
save(R/'visual-review.json',review)
summary=read(R/'unique-patches/model-summary.json')
summary.update(status='native_motion_diagnostic_complete_no_finger_supervision_adopted',
    preparation=read(R/'preparation.json')['status'],
    uniqueness=read(R/'unique-patches/summary.json'),actual_review_images=21,
    finding_zh='通过局部匹配筛选不等于获得可靠手指对应；可用点明显偏向衣物纹理。两版模型在这些稀疏候选上的运动差异相近，未证明模糊由单一运动误差导致。',
    model_updates=0,training_labels_accepted=False,goal_complete=False)
save(R/'final-summary.json',summary)
p=L/'manifests/dynamic-human-work.json';m=read(p)
m.update(status='native_motion_correspondence_diagnostic_complete_detail_weight_control_preparing',current_runs=[],
    outdoor_native_motion_tracks='outputs/'+R.name+'/final-summary.json',
    last_recorded_utc=datetime.now(timezone.utc).isoformat(),
    next_actions=['准备手部与鞋部原生照片窗口的颜色损失加权对照；不将稀疏跟踪当作手指真值。','与074保持输入、运动、初始高斯、抽样和15000次更新相同，检查原生手指、鞋和遮挡负例。','通过另外机位和时间片段验证真实清晰度后再接回大场景背景。'],
    goal_turn_classification='Progress:075native91trainingimage patch tracking and two frozen model motion comparisons completed;21actual diagnostic images inspected; rejected ambiguous and non-hand correspondences as finger supervision.')
entry='075native motion5933seeds7277initial5117uniqueobservations5072pairedmodelpoints;21actualpanels;nohandlabelsadopted'
if entry not in m['completed']:m['completed'].append(entry)
save(p,m);shutil.copy2(__file__,R/Path(__file__).name)
print(summary['status'])
