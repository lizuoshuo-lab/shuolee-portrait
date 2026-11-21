"""Record the actual16source/36orbit inspection and preserve export failure."""
from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a
from publish_indoor_circle16_124 import atomic

def main():
    root=a.REC/'cross-subject-sh0-124/indoor-circle16-paired18k124'
    snap=root/'gaussian-run/step-018000';orb=root/'native-checkpoint-orbit18k'
    review=a.read(snap/'review/result.json');native=a.read(orb/'result.json')
    diagnostic=a.read(root/'export-failure-diagnostic/result.json')
    repair=a.read(root/'export-scale-encoding124/result.json');progress=a.read(root/'progress.json')
    assert len(review['records'])==16 and len(review['pages'])==16 and native['views']==36
    assert not diagnostic['parity']['passed'] and not repair['parity']['passed']
    assert a.read(root/'process-exit.json')['exit_code']==1
    assert not a.lp(root/'delivery/publication.json').exists()
    now=datetime.now(timezone.utc).isoformat()
    finding='16帧已运行到18000步，71668点。全部16帧原图／当前mesh／高斯及脸、双手已实际检查：同相机头部与上衣轮廓基本连贯，手指偏软，纹身侧手臂、肩袖边缘仍有孔洞，遮挡下身仍未知。共同6帧原像素前景颜色MAE从0.01042升至0.01848，不能宣称整体质量改善。此对照同时更换了输入集合和联合拟合，固定总步数也减少每帧更新，未隔离单一因果。'
    next_action='Keep half-circle8/full-circle16 selection and all immutable results. Do not repeat this candidate or silently change its strict export threshold. Native18k review is complete; no18k PLY has been published. Investigate the quantified float32 scale roundtrip boundary separately from mask holes and fitting quality; only a justified minimal numerical/interface fix may proceed. No new optimizer run until a specific causal hypothesis is registered.'
    atomic(snap/'visual-review.json',dict(updated=now,step=18000,actual_visual_review=True,
        reviewed_frames=16,reviewed_regions=['body','face','left_hand','right_hand'],
        review_pages_sha256={r['path']:r['sha256'] for r in review['pages']},
        geometry=a.read(root/'gaussian-run/geometry-step-018000.json'),
        common_six_foreground_mae=review['common_six_foreground_mae'],finding_zh=finding,
        quality_accepted=False,native_replay_from_checkpoint=True,optimizer_updates_added_by_review=0,
        export_accepted=False,next_action=next_action))
    novel_finding='36个原生检查点连续转身视角已实际审阅。后背大块孔洞较旧6帧图定性减少，但肩臂、衣摆仍呈片状起伏与孔洞，侧面躯干倾斜、下身残片仍存在；相机取景和新拟合同时变化，不能把对照当严格同相机误差验证。遮挡腿脚不作可靠恢复。'
    atomic(orb/'visual-review.json',dict(updated=now,actual_visual_review=True,views=36,
        checkpoint_sha256=native['checkpoint_sha256'],pages=native['pages'],finding_zh=novel_finding,
        directly_rendered_checkpoint=True,portable_PLY_not_used=True,quality_accepted=False,
        independent_novel_ground_truth=False))
    atomic(root/'gaussian-run/result.json',dict(updated=now,state='training_budget_reached_snapshot_export_failed',
        snapshot_step=18000,optimizer_updates=17999,points=71668,views_seen=16,
        seconds_to_last_report=progress['seconds'],timing_includes_review_wait=True,cold_end_to_end_benchmark=False,
        process_exit_code=1,checkpoint_preserved=True,read_only_native_review_completed=True,
        source_replay=str(snap/'result.json'),novel_replay=str(orb/'result.json'),
        strict_native_ply_parity=diagnostic['parity'],quantization_control_passed=False,
        next_action=next_action,quality_accepted=False,export_published=False))
    atomic(root/'final-review.json',dict(updated=now,rule='half_circle8_full_circle16',
        source_finding=finding,novel_finding=novel_finding,export_failure=diagnostic,
        one_parameter_encoding_check=repair,next_action=next_action,quality_accepted=False,
        model_promoted=False,company00_deployed=False))
    print('INDOOR16_NATIVE_REVIEW_COMPLETE_EXPORT_FAILURE_PRESERVED')

if __name__=='__main__':main()
