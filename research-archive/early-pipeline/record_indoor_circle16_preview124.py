"""Record the36 actually inspected portable orbit images; no training changes."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import audit_gaussianavatars_params123 as a

def main():
    root=a.REC/'cross-subject-sh0-124/indoor-circle16-paired18k124'
    out=root/'delivery/step-018000';v=a.read(out/'validation.json')
    pages=[out/f'novel/orbit_{i:02d}-{i+11:02d}.png' for i in (0,12,24)]
    review=dict(updated=datetime.now(timezone.utc).isoformat(),actual_visual_review=True,
        views_reviewed=36,all_source_native_frames_reviewed=16,quality_accepted=False,
        finding_zh='导出点云的36个连续转身视角已逐页检查：与原生检查点同样可见肩臂和衣摆孔洞、侧身倾斜，以及遮挡下身残片；没有将导出数值解释当作质量改善。',
        review_scope='Full orbit montage of actual standard PLY native renders; unknown legs are not evaluated as recovered geometry.',
        ply_sha256=v['sha256'],pages=[str(p) for p in pages],page_sha256={p.name:a.sha(p) for p in pages},
        source_strict_image_parity_passed=False,rigid_strict_image_parity_passed=False,
        numerical_boundary_diagnosed=True,format_field_and_camera_checks_passed=v['format_passed'])
    a.save(out/'visual-review.json',review)
    record=Path('D:/00/记录/008');record.mkdir(exist_ok=True)
    for i,p in zip((0,12,24),pages):shutil.copyfile(a.lp(p),record/f'分析_室内一圈16帧_18k_导出点云_转身{i+1:02d}-{i+12:02d}.png')
    print('INDOOR16_PORTABLE36_ACTUAL_REVIEW_RECORDED')

if __name__=='__main__':main()
