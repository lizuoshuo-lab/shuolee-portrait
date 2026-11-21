"""Finalize observed review evidence, file delivery verification and task status."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile

ROOT=Path('D:/00/记录/2026-11-04_MSV2_mask原始_插值2倍_4倍对比')
STATUS=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)

now=datetime.now(timezone.utc).isoformat()
c=read(ROOT/'comparison.json');m=c['means']
audit=read(ROOT/'原视频到训练像素核验.json');assert audit['passed'] and audit['maximum_video_to_png_pixel_error']==0
findings=dict(reviewed_at=now,training_completed=True,quality_accepted=False,
    original_pixels_exact=True,pre_downsampling_detected=False,
    summary_paragraphs=[
        '三版均直接使用原视频原分辨率的人物裁切，没有先缩小再放大。原始版与2倍插值版本次各自从头完成18,000步，分别69,173点、83,033点；4倍沿用已完成的同mask惩罚87,217点结果。MSV2训练模型没有修改。',
        '统一59个原图相机下，原始/2倍/4倍人物L1分别0.03590、0.03516、0.03902；SSIM分别0.91474、0.91048、0.88058。2倍的人物L1略低，原始版SSIM最高。脸部L1分别0.06233、0.06230、0.06343，原始与2倍几乎相同，不能认定2倍显著改善脸部。',
        '面部外侧平均Alpha分别0.05947、0.07453、0.07679；原始版外侧更淡，同时面部内部覆盖也更低（0.93512、0.94406、0.94933）。降低噪点与保留人物内容仍存在权衡。2倍与4倍的部分正面轮廓较平滑，但三版在第47至59帧均有侧脸细节丢失与外侧残影；第25至31帧手持物穿过脸部的外观也未忠实恢复。',
        '已查看全部59帧脸部及整人对照、12个环绕方向，并检查白底、黑底和棋盘底样本。躯干、裤腿和鞋的主体轮廓相近；手持物、举起的手和袖口仍有不完整和漂浮边缘。环绕120至240度的背面存在孔洞与碎片。三版保留为本次对照，没有把放大或训练完成标为整体质量通过。'],
    visual_review=dict(all59_face=True,all59_body=True,orbit_views=12,
        contacts=[f'all_{kind}_{n:02d}.png' for kind in ['face','body'] for n in range(1,6)]+['all_orbit_000.png','all_orbit_180.png'],
        detailed=['views/frame_000028/white_compare.png','views/frame_000059/checker_compare.png',
                  'views/frame_000001/black_compare.png','views/frame_000059/face_compare.png'],
        held_object_labels_unchanged=True),
    model_architecture_unchanged=True,mask_penalty_same=True,no_postprocess_pruning=True,
    conclusion='No single enlargement method wins every metric; residual opacity, hand/object and unseen-back defects remain.',
    future_train_entry_requires_original_video_pixel_audit=True)
save(ROOT/'review-findings.json',findings)
viewer=[];archives=[]
for label in ['original','interpolation2x']:
    folder=ROOT/label;d=read(folder/'delivery.json');bundle=folder/d['bundle']
    assert sha(bundle)==d['sha256']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('point_cloud.ply')).hexdigest()==sha(folder/d['default_ply'])
        assert hashlib.sha256(z.read('cameras.json')).hexdigest()==sha(folder/d['camera_file'])
        cameras=json.loads(z.read('cameras.json'))
        assert len(cameras)==59
        expected=[476,960] if label=='original' else [952,1920]
        assert all([r['width'],r['height']]==expected for r in cameras)
    archives.append(dict(method=label,archive=bundle.name,sha256=d['sha256'],members_verified=True,
        points=d['points'],cameras=59,full_K_preserved=True,upright_parity=True))
    viewer.append(dict(method=label,actual_open_file_entry_tested=True,gaussian_mode_observed=True,
        point_mode_observed=True,background_observed='gray',points_displayed=d['points'],
        SH_degree_displayed=3,paired_cameras_displayed=59,first_camera_upright=True,
        ply_sha256=sha(folder/d['default_ply']),camera_sha256=sha(folder/d['camera_file']),
        quality_accepted=False))
save(ROOT/'viewer-import-verification.json',dict(verified_at=now,url='http://127.0.0.1:8870/?sh=3',
    local_windows_viewer=True,remote_00_opened=False,records=viewer))
save(ROOT/'package-verification.json',dict(passed=True,archives=archives))
s=read(STATUS);v=s['msv2_mask_1x2x4x_comparison'];v.update(updated=now,state='completed_with_quality_limitations',
    review_in_progress=False,report=str(ROOT/'查看mask倍率对照.html'),
    quality_accepted=False,all59_views_reviewed=True,orbit_views_reviewed=12,
    local_viewer_import_verified=True,completed_exports=archives,
    input_audit=str(ROOT/'原视频到训练像素核验.json'),mask4x_sent_and_received=True,
    own_gpu_training_jobs_running=False)
save(STATUS,s)
print('MASK_COMPARISON_FINAL_EVIDENCE_RECORDED')
