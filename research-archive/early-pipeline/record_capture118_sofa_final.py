from pathlib import Path
import json,hashlib,datetime
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118/all-persons-20261027')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();valid=read(R/'sofa-export-validation.json');idx=read(R/'sofa-gaussian-018000-review/index.json')
summary='18k完成，22.13万高斯。实际六相机审阅：正面整体与衣服基本贴合；右侧鼻唇、眼镜仍有细小扩散，头发外圈和袖口、裤腿、鞋尖有散边。36个连续未知视角显示后脑与整个背部严重缺损/噪声，未拍背面不能靠这轮训练补成可靠表面。格式和本机打开文件通过，重影与完整360度质量未通过。'
quality=dict(updated=now,subject='sofa',source_sha256=valid['sha256'],snapshot_sha256=idx['snapshot_sha256'],
    actual_same_camera_review=True,reviewed_images=[x['file'] for x in idx['images']],continuous_novel_views_reviewed=36,
    novel_views=list(range(36)),same_crop_all_columns=True,unseen_view_ground_truth_available=False,
    quality_accepted=False,physical_calibration_accepted=False,summary=summary,
    cause_limits='Front-facing image fitting does not prove physical calibration. The unobserved back is visibly unsupported; residual visible-view edge spread is not uniquely attributed to camera, correspondence, motion or representational bias.',
    export_control='Same-up diagonal gauge passed unchanged parity limits. Native repeated render max=0. Six export means<6.1e-9 and maxima<.001034. No training change.',
    no_new_training_started=True)
save(R/'sofa-quality-review.json',quality)
save(R/'sofa-gaussian-018000-review/actual-review.json',quality)
screens=[R/'sofa-final-review'/('viewer-openfile-'+m+'.png') for m in ['gaussian','points']]
assert all(p.exists() for p in screens)
save(R/'sofa-viewer-ui-review.json',dict(updated=now,file_opened_through_UI=True,source_sha256=valid['sha256'],
    observed_count=221287,observed_sh_degree=3,modes=['Gaussian','points'],local_viewer='http://127.0.0.1:8870/',
    displayed_filename='shuolee_118_sofa_s18000.ply',actual_visual_review=True,body_local_deformation_from_export_observed=False,
    known_source_quality_failure='Back and legs fragmented in both native render and viewer; no extra loader-induced local limb deformation detected.',
    local_standalone_verified=False,pc00_verified=False,screenshots=[dict(path=str(p),sha256=sha(p)) for p in screens]))
print(summary)
