from pathlib import Path
import json,hashlib,datetime
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118/all-persons-20261027')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();valid=read(R/'portrait-export-validation.json');idx=read(R/'portrait-gaussian-001000-review/index.json')
summary='1000步实际审阅失败后停止，并未完成18k。六相机中眼鼻口错位、帽沿扩散、裙摆与红包拉丝/色雾明显；36个连续视角显示背部空洞，五官不能稳定成一个表面。点云中心仍主要贴近裸身模板，宽裙及包靠较大的高斯覆盖，不能视为衣物表面重建。原生回读与本机OpenFile双模式通过，质量未通过。'
quality=dict(updated=now,subject='portrait',source_sha256=valid['sha256'],snapshot_sha256=idx['snapshot_sha256'],
    actual_same_camera_review=True,reviewed_images=[x['file'] for x in idx['images']],continuous_novel_views_reviewed=36,
    novel_views=list(range(36)),same_crop_all_columns=True,unseen_view_ground_truth_available=False,
    actual_steps=1000,planned_max_steps=18000,stopped_early=True,quality_accepted=False,physical_calibration_accepted=False,
    summary=summary,model_cause_evidence='Training uses triangle-local SMPL-X centers with local-position and scale regularizers. Actual points view remains body-shaped while Gaussian view spreads over skirt/bag. This supports a representation/initialization limitation but is not a single-variable causal proof.',
    remaining_competing_causes=['Unconfirmed intrinsics provenance/crop for online4','Observed head/expression and garment changes across frames','Feature correspondence ambiguity','Mesh-bound body initialization versus large garments and accessories'],
    next_training_started=False)
save(R/'portrait-quality-review.json',quality)
screens=[R/'portrait-final-review'/('viewer-openfile-'+m+'.png') for m in ['gaussian','points']]
assert all(p.exists() for p in screens)
save(R/'portrait-viewer-ui-review.json',dict(updated=now,file_opened_through_UI=True,source_sha256=valid['sha256'],
    observed_count=200000,observed_sh_degree=3,modes=['Gaussian','points'],local_viewer='http://127.0.0.1:8870/',
    displayed_filename='shuolee_118_portrait_s01000.ply',actual_visual_review=True,
    export_render_matches_native=True,quality_accepted=False,
    local_standalone_verified=False,pc00_verified=False,screenshots=[dict(path=str(p),sha256=sha(p)) for p in screens]))
print('Portrait stopped1000 actual final and UI review recorded.')
