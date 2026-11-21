"""Archive the completed mask-only experiment; preserve all trained parameters."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import zipfile

ROOT=Path('D:/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比/mask_penalty4x')
RUN=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-demo-boundary-mask-20261104')
BASE=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now(timezone.utc).isoformat()
c=read(ROOT/'comparison.json');old=c['means']['before'];new=c['means']['mask_penalty']
face_reduction=1-new['face_exterior_alpha_mean']/old['face_exterior_alpha_mean']
inside_drop=old['inside_alpha_mean']-new['inside_alpha_mean']
face_error_increase=new['face_foreground_l1']/old['face_foreground_l1']-1
assert c['models_unchanged'] and c['frames']==59 and c['novel_views']==12
assert read(RUN/'execution.json')['state']=='training_completed'
summary=[
    '训练模型保持原样：MSV2模型定义、优化器、增密与简化时序、59帧输入、相机和原始mask一致。只有mask损失增加一项轮廓外单独归一化的负样本BCE；从未训练的相同初始点重新训练18,000步，没有对已有高斯做裁剪或修补。',
    f'原来的mask确实参与反向传播，并未失效。新增惩罚后，59帧面部轮廓外平均Alpha由{old["face_exterior_alpha_mean"]:.5f}降至{new["face_exterior_alpha_mean"]:.5f}，降低{face_reduction:.1%}；整人轮廓外平均Alpha由{old["outside_alpha_mean"]:.5f}降至{new["outside_alpha_mean"]:.5f}。',
    f'代价也明确：人物内部平均Alpha由{old["inside_alpha_mean"]:.5f}降至{new["inside_alpha_mean"]:.5f}，下降{inside_drop*100:.2f}个百分点；脸部内部Alpha由{old["face_inside_alpha_mean"]:.5f}降至{new["face_inside_alpha_mean"]:.5f}。脸部L1误差增加{face_error_increase:.1%}，人物SSIM由{old["foreground_ssim"]:.4f}降至{new["foreground_ssim"]:.4f}。',
    '已审阅全部59帧脸部与整人、12个环绕视角，并检查白底、黑底与棋盘底代表帧。轮廓外散点减少，但侧脸细丝仍在，47至59帧五官仍模糊；28帧手持设备的有效内容明显变淡。衣服、裤子和鞋大体保留，背面缺口仍在。',
    '本次实验未通过人物覆盖保护条件，不作为最终修复采用，原有训练配置和结果继续保留。结论是：需要mask惩罚，但单独增加外侧惩罚还不够；剩余误差需区分mask/相机跨帧一致性和人物动作造成的监督冲突。本轮不修改训练模型，也未自动启动新权重试验。',
]
review=dict(created=now,training_completed=True,quality_accepted=False,candidate_adopted=False,
    original_training_configuration_retained=True,model_architecture_changed=False,
    postprocess_pruning=False,source_masks_changed=False,reviewed_source_views=59,
    reviewed_face_views=59,reviewed_orbit_views=12,
    reviewed_files=[f'all_{k}_{i:02d}.png' for k in ['body','face'] for i in range(1,6)]+
        ['all_orbit_000.png','all_orbit_180.png','views/frame_000028/white_compare.png',
         'views/frame_000059/checker_compare.png','views/frame_000001/black_compare.png'],
    mask_is_connected_to_gradients=True,face_exterior_alpha_reduction=face_reduction,
    foreground_alpha_drop=inside_drop,face_L1_relative_increase=face_error_increase,
    planned_criteria=dict(face_exterior_reduction_at_least_50_percent=face_reduction>=.5,
        foreground_alpha_drop_at_most_001=inside_drop<=.01,held_object_appearance_retained=False),
    source_tree_file_hashes_verified_equal=3042,summary_paragraphs=summary,
    remaining_cause_not_fully_established=True,next_training_started=False)
save(ROOT/'review-findings.json',review)
c.update(quality_accepted=False,candidate_adopted=False,review_completed=True)
save(ROOT/'comparison.json',c)

# Clarify descriptive fields inherited from the baseline protocol. Training inputs,
# numerical configuration, model code and model files are never changed here.
plan=read(RUN/'plan.json')
plan.update(comparability='Same transparent4x inputs, model, optimizer, seed, initial points and MSV2 schedule; only the additional exterior-boundary mask penalty differs.',
    known_baseline_limitations='Held-object ambiguity and missing back observations retained. Existing masks, cameras, source images and existing loss coefficients unchanged; one extra mask term added.',
    user_selected_visible_improvement=False,user_selected_baseline_visible_improvement=True,
    documentation_clarification=dict(at=now,reason='Inherited baseline wording clarified; no numerical training settings changed'))
save(RUN/'plan.json',plan);save(ROOT/'mask_penalty/plan.json',plan)
delivery=read(ROOT/'mask_penalty/delivery.json');out=ROOT/'mask_penalty'
world=out/delivery['world_ply'];assert sha(world)==sha(RUN/'model/point_cloud/iteration_18000/point_cloud.ply')
bundle=out/delivery['bundle'];temporary=bundle.with_suffix('.zip.metadata-update')
with zipfile.ZipFile(bundle) as src,zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as dst:
    for info in src.infolist():
        content=(out/'plan.json').read_bytes() if info.filename=='plan.json' else src.read(info.filename)
        dst.writestr(info,content)
temporary.replace(bundle)
delivery.update(sha256=sha(bundle),bytes=bundle.stat().st_size,candidate_adopted=False)
save(out/'delivery.json',delivery)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    assert hashlib.sha256(z.read('point_cloud.ply')).hexdigest()==sha(world)
    assert hashlib.sha256(z.read('cameras.json')).hexdigest()==sha(out/'cameras_world.json')
    assert len(json.loads(z.read('cameras.json')))==59

viewer=dict(verified=now,actual_open_file_entry_tested=True,source='Local viewer http://127.0.0.1:8870/?sh=3',
    gaussian_mode_observed=True,point_mode_observed=True,backgrounds_observed=['gray','light'],
    points_displayed=87217,SH_degree_displayed=3,paired_cameras_displayed=59,
    first_paired_camera_upright=True,world_ply_sha256=sha(world),camera_sha256=sha(out/'cameras_world.json'),
    screenshot_saved=False,quality_accepted=False,upright_coordinate_export_verified=False,
    metadata_repacked_after_visual_check=True,PLY_and_camera_payloads_unchanged=True)
save(ROOT/'viewer-import-verification.json',viewer)

html_path=ROOT/'查看mask对照.html';html=html_path.read_text(encoding='utf-8')
start=html.index('<section><h2>结论</h2>');end=html.index('</section>',start)+len('</section>')
html=html[:start]+'<section><h2>结论</h2>'+''.join('<p>'+escape(p)+'</p>' for p in summary)+'</section>'+html[end:]
html_path.write_text(html,encoding='utf-8')
main=ROOT.parent/'查看对照.html';text=main.read_text(encoding='utf-8')
link='<p><a href="mask_penalty4x/查看mask对照.html">只加强mask惩罚的完成对照：噪点减少，但人物覆盖损失，未采用</a></p>'
if 'mask_penalty4x/查看mask对照.html' not in text:
    text=text.replace('</section>',link+'</section>',1);main.write_text(text,encoding='utf-8')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
for page in [html_path,main]:
    p=Links();p.feed(page.read_text(encoding='utf-8'))
    for link in p.links:
        u=urlsplit(link)
        if not u.scheme and u.path:assert (page.parent/unquote(u.path)).is_file(),link
archive=read(ROOT/'archive-verification.json');archive.update(final_review_recorded=True,final_HTML_links_verified=True,
    world_PLY_byte_identical_to_training=True,ZIP_payloads_verified=True,upright_export_verified=False)
save(ROOT/'archive-verification.json',archive)

status_path=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
status=read(status_path);state=status['msv2_transparent_1x2x4x_comparison']
state.update(state='comparison_complete_mask_penalty_evaluated_not_adopted',updated=now,
    mask_fix_report=str(html_path),mask_fix_training_completed=True,mask_fix_quality_accepted=False,
    mask_fix_adopted=False,pending_gpu_phases=[],own_gpu_jobs_running=False,
    model_architecture_unchanged=True,postprocess_pruning=False)
save(status_path,status)
print(json.dumps(dict(record=str(html_path),face_exterior_alpha_reduction=face_reduction,
    inside_alpha_drop=inside_drop,candidate_adopted=False,world_bundle_sha256=delivery['sha256']),ensure_ascii=False))
