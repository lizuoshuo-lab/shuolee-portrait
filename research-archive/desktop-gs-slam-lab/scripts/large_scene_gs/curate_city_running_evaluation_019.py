"""Update only stale website status; actual full City scores stay unreported."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, re

LAB=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE=LAB/'outputs/large-scene-gs-001'
OUT=BASE/'research/city-running-evaluation-status-019'
EVAL=BASE/'independent-evaluation/small-city006-heldout-1404-001'
CPU=BASE/'independent-evaluation/small-city006-model-binding-CPU-001'
CHAPTER=LAB/'knowledge/30_large_scene_gs_sfm_partition_lod.md'
MANIFEST=LAB/'manifests/large-scene-gs-work.json'
SYNC=Path(r'D:\Projects\EngineLab\engine-home\scripts\sync-gs-lab.mjs')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=p.relative_to(LAB).as_posix(),sha256=sha(p))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def theory(s):return re.sub(r'(?m)^updated:.*$','updated: <metadata>',s.split('## 实验附录')[0]).replace('\r\n','\n')
def main():
 assert not OUT.exists(),'Curation already exists; do not replay'
 cpu=read(CPU/'outcome.json');assert cpu['status']=='passed_actual_local_pre_post_model_binding_CPU_no_GPU'
 assert cpu['formal_observation_reclassification'] is False and cpu['cuda_used'] is False
 outcome=read(EVAL/'outcome.json');assert outcome['status']=='running' and outcome['gpu_dispatch_attempted']
 groups=[]
 for model in ('merged-unoptimized','merged-post'):
  for setting in ('leaves','tau1','tau3','tau5','tau6','tau15'):
   folder=EVAL/model/setting;audit=folder/'per-frame-audit.json';timing=folder/'timings.json'
   if not (audit.is_file() and timing.is_file()):continue
   rows=read(audit);t=read(timing)
   assert len(rows)==117 and t['status']=='all117_valid' and t['valid_frames']==117
   assert all(r['model']==model and r['setting']==setting and r['status']=='valid_selection_interpolation_finite_render' for r in rows)
   groups.append(dict(model=model,setting=setting,driver_valid_frames=117,audit=ref(audit),timing=ref(timing)))
 assert len(groups)>=2
 utc=datetime.now(timezone.utc).isoformat()
 summary=dict(status='partial_actual_City_GPU_rendering_after_completed_CPU_model_binding',UTC=utc,evaluation_run=EVAL.name,CPU_model_binding=ref(CPU/'outcome.json'),CPU_model_binding_completed=True,completed_driver_frame_groups=groups,driver_valid_frames_snapshot=sum(r['driver_valid_frames'] for r in groups),count_scope='Only whole117-view groups with completed frame and timing audits; not an independent posterior completion or a quality score.',actual1404_pipeline_complete=False,quality_metrics_finalized=False,independent_posterior_passed=False,perceptual_CPU_passed=False,manual_dense_boundary_comparison_complete=False,formal_observation_reclassification=False,raw_models_or_datasets_copied_to_site=False)
 chapter=CHAPTER.read_text(encoding='utf-8');before=chapter
 replacements={
 'updated: "2026-10-12"':'updated: "2026-10-13"',
 '后续本地训练先采用四分之一分辨率，并记录真实进程与显存占用。':'本地训练采用四分之一分辨率，保存了实际进程与显存记录。',
 '拟用的本地训练划分按四个采集轮次':'本地训练划分按四个采集轮次',
 '第四块原常规运行没有通过验收，新的观察结果见下文；未优化全景合并已完成，优化后的配对合并及本地层级对照仍待完成。':'第四块原常规运行没有通过验收，新的观察结果见下文；优化前后两份全景合并均已完成，完整本地层级渲染对照仍在进行。',
 '同一组 117 个留出视角的质量、耗时和显存对照正在进行评测前的两份模型检查，尚无城市 LOD 渲染成绩。':'两份模型的独立 CPU 检查已通过，正在同一组 117 个留出视角上进行实际 GPU 渲染。两份模型各使用完整叶层及五档层次细节，共计 1,404 张图像；完整质量、耗时和显存对照尚未完成。',
 '这里核验的是固定 CPU 选择公式与资源估算，还没有核验原生 GPU 在浮点边界上的选择、实际工作区、画质或速度，不能作为完成城市 LOD 渲染评测的成绩。':'这组 CPU 结果核验固定选择公式与资源估算；原生 GPU 的选择、实际工作区和图像结果正在另一份完整评测中检查。部分设置完成渲染不等于整个城市 LOD 对照已经完成。',
 '城市合并图像与 LOD 评测尚未生成。':'城市合并模型现已开始生成实际图像，完整 LOD 质量结论仍待全部渲染及独立重算完成。',
 }
 for a,b in replacements.items():assert chapter.count(a)==1,a;chapter=chapter.replace(a,b)
 newrel=OUT.relative_to(LAB).as_posix()+'/verified-evaluation-progress.json'
 label='城市两份模型 CPU 核验与实际渲染状态摘要'
 anchor='- [城市四块完整层级优化与两份全景合并核验摘要]'
 line=next(x for x in chapter.splitlines() if x.startswith(anchor))
 chapter=chapter.replace(line,line+f'\n- [{label}](../{newrel})')
 assert theory(chapter)==theory(before)
 manifest=read(MANIFEST);city=next(r for r in manifest['milestones'] if r['id']=='public-large-scene-extension')
 assert city['local_hierarchy006_optimized_global_Merger_completed'] is True
 city.update(status='mixed4_valid_inputs_four_posts_two_merges_CPU_bound_actual_City1404_rendering',local_hierarchy006_native_and_supervision_status='Four full posts and both actual global merges complete. Independent CPU model binding passed. Actual117x12 GPU rendering underway; full metrics, independent posterior and visual review incomplete.',local_hierarchy006_actual_City1404_evaluation_frames=summary['driver_valid_frames_snapshot'],local_hierarchy006_actual_City1404_frame_count_scope=summary['count_scope'],local_hierarchy006_actual_City1404_frame_count_asof_utc=utc,local_hierarchy006_CPU_model_binding=ref(CPU/'outcome.json'),local_hierarchy006_actual_City1404_pipeline_complete=False)
 manifest['last_verified_utc']=utc
 sync=SYNC.read_text(encoding='utf-8');oldrel='outputs/large-scene-gs-001/research/city-four-post-paired-merges-016/verified-stage-summary.json'
 sync_anchor=f"  ['{oldrel}', '城市四块完整层级优化与两份全景合并核验摘要'],"
 assert sync.count(sync_anchor)==1 and newrel not in sync
 sync=sync.replace(sync_anchor,sync_anchor+f"\n  ['{newrel}', '{label}'],")
 OUT.mkdir(parents=True,exist_ok=False)
 for p,name in [(CHAPTER,'chapter-before.md'),(MANIFEST,'manifest-before.json'),(SYNC,'sync-before.mjs')]:shutil.copyfile(p,OUT/name)
 shutil.copyfile(__file__,OUT/'curator-source.py')
 save(OUT/'verified-evaluation-progress.json',summary)
 city['local_hierarchy006_evaluation_progress_summary']=ref(OUT/'verified-evaluation-progress.json')
 CHAPTER.write_text(chapter,encoding='utf-8');save(MANIFEST,manifest);SYNC.write_text(sync,encoding='utf-8')
 report=dict(status='source_status_correction_complete_local_serving_pending',UTC=utc,chapter=ref(CHAPTER),manifest=ref(MANIFEST),progress=ref(OUT/'verified-evaluation-progress.json'),theory_body_unchanged=True,updated_metadata_only_in_main=True,quality_scores_added=False,driver_valid_frames_snapshot=summary['driver_valid_frames_snapshot'])
 save(OUT/'source-update.json',report);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
