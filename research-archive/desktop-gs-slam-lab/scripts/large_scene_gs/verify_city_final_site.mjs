import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const LAB='C:/Users/Administrator/Desktop/GS-SLAM-Lab';
const SITE='D:/Projects/EngineLab/engine-home';
const REL='outputs/large-scene-gs-001/research/city-final-website-021';
const OUT=`${LAB}/${REL}`;
const ORIGIN='http://127.0.0.1:8799';
const hash=x=>createHash('sha256').update(x).digest('hex');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const theory=s=>s.split('## 实验附录')[0].replace(/^updated:.*$/m,'updated: <metadata>').replace(/\r\n/g,'\n');
const index=await read(`${SITE}/lib/gs-lab-content.json`);
const doc=index.documents.find(x=>x.slug==='large-scene-gs-sfm-partition-lod');
assert.ok(doc);
const main=doc.html.split(/<h2[^>]*>实验附录<\/h2>/)[0];
assert.equal((main.match(/<math\b/g)||[]).length,100);
assert.equal((main.match(/<img\b/g)||[]).length,6);
assert.ok(!/<(?:pre|code)(?:\s|>)/.test(main));
assert.equal(theory(await fs.readFile(`${OUT}/chapter-before.md`,'utf8')),
  theory(await fs.readFile(`${LAB}/knowledge/30_large_scene_gs_sfm_partition_lod.md`,'utf8')));
const response=await fetch(`${ORIGIN}/gs-lab/read/large-scene-gs-sfm-partition-lod`);
assert.equal(response.status,200);
const html=await response.text();
for(const text of ['完整质量、阶段耗时和显存对照已完成','1,404 张图像','城市完整留出对照：层级优化、画质与资源',
  '没有复现原 NaN','预检产物不用于正式合并','不是连续播放帧率','不覆盖全部原生临时工作区'])assert.ok(html.includes(text),`Missing ${text}`);
for(const text of ['完整质量、耗时和显存对照尚未完成','完整本地层级渲染对照仍在进行','完整 LOD 质量结论仍待全部渲染'])assert.ok(!html.includes(text),`Stale ${text}`);
const details=[...html.matchAll(/<details\b([^>]*)>([\s\S]*?)<\/details>/g)].filter(x=>/<summary[^>]*>[^<]*实验附录/.test(x[2]));
assert.equal(details.length,1);
assert.ok(!/\bopen\b/.test(details[0][1]));
const summary=await read(`${OUT}/verified-city-comparison.json`);
assert.equal(summary.status,'complete_actual_City1404_numeric_and_selected_visual_comparison');
assert.equal(summary.actual_verified_frames,1404);
assert.equal(summary.rows.length,12);
assert.equal(summary.manual_dense_boundary_comparison_complete,true);
assert.equal(summary.viewed_original_pairs,32);
assert.equal(summary.formal_observation_reclassification,false);
for(const source of summary.local_sources)assert.equal(hash(await fs.readFile(`${LAB}/${source.path}`)),source.sha256);
const numericHtml=details[0][2].replace(/&rarr;|&#8594;|&#x2192;/g,'→');
for(const setting of ['leaves','tau1','tau3','tau5','tau6','tau15']){
 const a=summary.rows.find(r=>r.model==='merged-unoptimized'&&r.setting===setting);
 const b=summary.rows.find(r=>r.model==='merged-post'&&r.setting===setting);
 assert.equal(a.views,117);assert.equal(b.views,117);
 for(const [key,precision] of [['psnr_numpy_mean_per_image_db',4],['ssim_mean_per_image',6],['lpips_alex_mean_per_image',6]]){
  const cell=`${a.quality[key].toFixed(precision)} → ${b.quality[key].toFixed(precision)}`;
  assert.ok(numericHtml.includes(cell),`Missing quality pair ${setting}/${key}: ${cell}`);
 }
 assert.ok(numericHtml.includes(b.stage_p50_ms.sum_synchronized_GPU_stages_seconds.toFixed(3)));
 assert.ok(numericHtml.includes(b.torch_peak_allocated_GiB.toFixed(3)));
}
const resources=[];
for(const rel of ['manifests/large-scene-gs-work.json',`${REL}/verified-city-comparison.json`]){
 const r=index.resources.find(x=>x.path===rel);
 assert.ok(r&&details[0][2].includes(r.url));
 assert.equal(hash(await fs.readFile(`${LAB}/${rel}`)),r.source_sha256);
 const response=await fetch(`${ORIGIN}${r.url}`);assert.equal(response.status,200);
 const bytes=Buffer.from(await response.arrayBuffer());assert.equal(hash(bytes),r.sha256);
 if(rel.startsWith('manifests/')){
  const manifest=JSON.parse(bytes);
  assert.equal(manifest.status,'completed_local_experiment_scope_presentation_delivery_separate');
  const city=manifest.milestones.find(x=>x.id==='public-large-scene-extension');
  assert.equal(city.local_hierarchy006_actual_City1404_evaluation_frames,1404);
  assert.equal(city.local_hierarchy006_actual_City1404_pipeline_complete,true);
  assert.deepEqual(city.local_formal_completed_chunks,['0_0','0_1','1_0']);
  assert.equal(city.local_formal_chunk11_failed.nonfinite_gaussians,1);
  assert.equal(city.local_observation_only_diagnostic.formal_completion_claimed,false);
 }
 resources.push({path:rel,HTTP:response.status,sha256:hash(bytes)});
}
const figures=[];
for(const match of main.matchAll(/<img\b[^>]*\bsrc="([^"]+)"/g)){
 const url=match[1],r=index.resources.find(x=>x.url===url);
 assert.ok(r,`Paper figure absent from resource index: ${url}`);
 assert.equal(hash(await fs.readFile(`${LAB}/${r.path}`)),r.source_sha256);
 const response=await fetch(`${ORIGIN}${url}`);assert.equal(response.status,200);
 assert.equal(hash(Buffer.from(await response.arrayBuffer())),r.sha256);
 figures.push(url);
}
assert.equal(figures.length,6);
assert.ok(!index.resources.some(x=>/\/small-city-hierarchy-006\/|\.(hier|ply|npz)$/.test(x.path)));
const report={status:'passed_final_city_actual_metrics_theory_figures_and_local_serving',UTC:new Date().toISOString(),
 HTTP:200,theoryBodyUnchanged:true,mathCount:100,fullPaperFigures:figures,singleClosedAppendix:true,
 actualVerifiedFrames:1404,reviewedOriginalPairs:32,resources,rawModelsOrDatasetsCopied:false};
await fs.writeFile(`${OUT}/served-final-check.json`,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
