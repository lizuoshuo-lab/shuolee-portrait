import fs from 'node:fs/promises';
const old=await fs.readFile('update_clothing.mjs','utf8');
let text=old.slice(0,old.indexOf('const s54='));
const shortStart=old.indexOf("s53.speakerNotes.text='本地观测");
const shortEnd=old.indexOf("\ns54.speakerNotes.text=",shortStart);
text+=old.slice(shortStart,shortEnd)+'\n';
text+="text(s53,'remaining','未知俯仰的头部碎片与脚边背景缺口仍在。静态和动态衣物效果接近，整体质量尚未通过。',66,665,1040,26,17,false,'#536674');\n";
const renderStart=old.indexOf("await fs.mkdir('clothing-renders'");
text+=old.slice(renderStart);
text=text.replaceAll("clothing-renders","clothing-final-renders").replaceAll("clothing-candidate.pptx","clothing-final-candidate.pptx").replaceAll("clothing-validation.json","clothing-final-validation.json").replaceAll("clothing-final-inspection.ndjson","clothing-final53-inspection.ndjson").replaceAll("clothing-montage.png","clothing-final53-montage.png");
text=text.replace("explicitTotalSlideCount:54","explicitTotalSlideCount:53").replace("n+' / 54'","n+' / 53'").replace("观测衣物高斯与独立时间形变","衣物观测补充与当前结论").replace("66,210,1148,398","66,205,1148,388").replace("66,627,1148,42,25","66,610,1148,40,24");
text=text.replace("02_有限视角重建优化路线_20261022_衣物迭代.pptx","02_有限视角重建优化路线_20261022_衣物修订.pptx");
await fs.writeFile('update_clothing_53.mjs',text);

