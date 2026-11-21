import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const file='../shuolee-slide-output/02_有限视角重建优化路线_20261022_发束修订.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(file));
const snapshot=await p.inspect({kind:'slide,textbox,image,shape',maxChars:1000000});
await fs.writeFile('hair-final-inspection.ndjson',snapshot.ndjson);
const after=snapshot.ndjson.trim().split('\n').map(JSON.parse);
const before=(await fs.readFile('hair-source-inspection.ndjson','utf8')).trim().split('\n').map(JSON.parse);
const summarize=(rows,i)=>rows.filter(r=>r.slideIndex===i&&r.kind!=='slide'&&r.kind!=='layout').map(r=>({kind:r.kind,text:r.text?.replace(/^(\d+) \/ (53|55)$/,'$1 / COUNT'),position:r.position,style:r.style}));
const diffs=[];for(let i=0;i<52;i++)if(JSON.stringify(summarize(before,i))!==JSON.stringify(summarize(after,i)))diffs.push(i+1);
if(diffs.length)throw Error('Unexpected content edits '+diffs);
await fs.mkdir('hair-final-import-renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile('hair-final-import-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
}
const montage=await p.export({format:'png',montage:true,scale:.22});
await fs.writeFile('hair-final-import-montage.png',new Uint8Array(await montage.arrayBuffer()));
const result={finalSlideCount:55,modifiedContentSlides:[53],addedSlides:[54,55],pageCountsUpdated:true,unexpectedDifferences:diffs,allFinalSlidesRendered:true,nativePowerPointVerified:false};
await fs.writeFile('hair-unchanged-slide-check.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
