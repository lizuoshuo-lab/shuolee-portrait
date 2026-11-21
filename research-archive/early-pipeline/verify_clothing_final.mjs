import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const file='../shuolee-slide-output/02_有限视角重建优化路线_20261022_衣物修订.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(file));
const after=(await p.inspect({kind:'slide,textbox,image,shape',maxChars:1000000})).ndjson.trim().split('\n').map(JSON.parse);
const before=(await fs.readFile('clothing-source-inspection.ndjson','utf8')).trim().split('\n').map(JSON.parse);
const summarize=(rows,i)=>rows.filter(r=>r.slideIndex===i&&r.kind!=='slide').map(r=>({kind:r.kind,text:r.text,position:r.position,style:r.style}));
const changed=new Set([1,52]),diffs=[];
for(let i=0;i<53;i++)if(!changed.has(i)&&JSON.stringify(summarize(before,i))!==JSON.stringify(summarize(after,i)))diffs.push(i+1);
if(diffs.length)throw Error('Unexpected slide edits '+diffs);
for(const n of [2,53]){
 const png=await p.export({slide:p.slides.items[n-1],format:'png',scale:1});
 await fs.writeFile('clothing-final-renders/final-import-'+n+'.png',new Uint8Array(await png.arrayBuffer()));
}
const montage=await p.export({format:'png',montage:true,scale:.22});
await fs.writeFile('clothing-final-import-montage.png',new Uint8Array(await montage.arrayBuffer()));
const result={finalSlideCount:53,modifiedSlides:[2,53],unexpectedDifferences:diffs,importedFinalRenderVerified:true};
await fs.writeFile('clothing-unchanged-slide-check.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));

