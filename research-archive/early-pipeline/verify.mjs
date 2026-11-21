import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
for (const [label,file] of [['source','source.pptx'],['final','../shuolee-slide-output/02_有限视角重建优化路线_20261022.pptx']]) {
 const p=await PresentationFile.importPptx(await FileBlob.load(file));
 const mont=await p.export({format:'png',montage:true,scale:.22});
 await fs.writeFile(`${label}-montage.png`,new Uint8Array(await mont.arrayBuffer()));
 const inspect=await p.inspect({kind:'slide,textbox,image',maxChars:1000000});
 await fs.writeFile(`${label}-final-inspection.ndjson`,inspect.ndjson);
}
const before=(await fs.readFile('source-final-inspection.ndjson','utf8')).trim().split('\n').map(JSON.parse);
const after=(await fs.readFile('final-final-inspection.ndjson','utf8')).trim().split('\n').map(JSON.parse);
const changed=new Set([1,32,33,40,50,51,52]);
function content(rows,i){return rows.filter(r=>r.slideIndex===i&&r.kind!=='slide'&&!/^\d+\s*\/\s*\d+$/.test(r.text??'')).map(r=>({kind:r.kind,text:r.text,position:r.position,style:r.style}));}
const diffs=[];
for(let i=0;i<51;i++)if(!changed.has(i)&&JSON.stringify(content(before,i))!==JSON.stringify(content(after,i)))diffs.push(i+1);
await fs.writeFile('unchanged-slide-check.json',JSON.stringify({sourceSlides:51,finalSlides:53,changedSlides:[...changed].map(x=>x+1),unexpectedContentOrLayoutDifferences:diffs},null,2));
console.log(JSON.stringify({unexpectedDifferences:diffs}));
