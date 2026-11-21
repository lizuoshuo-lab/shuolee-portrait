import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const b='C:/Users/Administrator/Documents/New project/shuolee-slide-update';
const src='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']})).records;
await fs.writeFile(b+'/mask-deck-before.json',JSON.stringify(snap,null,2));
console.log(JSON.stringify(snap.filter((_,i)=>[19,49,50,51].includes(i))));
for(const i of [19,49,50,51])await fs.writeFile(b+'/mask-before-'+(i+1)+'.png',new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
