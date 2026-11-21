import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const b='C:/Users/Administrator/Documents/New project/shuolee-slide-update';
const source='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style','notes.text']});
await fs.writeFile(b+'/076-slide52-inspect.json',JSON.stringify(snap.records,null,2));
await fs.writeFile(b+'/076-slide52-before.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
console.log(JSON.stringify(snap.records.map(s=>({id:s.id,layout:s.layout,elements:s.elements.map((e,k)=>({k,id:e.id,kind:e.kind,text:e.text,position:e.position}))})),null,2));
