import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const p=await PresentationFile.importPptx(await FileBlob.load(base+'/reports/02_有限视角重建优化路线.pptx'));
const snap=await p.inspect({select:'$.slides[51]',include:['id','elements.id','elements.kind','elements.text','elements.alt','elements.position','notes.text']});
await fs.writeFile('C:/Users/Administrator/Documents/New project/shuolee-slide-update/human093-source-inspection.json',JSON.stringify(snap,null,2));
console.log(JSON.stringify(snap.records.map(r=>({id:r.id,elements:r.elements})),null,2));
await fs.writeFile('C:/Users/Administrator/Documents/New project/shuolee-slide-update/human093-source-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
