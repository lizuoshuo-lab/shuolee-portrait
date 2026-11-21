import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const source='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const report=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile('vposer-slide-before.ndjson',report.ndjson);
const image=await p.export({slide:p.slides.items[54],format:'png',scale:1});
await fs.writeFile('vposer-slide-before.png',new Uint8Array(await image.arrayBuffer()));
console.log(report.ndjson.slice(0,18000));
