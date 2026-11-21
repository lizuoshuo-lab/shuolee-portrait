import {FileBlob,PresentationFile} from '@oai/artifact-tool';
import fs from 'node:fs/promises';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/00/报告/02_有限视角重建优化路线.pptx'));
const s=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records[51];
console.log(JSON.stringify(s));
for(const e of s.elements){if(e.kind==='image')console.log(JSON.stringify({image:e.id,frame:p.resolve(e.id).frame,crop:p.resolve(e.id).crop}));}
