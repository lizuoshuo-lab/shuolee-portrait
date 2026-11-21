import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const workspace='C:/Users/Administrator/Documents/New project',build=workspace+'/shuolee-slide-update';
const source='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx';
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_足部对应说明.pptx';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const expected='1c3844d0b12c4c836b60f0a19b54573c8aa9e337f804fcf546a43fb585d38b01';
if(await hash(source)!==expected)throw Error('Canonical changed');
const previous=JSON.parse(await fs.readFile(build+'/head048-deck-receipt.json','utf8'));
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const appendix=await fs.readFile(build+'/foot050-notes-append.txt','utf8');
const notes=q.slides.items[54].speakerNotes.text;
const normalize=x=>x.replace(/\s+/g,'');
await fs.writeFile(build+'/foot050-notes-imported.txt',notes);
if(!normalize(notes).includes(normalize(appendix)))throw Error('Notes content missing');
const old=await fs.readFile(build+'/head048-notes.txt','utf8');
if(!normalize(notes).includes(normalize(old)))throw Error('Previous notes content lost');
const checks=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 const h=digest(bytes);if(h!==previous.renders[i].render_sha256)throw Error('Unintended visual change '+(i+1));
 checks.push({slide:i+1,render_sha256:h,unchanged:true});
 if(i===54)await fs.writeFile(build+'/foot050-final55.png',bytes);
}
await fs.writeFile(build+'/foot050-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[],changed_notes_slides:[55],all_slides_rendered:55,renders:checks,
 notes_preserved:true,import_normalizes_paragraph_whitespace:true,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,visual_parity_all55:true,changed_notes:[55],notes_preserved:true}));
