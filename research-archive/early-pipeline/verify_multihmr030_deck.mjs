import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const build='C:/Users/Administrator/Documents/New project/shuolee-slide-update';
const source='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx';
const finalPath='C:/Users/Administrator/Documents/New project/shuolee-slide-output/02_有限视角重建优化路线_身体先验配准_定稿.pptx';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const sourceBytes=await fs.readFile(source);
if(digest(sourceBytes)!=='833baa3da9b4dec1c28576eec57708febf8d4e04f860d6aa7f8d8536f903c87f')throw Error('Source changed');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
if(p.slides.items.length!==55||q.slides.items.length!==55)throw Error('Slide count');
const out=build+'/multihmr030-final-renders';await fs.mkdir(out,{recursive:true});
const rows=[];
for(let i=0;i<55;i++){
 const newBytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(out+'/'+String(i+1).padStart(2,'0')+'.png',newBytes);
 const oldBytes=new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer());
 rows.push({slide:i+1,source_render_sha256:digest(oldBytes),final_render_sha256:digest(newBytes),unchanged:digest(oldBytes)===digest(newBytes)});
}
if(rows.slice(0,54).some(r=>!r.unchanged))throw Error('Unrelated slide changed');
await fs.writeFile(build+'/multihmr030-deck-receipt.json',JSON.stringify({source,source_sha256:digest(sourceBytes),finalPath,
 final_sha256:digest(await fs.readFile(finalPath)),changed_slides:[55],all_slides_rendered:55,unchanged_first54_render_hashes:54,
 renders:rows,canonical_replaced:false},null,2));
console.log(JSON.stringify({all_slides_rendered:55,unchanged_first54_render_hashes:54,changed_slide:55}));
