import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const receipt=JSON.parse(await fs.readFile(build+'/human116-p3-deck-receipt.json','utf8'));
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_116末页定稿.pptx';
await finalizePresentation({workspaceDir:work,candidatePath:build+'/human116-p3-exact-candidate.pptx',finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:receipt.source,referenceSha256:receipt.source_sha256},verifyArtifactToolImport:true,receiptPath:build+'/human116-p3-exact-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 if(digest(bytes)!==receipt.renders[i].render_sha256)throw Error('Package preservation changed rendering '+(i+1));
 if(i===51)await fs.writeFile(build+'/human116-p3-exact-52.png',bytes);
}
receipt.exact_final_path=finalPath;receipt.exact_final_sha256=digest(await fs.readFile(finalPath));
receipt.package_preservation='All members except slide52,its notes,relationships and unique image are byte-identical to source';
await fs.writeFile(build+'/human116-p3-deck-receipt.json',JSON.stringify(receipt,null,2));
console.log(JSON.stringify({validated:finalPath,canonical_replaced:false,all52renders_identical_to_reviewed:true}));
