import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=process.cwd();
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_地面修订.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const validation=await finalizePresentation({workspaceDir:workspace,candidatePath:build+'/ground-final-candidate.pptx',finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:53,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:build+'/source-before-ground.pptx',referenceSha256:'7342d60a349f4cf958043deb0cd7f1d43567249fcfa65aa764a8406b6b5d3114'},
 verifyArtifactToolImport:true,receiptPath:build+'/ground-final-validation.json'});
console.log(JSON.stringify(validation));
const p=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const after=(await p.inspect({kind:'slide,textbox,image,shape',maxChars:1000000})).ndjson.trim().split('\n').map(JSON.parse);
const before=(await fs.readFile('ground-source-inspection.ndjson','utf8')).trim().split('\n').map(JSON.parse);
const summarize=(rows,i)=>rows.filter(r=>r.slideIndex===i&&r.kind!=='slide').map(r=>({kind:r.kind,text:r.text,position:r.position,style:r.style}));
const diffs=[];
for(let i=0;i<51;i++)if(JSON.stringify(summarize(before,i))!==JSON.stringify(summarize(after,i)))diffs.push(i+1);
if(diffs.length)throw Error('Unexpected changes '+diffs);
await fs.mkdir('ground-final-import-renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile('ground-final-import-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
}
const montage=await p.export({format:'png',montage:true,scale:.22});await fs.writeFile('ground-final-import-montage.png',new Uint8Array(await montage.arrayBuffer()));
await fs.writeFile('ground-unchanged-slide-check.json',JSON.stringify({slideCount:p.slides.items.length,modifiedSlides:[52,53],unexpectedDifferences:diffs,finalImportedAndRendered:true},null,2));
console.log('Final imported; all53 rendered;51 unchanged slides match');
