import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const helper='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/runtime_helpers.mjs';
console.log('HELPER_IMPORT',new Date().toISOString());
const {importRuntimeModule}=await import(pathToFileURL(helper).href);
const {PresentationFile,FileBlob}=await importRuntimeModule('@oai/artifact-tool');
console.log('RUNTIME_IMPORTED',new Date().toISOString());
const file='clothing-candidate.pptx',before=crypto.createHash('sha256').update(await fs.readFile(file)).digest('hex');
const p=await PresentationFile.importPptx(await FileBlob.load(file));
console.log('PPTX_IMPORTED',p.slides.items.length,new Date().toISOString());
const after=crypto.createHash('sha256').update(await fs.readFile(file)).digest('hex');
if(before!==after||p.slides.items.length!==54)throw Error('Import mismatch');
await fs.writeFile('clothing-direct-import.json',JSON.stringify({passed:true,slides:p.slides.items.length,sha256:after}));
console.log('DONE');

