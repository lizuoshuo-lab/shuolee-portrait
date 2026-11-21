import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('source.pptx'));
const content=await p.inspect({kind:'slide,textbox,shape,image,notes',maxChars:1000000});
await fs.writeFile('inspection.ndjson',content.ndjson);
const layouts=await p.inspect({kind:'layout',maxChars:3000});
console.log(layouts.ndjson);
for(const i of [1,32,33,50]) {
 const slide=p.slides.items[i];
 const png=await p.export({slide,format:'png',scale:1});
 await fs.writeFile(`before-${i+1}.png`,new Uint8Array(await png.arrayBuffer()));
 const layout=await slide.export({format:'layout'});
 await fs.writeFile(`before-${i+1}.layout.json`,new Uint8Array(await layout.arrayBuffer()));
}
console.log('INSPECTED',p.slides.items.length);
