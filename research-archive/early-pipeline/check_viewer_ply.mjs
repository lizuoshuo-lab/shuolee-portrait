import fs from 'node:fs';
import crypto from 'node:crypto';
const [loader,ply,expectedHash,expectedCount]=process.argv.slice(2);
if(!loader||!ply||!expectedHash||!expectedCount)throw Error('Expected loader, PLY, SHA256, count');
const source=fs.readFileSync(loader,'utf8');
const {inspectPly}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const bytes=fs.readFileSync(ply);
const header=inspectPly(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
const digest=crypto.createHash('sha256').update(bytes).digest('hex');
if(header.kind!=='gaussian'||header.count!==Number(expectedCount)||digest!==expectedHash)throw Error('Shuolee Gaussian classification/count/hash mismatch');
if(header.format!=='binary_little_endian')throw Error('Expected standard binary little endian PLY');
console.log(JSON.stringify({passed:true,kind:header.kind,count:header.count,format:header.format,sha256:digest,
  loader_sha256:crypto.createHash('sha256').update(source).digest('hex'),
  scope:'Actual viewer header classifier; GPU display requires separate browser verification.'}));
