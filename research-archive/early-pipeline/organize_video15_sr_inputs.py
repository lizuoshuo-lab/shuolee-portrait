from pathlib import Path
import json, hashlib, shutil, datetime, time
import numpy as np
from PIL import Image

record=Path('D:/00/记录/2026-11-05_MSV2_video15').resolve(strict=True)
runs=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-video15-20261105/runs')
full_run=runs/'video15/swinir4x'
paired_run=runs/'video15-parent1-child4/swinir4x'
full=json.loads((full_run/'superresolution.json').read_text(encoding='utf-8'))
paired=json.loads((paired_run/'superresolution.json').read_text(encoding='utf-8'))
assert full['state']==paired['state']=='completed'
full_rows={x['name']:x for x in full['records']}
pair_rows={x['name']:x for x in paired['records']}
frames=sorted(p for p in (record/'frames').iterdir() if p.is_dir() and p.name.startswith('frame_'))
assert len(frames)==len(full_rows)==len(pair_rows)==65
assert {p.name+'.png' for p in frames}==set(full_rows)==set(pair_rows)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

original_files=[p/name for p in frames for name in ('input.png','original.png','interpolation2x.png','swinir4x.png')]
protected={str(p.relative_to(record)):sha(p) for p in original_files}
selected=[]
for folder in frames:
 name=folder.name+'.png';a,b=full_rows[name],pair_rows[name]
 assert a['native_sha256']==b['native_sha256']
 selected.append((full_run/'dataset/images'/name,folder/'input_swinir4x.png',a['output_sha256'],[6080,10816],folder.name,'full_frame_swinir4x',None))
 for i,child in enumerate(b['children'],1):
  selected.append((paired_run/child['image'],folder/f'input_face_swinir4x_{i:03d}.png',child['image_sha256'],child['wh'],folder.name,'paired_face_child4x',child['roi_native']))
bytes_to_copy=sum(a.stat().st_size for a,b,*_ in selected if not b.exists())
assert shutil.disk_usage(record).free-bytes_to_copy>25*1024**3
for source,destination,expected,wh,frame,kind,roi in selected:
 assert source.is_file() and destination.parent.parent==record/'frames'
 assert source.resolve().is_relative_to(full_run.resolve()) or source.resolve().is_relative_to(paired_run.resolve())
 assert destination.resolve().is_relative_to((record/'frames').resolve())
 if destination.exists():assert sha(destination)==expected,'Refuse to replace existing unrelated input: '+str(destination)
 else:assert not destination.with_name(destination.name+'.copying').exists()

started=time.monotonic();receipts=[]
for i,folder in enumerate(frames,1):
 name=folder.name+'.png';paired_item=pair_rows[name]
 with Image.open(folder/'input.png') as native:
  assert native.mode=='RGB' and list(native.size)==[1520,2704]
  rgb_sha=hashlib.sha256(np.asarray(native).tobytes()).hexdigest()
 assert rgb_sha==paired_item['composition']['source_rgb_sha256']
 for source,destination,expected,wh,frame,kind,roi in selected:
  if frame!=folder.name:continue
  assert sha(source)==expected,'Source differs from its frozen provenance'
  if not destination.exists():
   temporary=destination.with_name(destination.name+'.copying')
   shutil.copy2(source,temporary)
   assert sha(temporary)==expected
   temporary.replace(destination)
  assert sha(destination)==expected
  with Image.open(destination) as image:assert image.mode=='RGB' and list(image.size)==wh
  receipts.append(dict(frame=frame,kind=kind,source=str(source),file=str(destination.relative_to(record)),sha256=expected,wh=wh,roi_native=roi,resized=False,reencoded=False))
 if i%10==0 or i==65:print(json.dumps({'copied_frames':i,'total_frames':65,'seconds':round(time.monotonic()-started,2)}),flush=True)

children={folder.name:[{'file':f'input_face_swinir4x_{i:03d}.png','wh':c['wh']} for i,c in enumerate(pair_rows[folder.name+'.png']['children'],1)] for folder in frames}
block='\n<!-- video15-sr-training-inputs-v1 -->\n<section id="sr-training-inputs"><h2>实际超分输入图 · 训练前</h2>\n<p>以下是实际送入训练的输入像素，不是高斯模型渲染。整图四倍输入属于昨天的整图超分对照；面部子图属于这次的“mask外单倍、内部四倍”联合训练。点击图片可查看文件原始分辨率。</p>\n<div class="sr-input-grid"><figure><figcaption>昨天整图 SwinIR 四倍输入 · 6080×10816</figcaption><a id="sr-input-full-link" target="_blank"><img id="sr-input-full-image" loading="lazy" alt="完整四倍超分训练输入"></a></figure><div id="sr-input-children"></div></div>\n<p><a href="超分输入整理.json">输入来源与文件核验</a></p></section>\n<style>.sr-input-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.sr-input-grid figure{background:#fff;padding:12px}.sr-input-grid img{display:block;max-height:620px;object-fit:contain}.sr-input-grid figcaption{margin-bottom:8px}@media(max-width:900px){.sr-input-grid{grid-template-columns:1fr}}</style>\n<script>(()=>{const byFrame=__CHILDREN__;const picker=document.querySelector(\'#frame\');function updateInputs(){const frame=picker.value;const full=\'frames/\'+frame+\'/input_swinir4x.png\';document.querySelector(\'#sr-input-full-link\').href=full;document.querySelector(\'#sr-input-full-image\').src=full;const host=document.querySelector(\'#sr-input-children\');host.replaceChildren();const children=byFrame[frame]||[];if(!children.length){const p=document.createElement(\'p\');p.textContent=\'此帧无面部子图，使用单倍母图。\';host.append(p);}for(let i=0;i<children.length;i++){const child=children[i];const figure=document.createElement(\'figure\');const caption=document.createElement(\'figcaption\');caption.textContent=\'本次面部子图实际输入 \'+(i+1)+\' · \'+child.wh[0]+\'×\'+child.wh[1]+\'（mask内超分四倍，外部为原图上下文）\';const link=document.createElement(\'a\');link.href=\'frames/\'+frame+\'/\'+child.file;link.target=\'_blank\';const image=document.createElement(\'img\');image.src=link.href;image.loading=\'lazy\';image.alt=\'面部四倍训练输入\';link.append(image);figure.append(caption,link);host.append(figure);}}picker.addEventListener(\'change\',updateInputs);updateInputs();})();</script>\n<!-- /video15-sr-training-inputs-v1 -->\n'
block=block.replace('__CHILDREN__',json.dumps(children,ensure_ascii=False,separators=(',',':')))
html_receipts=[]
for name in ('comparison.html','查看完整图倍率对照.html'):
 path=record/name;original=path.read_bytes().decode('utf-8')
 marker='<!-- video15-sr-training-inputs-v1 -->'
 if marker in original:
  assert block in original
  html_receipts.append(dict(file=name,already_present=True,sha256=sha(path)))
  continue
 assert original.count('</html>')==1
 updated=original.replace('</html>',block+'</html>')
 assert updated.replace(block,'')==original
 path.write_text(updated,encoding='utf-8',newline='\n')
 html_receipts.append(dict(file=name,before_sha256=hashlib.sha256(original.encode()).hexdigest(),after_sha256=sha(path),original_content_preserved=True))
assert all(sha(record/name)==expected for name,expected in protected.items())
assert len(receipts)==130
assert all((record/x['file']).is_file() and sha(record/x['file'])==x['sha256'] for x in receipts)
manifest=dict(state='completed',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),destination=str(record/'frames'),frames=65,full4x_inputs=65,face4x_child_inputs=65,source_RGB_roster_matched=65,files=receipts,original_render_and_source_files_unchanged=len(protected),protected_sha256=protected,HTML=html_receipts,total_copied_bytes=sum((record/x['file']).stat().st_size for x in receipts),new_training=False,new_model_rendering=False)
(record/'超分输入整理.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':'completed','frames':65,'full4x':65,'face_children':65,'old_files_unchanged':len(protected),'copied_gib':manifest['total_copied_bytes']/1024**3,'seconds':round(time.monotonic()-started,2)}),flush=True)
