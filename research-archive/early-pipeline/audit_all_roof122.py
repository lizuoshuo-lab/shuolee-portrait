"""N12f exhaustive review preparation for existing roof families, no new measurements."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,sys,time
import numpy as np
from PIL import Image
I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
O=I/'clip03-grid-coverage/array-cache-repair';M=I/'clip03-scene-support';D=I/'clip03-roof-review';P=I/'clip03-roof-review-plan.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def register():
 assert not P.exists() and not D.exists()
 a=read(O/'allocation.json');v=read(O/'visual-index.json');rows=sorted((r for r in a['candidate_families'] if r['semantic_category']=='distant_roof'),key=lambda x:x['family'])
 assert len(rows)==31 and len({r['family'] for r in rows})==31
 assert Counter(r['split'] for r in rows)==Counter(fit=23,heldout=8)
 source=[O/n for n in ['allocation.json','visual-index.json','visual-review.json','observations.json','families.json']]+[M/f'frame_{f:06d}.png' for f in v['columns']]
 save(P,dict(stage='N12f',registered=datetime.now(timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),
  reason='N12d representative sampling limited each semantic category to4FIT/2heldout. Its nonground2heldout result does not prove all available independent roof support is absent. Exhaustively audit the entire already frozen roof category rather than selectively adding one point to pass the gate.',
  hypothesis='The material-support deficit may partly be an audit-sampling limit. All31existing roof families can establish the actual reviewed coarse support. New positives remain coarse until uncertainty/geometry independently checked.',
  causal_change='Read-only review coverage from6roof representatives to all31roof-family representatives. No feature, track, split, mask, K, local-refinement coordinate or camera change.',
  selection='All31 prior distant_roof category family representatives, ascending familyID, allnative observations. Include all FIT and heldout; no residual selection. Do not choose another representative within a family or relabel a rejected family based on error.',
  rows=rows,frames=v['columns'],budget=dict(runs=1,CPU_wall_seconds=60,review_rows=31,context_rows=31,cameras=0,training=0),
  review='Native128px point-centered patches plus256px context for every family, no text/warping in images. Record structural bay identity, opaque material vs glass/reflection/foliage, depth boundaries and ambiguity. Preserve all6 prior judgments unless new image evidence contradicts them with explicit reason.',
  gate='Keep N12d20FIT/8heldout total,6FIT/3heldout nonground,3imagebins,12sharedFITinitialpair and8FIT2D3D perlater camera. Passing coarse material count alone is not camera or subpixel admission. Separate fresh fixedK geometry plan is required.',
  stop='No new detector/matcher/threshold/window/region after this exhaustive category audit. If material support or observability is inadequate, preserve failure; no silent relaxation.',
  input_hashes={str(p):sha(p) for p in source}))
 print('N12F_REGISTERED',len(rows))
def run():
 p=read(P);assert sha(Path(__file__))==p['code_sha256'] and not D.exists();D.mkdir();start=time.monotonic()
 for path,h in p['input_hashes'].items():assert sha(Path(path))==h
 frames=p['frames'];ims={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in frames};rows=[];pages=[];contexts=[]
 for n,t in enumerate(p['rows']):
  assert time.monotonic()-start<60
  strip=Image.new('RGB',(128*7,128),(80,80,80));ctx=Image.new('RGB',(256*7,256),(80,80,80))
  for obs in t['observations']:
   x,y=np.rint(obs['uv']).astype(int);c=frames.index(obs['frame'])
   strip.paste(ims[obs['frame']].crop((x-64,y-64,x+64,y+64)),(c*128,0));ctx.paste(ims[obs['frame']].crop((x-128,y-128,x+128,y+128)),(c*256,0))
  file=f'review_{n:02d}_family_{t["family"]}.png';context=f'context_{n:02d}_family_{t["family"]}.png';strip.save(D/file);ctx.save(D/context)
  rows.append(dict(**t,row=n,file=file,context_file=context))
  if n%6==0:page=Image.new('RGB',(896,792),(235,235,235))
  page.paste(strip,(0,(n%6)*132))
  if n%6==5 or n==len(p['rows'])-1:file=f'review_page_{n//6:02d}.png';page.save(D/file);pages.append(file)
  if n%4==0:cp=Image.new('RGB',(1792,1040),(235,235,235))
  cp.paste(ctx,(0,(n%4)*260))
  if n%4==3 or n==len(p['rows'])-1:file=f'context_page_{n//4:02d}.png';cp.save(D/file);contexts.append(file)
 save(D/'visual-index.json',dict(rows=rows,pages=pages,context_pages=contexts,columns=frames,patch=128,context=256,source_pixels_unchanged=True,annotations_in_images=False,black_padding_outside_source=True))
 for path,h in p['input_hashes'].items():assert sha(Path(path))==h
 save(D/'result.json',dict(state='complete_pending_actual_review',stage='N12f',elapsed_seconds=time.monotonic()-start,rows=31,split_counts=dict(Counter(r['split'] for r in rows)),allexisting_roof_representatives_included=True,visual_index_sha256=sha(D/'visual-index.json'),source_hashes=p['input_hashes'],new_features=False,new_matches=False,new_camera=False,training=False))
 print('N12F_READY',len(rows))
if __name__=='__main__':globals()[sys.argv[1]]()
