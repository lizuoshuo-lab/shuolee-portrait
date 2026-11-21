"""Native crops of the preregistered72 queries; no point re-selection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,html
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';Q=R/'video10-correspondence-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(Q/'plan.json');result=read(Q/'run/result.json');allocation=read(Q/'review-allocation.json')
assert result['state']=='complete_pending_physical_review' and sha(Q/'review-allocation.json')==p['review_allocation_sha256']
D=Q/'review';D.mkdir();paths=[Path(s) for s in p['input_sha256']]
images={f:Image.open(next(x for x in paths if x.name==f'native_frame_{f:06d}.png')).convert('RGB') for f in p['frames']}
def crop(f,uv,size=256):
    if uv is None or not np.isfinite(uv).all():return Image.new('RGB',(size,size),(226,226,226))
    x,y=np.rint(uv).astype(int);return images[f].crop((x-size//2,y-size//2,x+size//2,y+size//2))
rows=[];pages=[];page=None
for q in allocation['rows']:
    i=q['row'];a=q['source'];b=q['target'];z=q['query_feature']
    datafile=Q/'run'/f'queries_{a:06d}_{b:06d}.npz'
    with np.load(datafile) as d:
        assert np.array_equal(d['query_uv'][z],np.array(q['source_uv'],np.float32))
        base=d['baseline_uv'][z];roma=d['uv'][z]
        row=dict(**q,baseline_uv=base.tolist() if np.isfinite(base).all() else None,roma_uv=roma.tolist(),
            confidence=float(d['confidence'][z]),fb_native_px=float(d['fb'][z]),gate_pass=bool(d['eligible'][z]),
            split_conflict=bool(d['split_conflict'][z]),method_displacement_px=float(d['method_displacement_px'][z]) if np.isfinite(d['method_displacement_px'][z]) else None,
            source_sha256=p['input_sha256'][str(next(x for x in paths if x.name==f'native_frame_{a:06d}.png'))],prediction_sha256=sha(datafile))
    strip=Image.new('RGB',(768,256),(226,226,226))
    for j,(f,uv) in enumerate([(a,q['source_uv']),(b,row['baseline_uv']),(b,roma)]):strip.paste(crop(f,uv),(256*j,0))
    name=f'row_{i:03d}_{a:06d}_{b:06d}.png';strip.save(D/name);row['file']=name;rows.append(row)
    if i%6==0:page=Image.new('RGB',(768,6*260),(245,245,245));page_rows=[]
    page.paste(strip,(0,(i%6)*260));page_rows.append(i)
    if i%6==5 or i==len(allocation['rows'])-1:
        name=f'page_{len(pages):02d}.png';page.crop((0,0,768,(i%6)*260+256)).save(D/name)
        pages.append(dict(file=name,rows=page_rows.copy()))
meta=dict(updated=datetime.now(timezone.utc).isoformat(),columns=['source_native_query','frozenRootSIFT_target_or_blank','RoMa_target'],
    native_crop_px=256,point_at_rounded_crop_center=[128,128],no_scaling_or_warp=True,burned_text=False,
    rounding_note='Nativecropcenter is rounded for display only; floating queries unchanged. These crops cannot certify subpixelerror.',rows=rows,pages=pages)
save(D/'index.json',meta)
sections=[]
for pg in pages:
    lines=[]
    for i in pg['rows']:
        r=rows[i];lines.append(f'<tr><td>{i}</td><td>{r["source"]}→{r["target"]}</td><td>{r["family"]}/{r["split"]}</td><td>{html.escape(r["review_group"])}</td><td>{r["confidence"]:.3f}</td><td>{r["fb_native_px"]:.3f}</td><td>{r["gate_pass"]}</td><td>{r["split_conflict"]}</td></tr>')
    sections.append('<section><h2>行'+','.join(map(str,pg['rows']))+'</h2><img src="'+pg['file']+'"><table><tr><th>行</th><th>帧</th><th>原族/划分</th><th>原区域</th><th>置信度</th><th>往返px</th><th>筛查通过</th><th>跨划分邻近</th></tr>'+''.join(lines)+'</table></section>')
(D/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:17px sans-serif;max-width:1000px;margin:30px auto}img{max-width:100%}td,th{padding:6px;border:1px solid #bbb}section{margin:32px 0}</style><h1>同一原生查询点的匹配方法对照</h1><p>每行从左到右：原图查询、原RootSIFT对应（没有则空白）、RoMa预测。每格256原生像素，未缩放、未变形；浮点查询保留，裁片显示中心取整。高置信度、双向/循环一致不等于同一物理点，也不是相机误差。</p>'+''.join(sections),encoding='utf-8')
print(json.dumps(dict(rows=len(rows),pages=len(pages),gate_pass=sum(x['gate_pass'] for x in rows),split_conflicts=sum(x['split_conflict'] for x in rows)),ensure_ascii=False))
