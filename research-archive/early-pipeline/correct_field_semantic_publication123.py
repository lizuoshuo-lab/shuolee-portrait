from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,html,re
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
OUT=REC/'field-preflight/semantic-face-binding';G=R/'video10-global-initializer'
SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
v=read(OUT/'visual-review.json');r=read(OUT/'review-result.json');status={'observed':'可见部位','occluded':'遮挡','unknown':'未知'}
table=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [q['source_frame'],q['semantic_label'],q['ibug'],q['anchor']['smplx_name'],', '.join(f'{x:.4f}' for x in q['fan_uv_continuous']),status[q['anatomical_visibility']],q['anchor']['triangle_index'],', '.join(map(str,q['anchor']['vertex_indices'])),', '.join(f'{x:.6f}' for x in q['anchor']['barycentric'])])+'</tr>' for q in v['points'])
p=G/'index.html';s=p.read_text(encoding='utf-8')
pattern=r'(<details><summary>展开40个实拍二维点及SMPL-X三角面绑定</summary>.*?<tbody>).*?(</tbody></table></details>)'
s,n=re.subn(pattern,lambda m:m.group(1)+table+m.group(2),s,count=1,flags=re.S);assert n==1
assert s.count('其中5点的取整偏移至少1px')==1;s=s.replace('其中5点的取整偏移至少1px','其中8点的取整偏移至少1px')
note='<p>侧脸复核更正：朝画面右方时可见人物右眼、右嘴角，朝画面左方时可见人物左眼、左嘴角。初次人工审阅把可见侧写反，已更正状态表并保留原记录；检测坐标、SMPL-X编号和三维绑定均未修改。</p>'
s=s.replace('<h3>实拍输入检查发现的误差来源</h3>',note+'<h3>实拍输入检查发现的误差来源</h3>',1);p.write_text(s,encoding='utf-8')
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG/'相机多解与人物对齐']:
    if dest!=G:shutil.copy2(G/'index.html',dest/'index.html')
    for name in ['review-result.json','visual-review.json','real-2d-smplx-bindings.json','profile-side-correction.json']:
        shutil.copy2(OUT/name,dest/'semantic-face-binding'/name)
p=OUT/'slide-review.json';q=read(p);q['slide']['notes']=[line.replace('部位可见的24点中5点','部位可见的24点中8点') for line in q['slide']['notes']]
q['slide']['notes'].append('本轮人工侧脸可见侧初次写反，已按人物自身左右更正。没有更改任何检测坐标或模型索引。原记录和修正证据均保留。');save(p,q)
p=OUT/'publication.json';q=read(p);q.update(updated=datetime.now(timezone.utc).isoformat(),profile_side_annotation_corrected=True,corrected_observed_ge1px=8,diagnostic_DOM_verified=False);save(p,q)
p=REC/'evidence-ledger.json';q=read(p);entry=next(e for e in q['entries'] if e.get('id')=='field_semantic_2d_smplx_5view');entry.update(profile_review_correction=str(OUT/'profile-side-correction.json'),observed_rounding_ge1=8);save(p,q)
print('Corrected anatomical-side annotations and dependent report statistics; detector coordinates unchanged.')
