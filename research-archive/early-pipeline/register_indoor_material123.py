"""Register the first indoor177 native material inventory, before execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, difflib

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'; PRE=REC/'indoor-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (PRE/'native-material').exists() and not (PRE/'native-material-plan.json').exists()
assert read(PRE/'semantic-face-all/review-result.json')['state']=='complete_native_visibility_review'
dec=read(PRE/'decoded.json'); assert len(dec['records'])==177
old=H/'field_native_material123.py'; new=H/'indoor_native_material123.py'
assert not new.exists()
s=old.read_text(encoding='utf-8-sig')
for a,b in {"PRE=REC/'field-preflight'":"PRE=REC/'indoor-preflight'",
            "preflight/field')":"preflight/indoor')",
            '/tmp/shuolee-field-material123.lock':'/tmp/shuolee-indoor-material123.lock'}.items():
    assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
compile(s,str(new),'exec');new.write_text(s,encoding='utf-8')
plan=read(REC/'field-preflight/native-material-plan.json');stamp=datetime.now(timezone.utc).isoformat()
plan.update(registered=stamp,state='registered_not_started',subject='indoor',
    source='D:/00/工作/数据集/video8.mp4',source_sha256=dec['video_sha256'],
    decoded_manifest=str(PRE/'decoded.json'),decoded_manifest_sha256=sha(PRE/'decoded.json'),
    ROI_observations_sha256=sha(PRE/'coverage/observations.json'),
    hypothesis='More densely sampled native images may retain rigid-scene material links absent from the historical24frame graph. Establish actual material support separately from face semantics before geometric solving.',
    changed_information='First123 indoor177 inventory,173 source indices absent from the121 subset; consistent current FFmpeg RGB. No old feature coordinates, optimized3D or camera initialization reused.',
    all_predeclared_source_indices=[r['source_frame_index'] for r in dec['records']],
    prerequisite_review=dict(path=str(PRE/'semantic-face-all/review-result.json'),sha256=sha(PRE/'semantic-face-all/review-result.json')),
    preceding_source_audit=dict(path=str(PRE/'decoder-comparison/result.json'),sha256=sha(PRE/'decoder-comparison/result.json')),
    implementation=dict(path=str(new),sha256=sha(new),reference_path=str(old),reference_sha256=sha(old),changes=['Subject input/output and lock paths only. Feature budgets, descriptor gates, pair graph and family split unchanged.']),
    stop='One1200s CPU pass. Preserve all177 native frames. Stop on hash, identity, missing input or timeout. No retry, K update, BA, SMPLX fitting or training in this stage.')
plan['method']['identity_status']='Appearance candidates only. Building joints, wall fixtures, shelving and floor details require original-image physical-identity review. Reflections, chandelier glints, curtains, cat, plants and furniture silhouettes are not automatically static surface points.'
plan['review']['scope']='First30 predeclared queries, actual original crops; sampled HELD families development only and never FIT. All observations of any proposed static FIT family must be reviewed before geometric admission. No geometric residual selection.'
save(PRE/'native-material-plan.json',plan)
print(''.join(difflib.unified_diff(old.read_text(encoding='utf-8-sig').splitlines(True),s.splitlines(True),fromfile=old.name,tofile=new.name)))
print(json.dumps(dict(plan=str(PRE/'native-material-plan.json'),frames=177,pairs=854,CPU_only=True),ensure_ascii=False))
