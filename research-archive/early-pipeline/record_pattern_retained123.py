"""Record actual saved-pixel review and trace three observed contradictions.

This reads immutable accepted pair files. No extraction, matching, point editing,
camera fitting or training is performed. The source observations stay intact.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, time
import numpy as np

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/pattern-preflight/native-material'
B=O/'bridge-control'; V=B/'physical-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()

assert not (V/'review.json').exists(), 'Review already recorded; do not overwrite.'
start=time.monotonic(); index=read(V/'index.json'); rows=index['rows']
plan=read(B/'physical-review-plan.json')
assert sha(B/'segments.json')==plan['input_sha256']
assert sha(B/'head-family-catalog.json')==plan['catalog_sha256']
chosen=[9,12,40]
save(V/'contradiction-trace-plan.json',dict(registered=now(),posthoc=True,
    purpose='Trace immutable graph edges of three actual visually contradictory FIT segments; not a fresh accuracy test.',
    rows=chosen,review_index_sha256=sha(V/'index.json'),pair_manifest_sha256=sha(O/'pair-hashes.json'),
    budget=dict(cpu_seconds=120,new_matching=0,new_features=0,new_geometry=0),
    no_parameter_or_membership_changes=True))
manifest=read(O/'feature-manifest.json')['frames']; pairs=read(O/'pair-hashes.json')['records']
graphs={r:{tuple(n):set() for n in rows[r]['nodes']} for r in chosen}; traces={r:[] for r in chosen}
for item in pairs:
    a,b=map(int,Path(item['file']).stem.split('_'))
    relevant=[r for r in chosen if a in {n[0] for n in graphs[r]} and b in {n[0] for n in graphs[r]}]
    if not relevant:continue
    assert time.monotonic()-start<120
    p=O/'pairs'/item['file'];assert sha(p)==item['sha256']
    with np.load(p,allow_pickle=False) as z:
        accepted={tuple(x) for x in z['accepted_pairs'].tolist()}
        for r in relevant:
            na=next(n for n in graphs[r] if n[0]==a); nb=next(n for n in graphs[r] if n[0]==b)
            if (na[1],nb[1]) not in accepted:continue
            f=z['forward_distances'][na[1]]; rev=z['reverse_distances'][nb[1]]
            assert z['forward_indices'][na[1],0]==nb[1] and z['reverse_indices'][nb[1],0]==na[1]
            assert f[0]<.75*f[1] and rev[0]<.75*rev[1]
            graphs[r][na].add(nb);graphs[r][nb].add(na)
            traces[r].append(dict(a=na,b=nb,source_frames=[manifest[a]['source_frame'],manifest[b]['source_frame']],
                forward_ratio=float(f[0]/f[1]),reverse_ratio=float(rev[0]/rev[1]),pair_sha256=item['sha256']))
results=[]
for r in chosen:
    adj=graphs[r]
    for e in traces[r]:
        a,b=tuple(e['a']),tuple(e['b']);e['triangles']=len(adj[a]&adj[b])
        seen={a};stack=[a]
        while stack:
            u=stack.pop()
            for v in adj[u]:
                if {u,v}=={a,b}:continue
                if v not in seen:seen.add(v);stack.append(v)
        e['bridge']=b not in seen
        assert not e['bridge']
    results.append(dict(row=r,segment_id=rows[r]['segment_id'],family_id=rows[r]['family_id'],split=rows[r]['split'],
        nodes=rows[r]['nodes'],edges=traces[r],all_edges_non_bridge=True,
        triangle_count=sum(e['triangles'] for e in traces[r])//3,
        used_for_camera_or_SMPLX=False))
assert all(len(traces[r])==3 and all(e['triangles']==1 for e in traces[r]) for r in [9,12])
save(V/'contradiction-trace.json',dict(updated=now(),posthoc=True,records=results,source_observations_unchanged=True,
    FIT_HELD_unchanged=True,thresholds_unchanged=True,new_geometry=False,seconds=time.monotonic()-start,
    conclusion='Two complete three-view triangles link incompatible objects. Cycle consistency alone is insufficient for physical identity.'))

false_notes={
9:'查询分别落在脸部、椅子金属连接处和玻璃门把手附近，不是同一物理点。三个节点的全部三条边形成完整三角形。',
12:'前两幅查询位于侧脸轮廓附近，第三幅落在墙面/门框区域。三个节点形成完整三角形，物理身份矛盾。',
40:'第一幅查询在上衣图案，另两幅在嘴部/下脸区域；已看的跨ROI端点也在脸部。该20节点段存在至少一次身份矛盾；未声称逐个查看全部20节点。'}
background_notes={
172:'叶片与树丛；静止性和同一叶片身份未确认。',173:'桌面手机；屏幕、反光及外壳边界的精确身份未知。',
174:'叶片后的深色竖杆；遮挡及轮廓身份未知。',175:'树丛叶片；静止性及身份未知。',
176:'折叠椅金属连接件，展示视角内物体级一致；精确点和静止性未认证。',177:'植物茎叶及前景杆件遮挡，身份未知。',
178:'反光花盆与叶片；不能作为已认证固定材料。',179:'重复地毯纹理，具体纤维身份未知。',
180:'桌面排插/接头，物体级一致；精确点及静止性未认证。',181:'椅子连接件与百叶窗；物体级一致，重复结构风险。',
182:'椅架与手臂遮挡；查询可见性及身份未知。',183:'玻璃/百叶窗上的明亮条纹或反射，未认证固定材料。',
184:'办公椅网布，重复纹理，精确材料身份未知。',185:'椅架及人物局部遮挡，身份未知。',
186:'深色玻璃区域或其后场景，反射与真实材料未分清。',187:'办公椅网布，重复纹理及视角变化。',
188:'亮面黑色桌沿/台面，反射和轮廓身份未知。',189:'地面门挡/墙脚附近，物体级一致；精确点和静止性未认证。',
190:'植物/花盆及袖子遮挡，身份未知。',191:'重复百叶窗结构，身份未知。',
192:'植物后的竖杆，遮挡/边界身份未知。',193:'玻璃磨砂字样；重复字形与反射，具体材料身份未认证。',
194:'手臂附近的黑色台面/窗台；查询仍可能在台面，不标作明确误连，反光及遮挡身份未知。',
195:'植物叶片，身份及静止性未知。',196:'桌面手机；屏幕/壳体边界精确身份未知。',
197:'折叠椅金属连接件，展示视角物体级一致；精确点与静止性未认证。',198:'重复百叶窗结构，身份未知。',
199:'玻璃门角及反射，身份未知。',200:'玻璃门把手，固定物体可能但精确身份/反射未认证。',
201:'椅子金属连接件，物体级一致；精确点与静止性未认证。',202:'玻璃门把手接头，物体级一致；精确点与反射未认证。',
203:'百叶窗拉线/亮条，重复与反射风险。',204:'墙脚门挡附近，第一幅旁有袖子；未见足够证据判跨材料误连。',
205:'重复地毯纹理，身份未知。',206:'椅脚连接件，物体级一致；精确点与静止性未认证。',
207:'地毯重复纹理且裁片包含原图外补黑，身份未知。',208:'玻璃门地面铰接件，物体级一致；精确点与静止性未认证。',
209:'重复地毯纹理，身份未知。',210:'折叠椅座/金属钩，物体级可能一致，精确身份未认证。',
211:'办公椅扶手轮廓，轮廓点随视角变化，未认证固定材料。'}
annotations=[]
for row in rows:
    i=row['row'];note='展示裁片中可见相似脸部/头部区域；相似外观不能证明中心查询是同一材料点。精确身份、刚性及亚像素定位仍未知。'
    cls='unresolved_physical_identity'
    if i in false_notes:note=false_notes[i];cls='confirmed_physical_identity_contradiction'
    elif i in background_notes:note=background_notes[i]
    elif i==54:note='上部头发到胸前头发/衣服区域，重复发丝身份不清。大位移本身不是误连判据，保留未知。'
    elif i==34:note='头发/颈部边界，原生裁片中未确认同一具体发丝，不标作明确矛盾。'
    annotations.append(dict(row=i,category=row['category'],family_id=row['family_id'],segment_id=row['segment_id'],split=row['split'],
        development_only=row['development_only'],classification=cls,note=note,
        actual_saved_views_reviewed=[dict(slot=v['slot'],source_frame=v['source_frame'],keypoint=v['keypoint'],uv=v['uv']) for v in row['views']+row['extra_views']],
        reviewed_nodes_count=len(row['views'])+len(row['extra_views']),total_nodes=len(row['nodes']),
        all_nodes_actually_reviewed=len(row['views'])+len(row['extra_views'])==len(row['nodes']),
        exact_material_accepted=False,staticity_verified=False,localization_uncertainty_px=None))
counts={s:dict(rows=sum(x['split']==s for x in rows),head_rows=sum(x['split']==s and x['category']=='head' for x in rows),
    families=len({x['family_id'] for x in rows if x['split']==s})) for s in ['FIT','HELD']}
review=dict(updated=now(),actual_saved_pixels_reviewed=True,all22_contact_sheets_reviewed=True,
    all38_ROI_crossing_views_reviewed=True,reviewed_native_views=212*3+38,
    individual_reinspection_rows=[9,12,40,54,34,194],
    head_families_reviewed=160,head_families_not_reviewed=33,head_segments=172,background_families=40,
    rows=annotations,counts=counts,confirmed_contradictions=chosen,
    confirmed_contradictions_FIT=3,confirmed_contradictions_HELD=0,
    no_new_accuracy_estimate=True,physical_face_HELD_accepted=0,
    no_review_based_pixel_or_split_edits=True,ROI_label_is_not_semantic_truth=True,
    exact_static_background_accepted=0,
    limitations=['First/middle/last plus ROI-crossing endpoints were reviewed, not all long-track observations.',
        '33 head families are explicitly unreviewed. These samples do not prove all background support is absent.',
        'No camera was fitted in this subject. RANSAC could reject candidate outliers; these examples alone do not prove a final camera or old SMPL-X cause.',
        'Coarse visual review cannot measure subpixel localization uncertainty or certify rigid motion.'],
    next='当前一次材料支持预算已结束，不启动该人物相机或SMPL-X。保留错误FIT段和未知材料；下一步只核验已训练SMPL-X实际消费的观测与权重，不重新提点、调阈值或补回缺口。')
save(V/'review.json',review)
save(V/'progress.json',dict(updated=now(),stage='actual_review_complete_support_not_admitted',pid=None,training=False))
print(json.dumps(dict(reviewed_rows=len(rows),native_views=674,counts=counts,
    traced=[dict(row=x['row'],edges=len(x['edges']),triangles=x['triangle_count']) for x in results],
    confirmed_FIT_contradictions=3,training=False),ensure_ascii=False))
