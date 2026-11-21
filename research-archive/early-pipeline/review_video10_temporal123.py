"""Record actual review of96rows/all observations and8native context sheets."""
from pathlib import Path
from datetime import datetime,timezone
import json,itertools,hashlib,collections
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';O=R/'video10-temporal-bridge';B=R/'video10-static-material'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'visual-review.json').exists()
v=read(O/'run/visual-index.json');res=read(O/'run/result.json');assert len(v['rows'])==96 and len(v['pages'])==9
notes={
0:'扩大上下文后可辨认同一顶棚木节；未认证点位精度。',1:'灯附近木条和光斑，难以确定唯一材料点。',2:'光照溢出与相似木纹，保留身份不确定。',3:'金属锅表面高光和轮廓，不能当固定材料点。',4:'冰箱上设备面板与相邻白色物体形成可辨认固定局部。',5:'远处货架物件模糊，无法唯一指认中心材料点。',6:'继承旧族1的灯光/木条歧义，增加中间帧没有解除歧义。',7:'继承旧族5的高光/木条歧义。',8:'灯光附近亮边，不能作为固定表面对应。',9:'灯光亮边与木纹混合，不是已确认材料点。',10:'靠近灯光的裁片且部分超出图边，身份未定。',
11:'扩大原生上下文可确认同一白色顶梁端部与木条交界。',12:'扩大上下文可确认同一水烟器托盘与支架接合处；金属反光影响定位仍需检验。',13:'锅底和高光随视角变化，只有物体身份，未确认固定材料点。',14:'货架喷瓶的同一印刷图案可辨认。',15:'水烟管弯曲轮廓、反光和植物遮挡，点位身份不确定。',16:'金属锅的光滑轮廓，不能把轮廓点当固定表面点。',17:'扩大上下文可确认海报左侧虹膜印刷纹理，反光影响精度尚未测量。',18:'桌上布料折痕模糊，无法确认中心同一点。',19:'挂钥匙处的同一黑色挂件局部可辨认。',20:'墙上小画框与内部图案局部可辨认。',21:'台面边缘和盆栽遮挡混合，物理点不确定。',
22:'蘑菇饰物的光滑轮廓和阴影，不能确认中心固定材料点。',23:'扩大上下文可确认镜子外木框局部；镜中反射不算该点证据。',24:'重复鞋底与鞋架，未能唯一识别中心材料点。',25:'黑色物件和悬挂衣物交界，身份未定。',26:'黑色物体上的移动亮点，不能按固定材质点使用。',27:'同一黑色钥匙挂件的局部形状可辨认。',28:'头盔亮面与鞋架交界，反光点不确定。',29:'钥匙挂件的另一局部可辨认，但独立物理身份仍需细化。',30:'袋内白色小物件与包装边角，粗局部可辨认。',31:'白色小物件光滑边缘，点沿边定位尚不明确。',32:'袋中青色包装的折角可辨认。',
33:'盆栽条纹重复且边界会随视角变化，无法确认唯一点。',34:'扩大上下文可确认同一木抽屉上把手左端。',35:'地上浅绿拖鞋及红色标记附近局部可辨认。',36:'浅绿拖鞋上同一局部可辨认；不计非地面刚性支持。',37:'继承旧族52的鞋架/鞋纹歧义，没有改为可信点。',38:'继承旧族49的拖鞋轮廓歧义。',39:'拱门底部踢脚线与墙角接合局部可辨认。',40:'浅绿拖鞋上表面局部可辨认；非地面刚性支持不计。',41:'鞋子边缘与地缝混合，中心点不明确。',42:'继承旧族50的拖鞋/地缝歧义。',
43:'白色小设备与黑色面板交界局部可辨认。',44:'黑绿设备面板上印刷区域可辨认。',45:'扩大上下文仍是飞镖盘重复扇区/环线；没有足够证据认证唯一材料点，也不凭视觉猜成明确错接。',46:'继承旧族3的顶棚灯光歧义。',47:'继承旧族2的顶棚灯光歧义。',48:'顶棚灯光溢出和木条，身份未定。',49:'挂衣架的同一支架局部可辨认。',50:'水烟器顶端金属高光，不作为固定材料点。',51:'扩大上下文可确认海报右侧两枚虹膜之间同一印刷纹理区域。',52:'钥匙挂件局部可辨认，精确点位未认证。',53:'相邻钥匙挂件局部可辨认，仍需验证物理点独立性。',54:'桌布和餐具附近模糊折痕，中心材料身份未定。',
55:'鞋架上重复鞋底花纹，局部身份未定。',56:'黑色物体与鞋架的光滑轮廓，不能确认同一表面点。',57:'重复鞋纹，未确认唯一材料点。',58:'袋内青色包装折角局部可辨认。',59:'柜底与地面接合角落可辨认。',60:'白鞋的光滑轮廓和阴影，点位未定。',61:'继承旧族55可辨认踢脚线角落；仍未认证亚像素。',62:'白鞋光滑轮廓，不能直接当固定材料点。',63:'浅绿拖鞋边缘及地缝，中心身份未定。',
64:'头部框实际落在前颈/胸口与衣领，不能算眼鼻嘴对应。',65:'耳部和镜腿附近同一粗头面局部连续可辨认。',66:'另一侧耳部粗局部连续可辨认。',67:'眼镜外侧/颞部附近粗局部可辨认，不代表已确认同一眼部皮肤点。',68:'后颈发际线区域，不是眼鼻嘴面部覆盖。',69:'耳部/镜腿附近粗局部可辨认。',70:'后衣领印花与条纹，不是面部点。',71:'下唇附近胡须局部可辨认，继承旧族62；尚未认证物理单点。',72:'耳部粗局部HELD可辨认；与FIT同一耳区的独立性尚未证实。',73:'耳部/镜腿粗局部HELD可辨认。',74:'下唇胡须附近HELD粗局部可辨认；与FIT邻近区域独立性尚未认证。',75:'后衣领/发际交界，不算面部点。',76:'脸颊与口鼻附近粗局部可辨认，未达到逐点物理身份和精度验收。',77:'衣领区域，不能算面部。',78:'眼镜/鼻侧附近粗局部可辨认，未认证唯一物理点。',79:'衣领区域，不能算面部。',
80:'胸前口袋装饰局部可辨认；衣物形变另行检验。',81:'前臂字母纹身局部可辨认，不能直接当静止背景。',82:'胸前口袋印花局部可辨认。',83:'胸前口袋装饰边缘可辨认。',84:'前门襟/衣领纹理局部可辨认。',85:'前臂图案纹身局部可辨认。',86:'字母纹身局部可辨认。',87:'手腕附近纹身局部可辨认。',88:'字母纹身靠近手镯的局部可辨认。',89:'前门襟和衣领图案局部可辨认。',90:'前臂图案纹身局部可辨认。',91:'前门襟花纹局部可辨认。',92:'握合手部指节与皱褶局部可辨认；手指微动没有检验。',93:'字母纹身局部可辨认。',94:'袖口接缝/褶皱可辨认；不等于稳定皮肤点。',95:'袖口接缝局部可辨认，褶皱与形变未检验。'}
fixed={0,4,11,12,14,17,19,20,23,27,29,30,32,34,35,36,39,40,43,44,49,51,52,53,58,59,61}
ground={35,36,39,40,59,61}
face={65,66,67,69,71,72,73,74,76,78};skin={81,85,86,87,88,90,92,93};nonface={64,68}
rows=[]
for t in v['rows']:
 i=t['row'];verdict='fixed_patch_candidate' if i in fixed else 'head_face_patch_candidate' if i in face else 'skin_patch_candidate' if i in skin else 'nonface_headROI' if i in nonface else 'clothing_patch' if i>=64 else 'ambiguous'
 rows.append(dict(row=i,family=t['family'],track=t['track'],split=t['split'],original_category=t['category'],frames=t['cell_frames'],spatial_bin=t['spatial_bin'],verdict=verdict,note=notes[i],coarse_fixed_patch=i in fixed,nonground_rigid_candidate=i in fixed-ground,coarse_head_face_patch=i in face,physical_subpixel_verified=False,pointwise_independence_verified=False,inherited_reviews=t['inherited_reviews']))
assert len(rows)==len(notes)==96
fit=[r for r in rows if r['coarse_fixed_patch'] and r['split']=='fit'];held=[r for r in rows if r['coarse_fixed_patch'] and r['split']=='heldout']
counts={s:sum(r['verdict']=='head_face_patch_candidate' and r['split']==s for r in rows) for s in ['fit','heldout']}
byframe={str(f):{s:sum(r['coarse_fixed_patch'] and r['split']==s and f in r['frames'] for r in rows) for s in ['fit','heldout']} for f in read(O/'plan.json')['frames']}
pairs=[dict(frames=[a,b],fit=sum(a in r['frames'] and b in r['frames'] for r in fit)) for a,b in itertools.combinations(read(O/'plan.json')['frames'],2)]
best=sorted(pairs,key=lambda p:-p['fit'])[0]
support=dict(coarse_fixed_fit=len(fit),coarse_fixed_heldout=len(held),nonground_rigid_fit=sum(r['nonground_rigid_candidate'] for r in fit),nonground_rigid_heldout=sum(r['nonground_rigid_candidate'] for r in held),max_pair_shared_fit=best,by_frame=byframe,frames_without_reviewed_fixed_fit=[int(f) for f,x in byframe.items() if x['fit']==0],coarse_head_face=counts,independent_subpixel_face_holdout=0,gate_passed=False,required_coarse_fit=20,required_coarse_heldout=8,required_initial_pair_shared_fit=12)
assert len(fit)<20 or best['fit']<12
summary=f'视频10补入24帧后，33帧诊断产生1732条轨迹、1191个近邻族，3族因FIT/HELD相连被隔离。实际审阅96族全部观测和8组扩大上下文，固定背景粗候选{len(fit)}FIT/{len(held)}HELD，最大帧对共享FIT{best["fit"]}；未通过相机支持门槛。16个头部框代表中10个可辨认头面粗局部（FIT5/HELD5，含耳部），仍没有认证的独立亚像素面部点。未改变K、求新相机或启动SMPL-X/高斯训练。'
stamp=datetime.now(timezone.utc).isoformat()
save(O/'visual-review.json',dict(updated=stamp,summary=summary,actual_saved_visual_review=True,reviewed_rows=96,reviewed_pages=9,reviewed_context_rows=[0,11,12,17,23,34,45,51],rows=rows,support=support,
 material_limits=['Coarse patch recognizability is not a certified identical physical subpixel point.','Ten head/face patches include ears and are not ten independent eye/nose/mouth landmarks.','Manual review samples selected representatives, not a random1191family error-rate estimate.','The original6 contradicted families remain excluded, not repaired.','Visible blur, reflections, repeated textures and cross-split merging remain explicit.'],
 causal_boundary=res['causal_limit'],new_geometry_admitted=False,new_training_admitted=False,conclusion='Temporal additions improve diagnostic track coverage but have not established camera or SMPL-X alignment. This is not a unique explanation of prior grayT misalignment.'))
save(O/'support-result.json',support)
save(O/'result-reviewed.json',dict(updated=stamp,extraction_result=str(O/'run/result.json'),visual_review=str(O/'visual-review.json'),summary=summary,support=support,quality_accepted=False,cameras_estimated=False,training_started=False,bounded_feature_pass_complete=True))
save(O/'progress.json',dict(updated=stamp,state='complete_reviewed_camera_support_failed',pid=None,result='result-reviewed.json',new_feature_passes=1,new_feature_frames=24,new_cameras=0,training_started=False))
for name,h in read(O/'plan.json')['cached_npz_sha256'].items():assert sha(B/'run'/name)==h
print(json.dumps(dict(summary=summary,support=support),ensure_ascii=False))
