"""Record actual review of113 frozen native families and16 expanded contexts."""
import collections,itertools,json
import match_video10_temporal_bridge123 as bridge
b=bridge.base; N=b.R/'video10-semantic-role'; O=bridge.O
assert not (N/'material-visual-review.json').exists()
idx=b.read(N/'material-review/index.json'); plan=b.read(N/'material-review-plan.json')
assert len(idx['rows'])==113 and len(idx['pages'])==11
notes={
0:'鞋架与鞋底重复纹理，中心材料身份仍不唯一。',
1:'木镜框上角与邻近红色物体可辨认；反射不作为材料依据。',
2:'钥匙和木框交界模糊，精确材料点未确认。',
3:'扩大上下文确认同一扇门上方空调机的局部；不是不同抽屉把手。',
4:'挂衣架上端红色物体旁的固定支架局部一致。',
5:'同一空调机前板局部可辨认，定位精度未认证。',
6:'悬挂衣物与黑色间隙，不能直接作为刚性背景。',
7:'重复鞋架/鞋纹且模糊，身份不确定。',
8:'冰箱磁贴图案与邻近文字可辨认；第107帧模糊不等于亚像素有效。',
9:'冰箱上方一组磁贴图案可辨认；第107帧严重模糊，精度另验。',
10:'墙上虹膜海报的同一蓝色圆形印刷局部可辨认。',
11:'顶棚重复木条和边界，缺唯一局部身份。',
12:'水烟器横向手柄与托架接合的局部可辨认；不把金属亮点当精确点。',
13:'扩大上下文确认雕花镜子的木框上缘，镜中变化影像不计。',
14:'扩大上下文确认木柜下部边角与地面接合的同一区域。',
15:'悬挂金属锅的高光与轮廓，未确认材料点。',
16:'扩大上下文后，中心相对黑地垫角和瓷砖缝的位置发生明显变化；保留定位/重复纹理歧义，不纳入固定点。',
17:'重复顶棚木条沿边，不能唯一识别。',
18:'同一黑色门口左上角及外框可辨认。',
19:'门框直边低纹理，沿边点位不确定。',
20:'扩大上下文确认顶梁下同一线缆与固定点附近局部；非精确角点真值。',
21:'扩大上下文为水烟器顶盘/软管附近，金属反光与阴影混合，材料点未定。',
22:'黑色门口顶角与内沿可辨认。',
23:'水烟器横向手柄的末端局部可辨认；稳定性与精确点位仍待几何检验。',
24:'扩大上下文确认柜脚前方同一组地面接缝；不按手部或其影子作为对应。',
25:'水烟器横柄与托架接合的局部可辨认；反光精度另验。',
26:'水烟器弯曲管上端局部可辨认，只有粗物体局部身份。',
27:'椅脚/地毯与地缝附近局部可辨认，阴影不作为固定材料证据。',
28:'扩大上下文确认虹膜海报右侧相邻两圆间的同一印刷区域。',
29:'顶棚木纹重复，唯一材料点未确认。',
30:'柜脚前地缝局部可辨认；与同区域其他族的物理独立性未认证。',
31:'扩大上下文确认同一白色顶梁端部与墙/顶棚接合。',
32:'台面遥控器与桌面边角的局部可辨认；未检验物体是否微动。',
33:'吊灯饱和亮线和重复框架，不能确认为固定材料点。',
34:'壁灯木支架同一端部接合可辨认。',
35:'台面同一遥控器局部可辨认；与另一族不宣称物理独立。',
36:'扩大上下文为钟体、玻璃与金属立杆交界，反光/轮廓使中心材料点不确定。',
37:'植物叶缘和投影，不能默认刚性背景。',
38:'扩大上下文后仍为飞镖盘重复扇区/环线；没有认证具体同一材料点。',
39:'飞镖盘重复扇区与数字附近，中心材料身份不确定。',
40:'飞镖盘外黑色底座与白门外缘接合局部可辨认。',
41:'飞镖盘数字/重复环线附近，保留身份不确定。',
42:'扩大上下文确认冰箱上方实物置物架的小房屋饰物/包装，不是屏幕影像。',
43:'同一置物架上房屋形饰物的印刷局部可辨认。',
44:'飞镖盘左侧重复扇区且邻近人物遮挡，未认证唯一点。',
45:'飞镖盘相邻重复扇区和人物边缘，未认证唯一点。',
46:'飞镖盘右侧重复扇区，中心材料身份未定。',
47:'飞镖盘顶端重复环线，未认证唯一点。',
48:'瓶体边缘/高光，只有物体身份，没有稳定材料点。',
49:'冰箱顶白色小设备与黑绿面板接合局部可辨认。',
50:'冰箱上端特有磁贴及邻近边缘可辨认。',
51:'门把手金属高光和轮廓，材料点未认证。',
52:'数字显示屏内容由20:14到20:15变化；显示数字不能当固定场景纹理。',
53:'杯架上重复杯子且模糊，中心点不唯一。',
54:'黑板上同一白色字形局部可辨认。',
55:'悬挂锅的反光与轮廓，不纳入固定材料。',
56:'冰箱磁贴组合的同一印刷局部可辨认。',
57:'台面设备外壳与白色标签局部可辨认；反光定位另验。',
58:'白色设备与黑绿面板交界同一局部可辨认。',
59:'白纸边缘和低纹理，中心材料点未确认。',
60:'黑板同一白色文字局部可辨认。',
61:'锅底镜面/金属高光，不作为稳定材料对应。',
62:'鞋底与鞋架重复图案，身份未确认。',
63:'木柜侧板的挂钩与周围木纹局部可辨认。',
64:'挂衣架/木柜侧边固定端部局部可辨认。',
65:'水烟器金属反光与植物遮挡交叉，中心材料身份未定。',
66:'金属锅侧面随视角改变的高光，不纳入固定点。',
67:'木柜上部挂钩附近局部可辨认；不使用后方衣物。',
68:'扩大上下文仍为海报相似虹膜/边界；中心相对印刷图案有漂移，保留歧义而不猜为明确错接。',
69:'镜子上部雕花木框纹理可辨认，不用镜中反射。',
70:'扩大上下文确认木柜侧板同一木节，非跨抽屉连接。',
71:'钥匙悬挂交界模糊，固定材料点未确认。',
72:'顶梁下同一线缆接合区域可辨认，亚像素与刚性未认证。',
73:'黑地垫旁同一地缝/墙根局部可辨认。',
74:'水烟器横向手柄末端局部可辨认，未认证点位和独立性。',
75:'同一手柄另一粗局部可辨认，不能当新增独立真值。',
76:'饱和吊灯亮边和重复金属结构，中心材料身份未定。',
77:'顶梁与线缆附近同一局部可辨认。',
78:'壁灯木支架与墙面接合端部可辨认。',
79:'顶棚重复木纹/照明渐变，身份未确认。',
80:'木支架顶部与顶棚接合局部可辨认。',
81:'吊灯饱和亮框，不是已确认材料点。',
82:'重复木纹和照明渐变，唯一点未确认。',
83:'瓶盖高光与低纹理瓶体，材料点未确认。',
84:'同一门上空调机前部局部可辨认。',
85:'飞镖盘下部重复扇区及遮挡，中心材料身份未定。',
86:'瓶体反光/轮廓，没有可认证材料点。',
87:'金属锅反射和轮廓，不作为固定材料点。',
88:'飞镖盘重复扇区，未认证唯一材料点。',
89:'冰箱上缘白色磁贴/邻近纹理的局部可辨认。',
90:'金属锅轮廓随视角变化，不作为同一材料点。',
91:'冰箱上方磁贴与竹编/物架边缘局部可辨认。',
92:'冰箱特有磁贴组合的局部可辨认。',
93:'瓶体表面反光，材料点身份未确认。',
94:'瓶体/木柜边缘与低纹理混合，点位未定。',
95:'地上拖鞋轮廓与墙根相邻，未确认中心材料点。',
96:'扩大上下文确认壁架上同一小照片/卡片，不是动态屏幕。',
97:'冰箱上同一特色磁贴局部可辨认；第617帧不使用失败遮罩。',
98:'袋内软物/折角，未确认刚性材料点。',
99:'墙上小画框的固定印刷局部可辨认。',
100:'红色灯具与亮面边缘，中心材料身份未认证。',
101:'壁架上的同一小画框/卡片局部可辨认。',
102:'冰箱上同一彩色印刷磁贴可辨认。',
103:'台面木盘/雕刻物的特有粗局部可辨认。',
104:'黑色电视边缘与低纹理，材料点未认证。',
105:'冰箱磁贴与文字组合的同一局部可辨认。',
106:'植物叶片与反光背景交界，不作为刚性背景。',
107:'扩大上下文确认台面不规则木盘/雕刻边缘，同一实物局部可辨认。',
108:'顶棚相似木纹，唯一材料点未确认。',
109:'顶棚木条和照明渐变，唯一材料点未确认。',
110:'冰箱单枚彩色磁贴图案可辨认。',
111:'木盘/雕刻物同一不规则局部可辨认。',
112:'同一壁挂小画框及印刷局部可辨认。'}
fixed={1,3,4,5,8,9,10,12,13,14,18,20,22,23,24,25,26,27,28,30,31,32,34,35,40,42,43,49,50,54,56,57,58,60,63,64,67,69,70,72,73,74,75,77,78,80,84,89,91,92,96,97,99,101,102,103,105,107,110,111,112}
# Flexible hoses/cables, movable small items, and ground are not counted as nonground rigid support.
nonrigid_or_ground={12,20,23,24,25,26,27,30,32,35,72,73,74,75,77}
rows=[]
for t in idx['rows']:
    i=t['row']; assert t['family']==plan['rows'][i]['family'] and t['split']==plan['rows'][i]['split']
    rows.append(dict(row=i,family=t['family'],track=t['track'],split=t['split'],frames=t['frames'],
        allocation=t['review_reason'],verdict='dynamic_display_excluded' if i==52 else 'coarse_fixed_patch' if i in fixed else 'ambiguous',note=notes[i],
        coarse_fixed_patch=i in fixed,nonground_rigid_candidate=i in fixed-nonrigid_or_ground,
        physical_subpixel_verified=False,pointwise_independence_verified=False))
assert len(notes)==len(rows)==113
old=b.read(O/'visual-review.json')['rows']; families=b.read(N/'secondary-families.json')['rows']; lookup={x['family']:x for x in families}
# Keep each frozen family once. Whole-family existing contradictions/quarantines take priority.
combined={}
for source,rr in [('temporal_review',old),('semantic_review',rows)]:
    for r in rr:
        if not r['coarse_fixed_patch']:continue
        fa=lookup[r['family']]
        if fa['conflict'] or fa['inherited_contradiction'] or fa['split']=='quarantine':continue
        if r['family'] not in combined:
            combined[r['family']]=dict(**r,review_source=source)
        else:
            assert combined[r['family']]['split']==r['split']
# New rejection affects family-use only, never deletes raw data or moves split.
for r in rows:
    if r['verdict']=='dynamic_display_excluded':combined.pop(r['family'],None)
frames=b.read(O/'plan.json')['frames']; accepted=list(combined.values())
def count(rr):return {s:sum(x['split']==s for x in rr) for s in ['fit','heldout']}
coverage={str(f):count([r for r in accepted if f in r['frames']]) for f in frames}
pair_rows=[dict(frames=[a,c],fit=sum(r['split']=='fit' and a in r['frames'] and c in r['frames'] for r in accepted)) for a,c in itertools.combinations(frames,2)]
pair_rows.sort(key=lambda x:(-x['fit'],x['frames']))
total=count(accepted); nonground=count([r for r in accepted if r['nonground_rigid_candidate']])
support=dict(semantic_only=count([r for r in rows if r['coarse_fixed_patch']]),combined_coarse=total,combined_nonground=nonground,
    top_shared_pairs=pair_rows[:12],by_frame=coverage,frames_below8_fit=[int(f) for f,c in coverage.items() if c['fit']<8],
    coarse_count_gate=total['fit']>=20 and total['heldout']>=8 and nonground['fit']>=6 and nonground['heldout']>=3,
    initial_pair_count_gate=pair_rows[0]['fit']>=12,
    full_registered_camera_graph_admitted=False,face_subpixel_heldout=0,
    scope='Necessary counts of coarse recognizable patches only; not physically independent point count, triangulation success, multi-depth proof, all860frame registration, or calibration accuracy.')
stamp=b.now()
hashes={str(p.relative_to(N)):b.sha(p) for p in [N/'material-review-plan.json',N/'material-review/index.json',N/'secondary-observations.json',N/'secondary-families.json',N/'classification-result.json']}
ctx=b.read(N/'material-review/contexts/index.json')
b.save(N/'material-visual-review.json',dict(updated=stamp,actual_saved_visual_review=True,reviewed_rows=113,reviewed_pages=11,
    reviewed_context_rows=[r['row'] for r in ctx['rows']],rows=rows,support=support,source_sha256=hashes,
    new_cameras=0,new_smplx_fits=0,training_started=False,
    limits=['Every representative observation was visually inspected. Other tracks in a family are not silently accepted.',
      'A coarse patch is not a physical subpixel point or certified independent material observation.',
      'FIT/HELD identities, raw pixels, matches and K unchanged; no residual selection.',
      'Frame617 mask is still rejected. Original background controls were reviewed on native RGB.',
      'This selected sample is not an unbiased matching error rate or a proof of all grayT misalignment causes.']))
b.save(N/'material-support.json',dict(updated=stamp,**support,combined_rows=accepted))
b.save(N/'material-review/contexts/review.json',dict(updated=stamp,actual_saved_visual_review=True,reviewed_rows=[r['row'] for r in ctx['rows']],new_images_beyond_budget=0))
print(json.dumps(support,ensure_ascii=False))
