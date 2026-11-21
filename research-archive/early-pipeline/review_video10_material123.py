"""Record actual visual review, preserving all raw observations and frozen splits."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,itertools
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'video10-static-material'; U=O/'run'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (O/'visual-review.json').exists()
idx=read(U/'visual-index.json'); stamp=datetime.now(timezone.utc).isoformat()
labels=[
 ('ambiguous','亮灯附近木条/光斑，不能确认独立材质点。'),
 ('ambiguous','相似木条与木节，扩大原生上下文后仍不足以确认唯一物理点。'),
 ('ambiguous','灯光溢出和重复木条，点位置落在变化的亮边附近。'),
 ('fixed_patch_candidate','装饰木牌的同一上部角落及纹理，粗材料身份可信，亚像素位置未认证。'),
 ('ambiguous','同一拱门可见，但坐标在平滑边缘附近，不是已证实的同一个表面点。'),
 ('contradicted','第0帧鞋架与107/644帧冰箱便签连接，明确跨物体错误。'),
 ('fixed_patch_candidate','墙面电暖器右上框角，三个视角可识别同一实体角。'),
 ('ambiguous','鞋架重复鞋底，752帧模糊，不能确认同名点。'),
 ('contradicted','飞镖盘相同黑白图案的不同扇区相连，扩大上下文确认角位置不同。'),
 ('contradicted','飞镖盘另一条轨迹同样跨不同扇区。'),
 ('ambiguous','同一冰箱便签群，但特征在相邻便签边缘/间隙，不能唯一定位对应点。'),
 ('ambiguous','鞋架重复鞋底且752帧模糊。'),
 ('fixed_patch_candidate','同一黑色门垫角落与地砖背景可辨认，坐标是斑块中心而非认证角点。'),
 ('ambiguous','鞋架重复鞋纹，斜视帧模糊。'),
 ('contradicted','地砖接缝和交叉点被接到不同位置，重复平铺不能当材料身份。'),
 ('ambiguous','拖鞋附近轮廓在752帧模糊，材料斑块可能相同但同名点未证实。'),
 ('ambiguous','拖鞋鞋头/鞋面重复区域且752帧模糊，不能认证。'),
 ('ambiguous','鞋架低照度重复纹理，无法唯一辨认。'),
 ('ambiguous','灯附近高光、重复木条与视角相关亮边。'),
 ('ambiguous','灯附近木条，过曝和图像上边界影响唯一性。'),
 ('fixed_patch_candidate','门旁踢脚线转角可识别，同一实体斑块，亚像素定位尚未认证。'),
 ('face_patch_candidate','嘴角下方胡须/下巴斑块似为同一区域，仅FIT，没有独立面部HELD；不作为固定背景相机点。'),
 ('clothing_patch','头部框内实际为衣领/门襟，并非皮肤或鼻嘴点。'),
 ('clothing_patch','头部框内实际为衣领/纽扣附近。'),
 ('clothing_patch','头部框内实际为衣领边缘与项链邻域。'),
 ('clothing_patch','HELD头部框点实际在衣领花纹，不能算面部留出。'),
 ('clothing_patch','HELD头部框点实际在衣领花纹，重复纹样的精确对应仍待核。'),
 ('contradicted','正面门襟、背部衣领、下摆和其他衣物区域串成同一轨迹。'),
 ('clothing_patch','同一胸袋纽扣与口袋条纹可辨认，衣物不是静止场景。'),
 ('clothing_patch','门襟末端衣物斑块，不能当面部点。'),
 ('clothing_patch','胸袋装饰条纹，精确同名点仍需验证。'),
 ('contradicted','眼镜/脸部与背后手腕或前臂纹理连接，明确跨人体部位错误。'),
 ('clothing_patch','衣服门襟斑块，在部分视角落入头部矩形框。'),
 ('clothing_patch','胸袋边缘斑块，非皮肤点。'),
 ('clothing_patch','HELD胸袋装饰纹理，不代表面部验证。'),
 ('clothing_patch','HELD胸袋纹理，斜视帧可见但精确物理点与定位不确定性尚未认证。')]
assert len(labels)==len(idx['rows'])==36
rows=[]
for t,(verdict,note) in zip(idx['rows'],labels):
 rows.append(dict(row=t['row'],family=t['family'],track=t['track'],split=t['split'],original_category=t['category'],frames=[v['frame'] for v in t['observations']],verdict=verdict,note=note,coarse_fixed_patch=verdict=='fixed_patch_candidate',physical_subpixel_verified=False))
selected=[r for r in rows if r['coarse_fixed_patch']]
support={str(f):{s:sum(f in r['frames'] and r['split']==s for r in selected) for s in ['fit','heldout']} for f in idx['columns']}
pairs={f'{a}-{b}':sum(a in r['frames'] and b in r['frames'] and r['split']=='fit' for r in selected) for a,b in itertools.combinations(idx['columns'],2)}
counts=Counter(r['verdict'] for r in rows)
contexts=read(O/'native-context/index.json'); assert len(contexts['rows'])==16
gallery=[dict(file='run/'+f,sha256=sha(U/f),actually_reviewed=True) for f in idx['pages']]+[dict(file='native-context/'+v['file'],sha256=v['sha256'],actually_reviewed=True) for v in contexts['rows']]
summary='视频10的9帧稀疏诊断形成87条三视角轨迹、69个去近重复族。实际审阅36族和16个扩大原生上下文，6族存在明确物理错接。21条背景样本仅3个FIT和1个HELD达到粗材料斑块可辨认；头部框7条中6条为衣领，面部没有独立HELD。当前相机支持不足，未求相机或启动SMPL-X/高斯训练。'
save(O/'visual-review.json',dict(updated=stamp,summary=summary,actual_saved_visual_review=True,rows=rows,gallery=gallery,counts=dict(counts),reviewed_rows=36,reviewed_pages=6,reviewed_contexts=16,
 native_context_resized=False,raw_and_splits_preserved=True,residual_based_rejection=False,selection_bias='This was a frozen spatial/role review sample, not random sampling of all69families. Do not extrapolate6/36as globalerrorrate.',
 source_scope='Only9registered frames out of860. Mostly107/108frame gaps (~4.5seconds), and cycle tracks concentrate near orbit closure. No conclusion that complete video is irrecoverable.',
 material_identity_and_subpixel_distinction='Four coarse fixed material patches are provisional support only; SIFT center need not coincide with a physical corner. No point is certified below1pixel.',
 head_roi=dict(reviewed=7,clothing=6,face_patch_fit=1,independent_face_heldout=0,eye_heldout=0,nose_heldout=0,mouth_heldout=0),
 coarse_fixed_support=dict(fit=3,heldout=1,nonground_fit=2,nonground_heldout=1,spatial_bins=3,per_frame=support,pair_shared_fit=pairs),
 geometry_admitted=False,quality_accepted=False,cameras_estimated=False,K_changed=False,training_started=False,
 causal_boundary='Proves specific wrong correspondences and insufficient reviewed support for this sparse video10 candidate. Does not establish that all old SMPL-X misalignment is caused by camera calibration or by these newvideo tracks. Residual distortion, camera extrinsics, shape/pose model and subject motion remain separate hypotheses.'))
completed=read(U/'result.json'); completed.update(review_updated=stamp,state='complete_reviewed_support_failed',coarse_fixed_fit=3,coarse_fixed_heldout=1,contradicted_review_families=6,independent_face_heldout=0,geometry_admitted=False,quality_accepted=False,summary=summary)
save(O/'result-reviewed.json',completed)
save(O/'progress.json',dict(updated=stamp,state='complete_reviewed_support_failed',pid=None,training_started=False,new_camera=False,result='result-reviewed.json'))
print(json.dumps(dict(summary=summary,counts=dict(counts),support=support,max_shared_fit=max(pairs.values())),ensure_ascii=False))
