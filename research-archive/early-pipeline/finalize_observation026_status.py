from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import requests

B=Path(__file__).parent
BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=BASE/'current/smplx-observation-uncertainty-fresh-026'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
pub=read(R/'web-publication.json')
urls=['http://127.0.0.1:8870/body-smplx.html']+[f'http://127.0.0.1:8870/body-fitting-method/gvhmr-motion/{a["name"]}' for a in pub['assets']]
checks=[]
for url in urls:
    res=requests.get(url,timeout=10);assert res.status_code==200,(url,res.status_code)
    checks.append({'url':url,'http_status':res.status_code,'bytes':len(res.content)})
proof=B/'observation026-page-proof.png';assert proof.exists()
pub.update(browser_verification_pending=False,verified_at=now,http_checks=checks,proof=str(proof),
    visual_check='Actual browser loaded both new images and all5 block MathML formulas; one collapsed appendix preserved.',
    body_geometry_accepted=False)
save(R/'web-publication.json',pub)
deck=read(B/'observation026-deck-receipt.json');assert deck['canonical_replaced']
assert sha(Path(deck['source']))==deck['final_sha256']
status=read(BASE/'work-status.json')
status['body_observation_026'].update(web=pub['anchor'],presentation={'path':deck['source'],'sha256':deck['final_sha256'],'changed_slides':[55]},
    next_route={'name':'SAM3DBody and MHR conversion','public_code_checked':True,'weights_available':False,
        'access_request_pending':True,'parallel_next':'Validate official public conversion examples and camera/unit contracts without subject inference.'})
status['latest_report']={'path':deck['source'],'sha256':deck['final_sha256'],'changed_slides':[55],'updated':now}
status['updated']=now
save(BASE/'work-status.json',status)
plan=BASE/'long-term-plan.md';text=plan.read_text('utf-8-sig')
old='下一步检查两套检测器在真实肩、肘、膝位置的差异、裁切/像素约定与可见性，再做观测不确定性的从头对照；不直接把ViTPose当真值，不重复整个人体冻结或盲目加步数。'
assert text.count(old)==1
text=text.replace(old,'这些观测核验及方向不确定性对照已在026完成，详见下段；不直接把ViTPose当真值，不重复整个人体冻结或盲目加步数。')
insert='''026全部59帧的源图、时间、裁切和像素坐标核对通过；原图中的肘膝位置仍受衣物、遮挡和检测定义影响。三组分别使用原检测、两检测器均值、均值加分歧方向降权，均从原始GVHMR/相机/模板重建起点，各1200步。第一组与024身体纹理组所有输出数组散列相同，三组初始数组相同。身体纹理诊断中位残差2.971/2.815/2.697px，足部1.312/1.370/1.439px；固定左右侧面及23/45实拍叠图未见明确形态改善，三组不采用。新3600次身体更新、0次高斯更新。未将检测点当GT或宣称连续视频通过。

下一条独立对照为SAM 3D Body原始图像预测与官方MHR到SMPL-X转换。已固定源码版本并检查真实内参、遮罩输入与转换接口。官方HF权重需人工批准访问，匿名请求401、当前未配置HF凭据；已向用户提供申请入口和三项必要文件，不从非官方来源绕过访问要求。等待权重期间可用官方公开示例检查单位/坐标与转换误差。实际转换代码方法名为convert_sam3d_output_to_smpl，与README不同；SAM输出顶点尚未加pred_cam_t，转换到MHR输入时米乘100，不能重复加位移或重复翻转轴。便捷函数默认single_identity=False，序列共享形状需明确调用底层转换。原始几何通过已知相机的叠图后，才考虑共享世界注册和新的高斯训练。

'''
anchor='通过裸网格原图叠加、固定时刻多方向与时间一致性后，再从20万新高斯和新优化器训练18000步。'
assert text.count(anchor)==1;text=text.replace(anchor,insert+anchor)
text=text.replace('[运动与绝对几何](http://127.0.0.1:8870/body-smplx.html#gvhmr-body-observation-025)', '[观测分歧与拟合](http://127.0.0.1:8870/body-smplx.html#body-observation-026) · [运动与绝对几何](http://127.0.0.1:8870/body-smplx.html#gvhmr-body-observation-025)',1)
plan.write_text(text,'utf-8')
rp=R/'research/prior-alternative-audit.json';research=read(rp)
research['code_contracts']={'actual_conversion_method':'Conversion.convert_sam3d_output_to_smpl',
    'readme_method_name_matches_current_code':False,
    'sam3d_vertices_camera_meters':'pred_vertices + pred_cam_t[:,None,:]',
    'mhr_conversion_vertices_centimeters':'100 * (pred_vertices + pred_cam_t[:,None,:])',
    'axis_conversion':'pred_vertices branch no extra flip; mhr_model_params replay branch flips Y/Z before translation.',
    'convenience_conversion_single_identity':False,
    'camera_input':'Pass torch.Tensor, code calls cam_int.to(batch[img]); preserve calibrated intrinsics.'}
save(rp,research)

targets=[]
for arm in ['rtmw','mean','uncertainty']:
    for name in ['fitted.npz','initialization.npz','starting-state.npz']:
        targets.append((R/arm/name,'Unaccepted body fit or reproducible duplicate initialization; metrics, input contracts and source preserved.'))
keep={'source-000023.jpg','source-000045.jpg','fixed-left90.jpg','fixed-right90.jpg','fixed-left90-time-sheet.jpg','texture-45.jpg'}
targets += [(p,'Redundant diagnostic render; selected views retained.') for p in (R/'review').glob('*.jpg') if p.name not in keep]
targets += [(p,'Unchanged draft render; canonical deck and changed slide proof retained.') for p in (B/'observation026-final-renders').glob('*.png') if p.name!='55.png']
targets += [(Path(deck['candidate']),'Private draft after canonical publication.'),(Path(deck['finalPath']),'Validated duplicate after canonical publication.')]
protected=[Path(deck['source']),BASE.parent/'viewer/body-smplx.html',
    BASE/'current/smplx-gvhmr-shoe-motion-fresh-023/fitted.npz',
    BASE/'current/smplx-gvhmr-shared-world-fresh-025/shared/fitted.npz',
    Path('D:/GS-SLAM-Data/models/vposer/V02_05/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt'),
    R/'phase-conclusion.json',proof,B/'observation026-final-renders/55.png']
files=[]
for p,reason in targets:
    assert p.is_file(),p
    entry={'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p),'reason':reason}
    if p.suffix=='.npz':
        with np.load(p) as z:entry['arrays']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype),'sha256':hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()} for k in z.files}
    files.append(entry)
assert len({x['path'] for x in files})==len(files)
manifest={'created':now,'state':'planned','scope':'Only026 rejected models/redundant renders and026 publication drafts. No previously blocked targets.',
    'allowed_roots':[str(R.resolve()),str(B.resolve()),str((B.parent/'shuolee-slide-output').resolve())],
    'protected':[{'path':str(p.resolve()),'sha256':sha(p)} for p in protected],
    'files':files,'total_bytes':sum(x['bytes'] for x in files),'count':len(files)}
save(R/'cleanup-plan.json',manifest)
print(json.dumps({'website_verified':True,'canonical_deck_verified':True,'cleanup_files':len(files),'cleanup_bytes':manifest['total_bytes']},ensure_ascii=False))
