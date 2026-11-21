"""Record actual local viewer checks; PC00 is explicitly not claimed."""
import datetime,hashlib,json,urllib.request
from pathlib import Path
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-world-fresh-062';V=B.parent/'viewer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');q.replace(p)
pub=V/'models/whole-body-static-fresh/delivery.json';delivery=read(pub)
ply=Path(delivery['download_path']);assert hashlib.sha256(ply.read_bytes()).hexdigest()==delivery['sha256']
with urllib.request.urlopen(urllib.request.Request(delivery['download_url'],method='HEAD'),timeout=15) as r:
    assert r.status==200 and int(r.headers['Content-Length'])==ply.stat().st_size
    http=dict(status=r.status,bytes=int(r.headers['Content-Length']))
names=['preset-gaussian-proof.png','standalone-ply-gaussian-proof.png','standalone-ply-points-proof.png','delivery-page-proof.png']
assert all((R/'review'/n).stat().st_size>1000 for n in names)
proofs={n:hashlib.sha256((R/'review'/n).read_bytes()).hexdigest() for n in names}
verification=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    local_viewer=True,actual_browser_interaction=True,preset_load_passed=True,
    standalone_open_file_passed=True,keep_previous_view=False,orientation='default +Y up',
    gaussian_mode_passed=True,center_point_mode_passed=True,displayed_points=150692,displayed_sh_degree=3,
    no_external_template_or_transform=True,quality_accepted=False,pc00_opened_this_round=False,
    page_images_loaded=6,http_download=http,proof_sha256=proofs,ply_sha256=delivery['sha256'])
save(R/'review/viewer-validation.json',verification)
for p in [pub,R/'gaussian-run/viewer-export/delivery.json']:
    d=read(p);d['local_viewer_validation']=verification;save(p,d)
index=read(B/'gaussian-deliveries.json');index['rounds']['062']['local_viewer_validation']=verification;index['updated']=verification['created'];save(B/'gaussian-deliveries.json',index)
con=read(R/'phase-conclusion.json');con['local_viewer_validation']=verification;save(R/'phase-conclusion.json',con);save(V/'body-static-current/conclusion.json',con)
status=read(B/'work-status.json');status['updated']=verification['created'];status['latest_local_execution']['local_viewer_passed']=True;status['latest_local_execution']['local_ply']=str(ply);save(B/'work-status.json',status)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8');a=text.index('<!-- GVHMR_059_ACTIVE -->');b=text.index('<!-- END_GVHMR_059_ACTIVE -->')+len('<!-- END_GVHMR_059_ACTIVE -->')
active=f'''<!-- GVHMR_059_ACTIVE -->
## 当前执行：整个穿衣人物先做静态

2026-10-23用户明确“先把整体做静态”，避免观察方向变化被解释为身体/衣物运动。当前所有175个有效实拍帧共同监督同一组三维高斯，只改变各帧相机；静态整体未验收前不恢复动态主线。高斯应形成衣物外表面，不要求贴裸模。未拍摄的俯仰方向只作诊断，不当作真值。

062已完成：原图和相机出发，200000个新点、18000步、最终150692点；175帧每帧102—103次更新。59个原始标定相机，116个时间插值相机，后者不能称为真值。重新拟合的身体仅供初始化粗范围和尺度；无人体绑定、无身体驱动、无衣物时间形变、无旧高斯或旧优化器。

同相机定性比较，062的正面、斜侧面和俯视轮廓较060完整，衣摆与裤腿拖影减少；侧后衣袖、仰视鞋底与脸部细节仍未通过。静态点中心构成穿衣轮廓，但没有衣物三维真值。060是身体驱动固定时刻，062是共享静态模型，两者并非同姿态单项消融。

交付：{delivery['download_url']} 。本地：{ply} 。查看：http://127.0.0.1:8870/?model=whole-body-static-fresh 。当前讲解：http://127.0.0.1:8870/body-smplx.html#pose-clothing-current 。事实记录：current/person-static-world-fresh-062/phase-conclusion.json。

175个帧索引下高斯中心与协方差逐位相同；八相机原生/PLY重载、+Y坐标和SH转换核验通过；本地查看器预设及独立打开文件的高斯/点云模式已实测。PLY文件名f0100仅作查看相机参考，不是动态姿态。旋转参数归一化的导出修复未做任何新增优化，18000步检查点未改变。PC00本轮未远端打开，不宣称部署或实机验收。

下一项：先核验175帧原图特征与相机投影，分别统计59个标定相机和116个插值相机的残差、覆盖与遮挡，区分相机误差、真实微动、遮罩错误。证据支持后再选择从原图重估全帧相机或有界相机优化；下一轮高斯仍重新初始化，不接着062修补。不得只凭训练误差变小宣布新视角质量通过。

历史对照：060身体驱动132663点；061观测衣物运动134077点，控制拟合残差降低但图像重影未见整体收益。保留必要对照和原始观测依赖，不恢复061作为当前主线。当前训练已完成，GPU没有正在跑下一轮；先做静态相机诊断。

每轮都须提供00查看器可直接读的+Y向上标准3DGS PLY和完整路径，核验全部协方差/SH变换及原生/重载一致性，登记gaussian-deliveries.json。独立导入不依赖人体模板或骨架。每次新实验从原始图像、相机、许可模板或官方预训练观测开始，旧个体拟合、高斯、残差场与优化器只读比较。先本地，质量通过后再操作00。眼镜专项停止，不同步E盘。工程清理保留原视频、代码、许可模板、必要对照和当前候选；后续有实质结论更新同一份方法PPT，不累计重复报告。
<!-- END_GVHMR_059_ACTIVE -->'''
plan.write_text(text[:a]+active+text[b:],encoding='utf8')
print(json.dumps({'viewer_checks':True,'download':delivery['download_url'],'bytes':ply.stat().st_size,'sha256':delivery['sha256'],'pc00_actual_test':False},ensure_ascii=False))
