"""Prepare the existing slide52 edit with an unannotated actual-UI crop."""
from pathlib import Path
from PIL import Image
import json,hashlib
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-training-entry')
source=O/'report-ui.png';box=(45,122,786,596)
with Image.open(source) as im:im.crop(box).save(O/'report-table.png')
(O/'figure-provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),crop_xyxy=box,output='report-table.png',resampled=False,added_text=False,meaning='Actual T3 report table; not a reconstruction image.'),ensure_ascii=False,indent=2),encoding='utf-8')
s=(H/'update_deck_nativeUIV2_122.mjs').read_text(encoding='utf-8')
s=s.replace('nativeUIV2-122','nativeEntry-122').replace('native-ui-v2','native-training-entry').replace("'/ui-completed.png'","'/report-table.png'")
s=s.replace("'/nativeV2-122-deck-receipt.json'","'/nativeUIV2-122-deck-receipt.json'")
s=s.replace("if(review.CPU_checks!==20||review.actual_UI_renders!==4)","if(review.checks!==46||review.actual_exit_code!==3)")
s=s.replace("replace('原生渲染修复与点云回读','原生界面的相机与点云包');","replace('原生界面的相机与点云包','训练前的相机与材料验收');")
s=s.replace("replace('上排针孔，下排已知鱼眼。三列为相同三组相机。','按包内完整相机渲染，RGB、Alpha与配套下载已核验。');","replace('按包内完整相机渲染，RGB、Alpha与配套下载已核验。','实际输入检查：原图、相机、遮罩、材料轨迹。');")
s=s.replace("replace('20万点恢复30,681点更新。','4次实际界面渲染与参考一致，点云和相机下载字节一致。');","replace('4次实际界面渲染与参考一致，点云和相机下载字节一致。','46项检查通过；当前视频在高斯初始化前停止。');")
s=s.replace("replace('组件41项和命令行2项通过。','仅完成合成场景加载链路验证，实拍人脸标定仍未通过。');","replace('仅完成合成场景加载链路验证，实拍人脸标定仍未通过。','面部最大误差仍13.049px，未启动新人物训练。');")
s=s.replace('frame.height*1265/712','frame.height*741/474').replace("alt:'实际原生v2界面：按包内记录相机显示已完成合成高斯，无新增图片注释'","alt:'实际训练入口报告表；截图裁剪无重采样，没有在照片上添加文字'")
begin=s.index('p.resolve(slide.id).speakerNotes.text=[');end=s.index("].join('\\n\\n');",begin)+len("].join('\\n\\n');")
notes=[
 '2026-10-28 T3已准备原生输入的训练入口。实际实现native_training_admission122.py与train_native_person122.py，主命令在任何高斯初始化、torch/GPU和优化器之前无条件调用材料、相机、原图与遮罩验收。没有绕过或续训参数，原113入口、运行中的v1/v2界面及旧高斯保持不变。',
 '114.036秒的单次CPU验证完成46项检查。已知四视角合成输入用于正控制；缺少帧清单、相机／原图帧不匹配、插值位姿、非刚性外参、像素约定错误、二次鱼眼、材料／静止性／测量不确定性、开发暴露和拟合留出混族、面部误差超限、遮挡监督、非布尔mask、缩放图像和无有效SSIM窗口均被对应原因拒绝。合成负例重绑证据，避免仅因陈旧哈希通过拒绝测试；不把这些合成标签移到实拍。',
 '实际白衣片clip03输入源SHA0a4d31cfcf31036ef2244b2709daaf29b425c1921c5516183302670728d030de，1520×2704，464为编码帧总数。现有184、200、216、232、248、264、280共7个诊断相机；完整训练帧清单尚未建立。纠正之前个别摘要把464帧直接称为训练帧的说法；没有据此宣布457个已登记训练相机缺失，也不把7帧局部验证冒充全视频。',
 '实际train命令退出3，status为input_rejected，GPU_imported、initializer_loaded、optimizer_constructed、training_started全部false。原有102条N13观测完整保留，逐点数值差0；13条脸部数值预测最大13.0487576436px，物理准入0条。此次入口拒绝不是新的相机实验，不是误差改善，也没有凭错误点筛除取得通过。',
 '入口报告保留具体缺口：7张原生RGB可核对，但前景RGB有效mask、Alpha、遮挡／观测有效范围未形成训练输入；已有人物排除mask是场景匹配用途，不能替代人物衣物Alpha。历史FIT轨迹存在，但本诊断包未包含其物理合格N3观测。另缺区域可见性、输出投影、位姿深度支持、静止场景与人物微动、世界+Y依据。不能用单一accepted=true代替可追溯审阅。',
 '实际CPU读取验证单张图像缓存和完整K的整数中心到像素角点+0.5桥接，不缩放图像。20万点初始化读取控制保留中心、完整四元数尺度、透明度和三阶SH，且必须绑定本轮视频、相机、训练帧及新执行拟合；旧个体、非零训练步、相机不同、点数不等于20万均拒绝。该正例为已声明合成数组，未冒充新的人体拟合。',
 '本轮只完成准备后输入验收／读取／分派接口；成功实拍GPU训练、原视频相机求解、从头人体拟合和衣物表面生成执行器仍是上游待接环节，原生分辨率容量未测试。物理证据的SHA仅绑定原文，不能防止人为伪造或证明其物理真值。没有新点云，没有00部署、SIBR/Spark等价、连续未知视角、四人物或新视频泛化验收。',
 '继续采用用户说明的第一组YAML K在4000×3000去畸变输出基准，再下采样、居中裁剪；fx1658.0353708、fy1656.5394007、cx2001.5157183、cy1505.8269491。YAML字段证明参数的标定像素尺寸，不单独证明K_rect与输入K完全相同。N5/N6/N12g已使用4000基准，现有大误差不能归因于本轮仍按4096计算。实际输出与整数裁剪取整待独立证实，不二次去畸变，不重新拟合K。',
 '页面、查看器latestDiagnostic与同一工作日志已更新；历史119沙发204649点、89个相机保留。图为当前入口检查表的真实浏览器截图裁剪，未重采样、没有新增图中文字。52页PPT只替换本页，前51页包内容和渲染不变。下一步接通从头拟合和自由衣物表面初始化，继续执行相机和材料准入；不再重复这些已通过控制或失败旧实验。',
 '证据：robust-portrait-122/native-training-entry/plan.json、result.json、review.json、actual-entry/admission.json、actual-entry/status.json、report-ui.png、figure-provenance.json。以下保留历史记录；其中旧464训练帧措辞以上述纠正为准。']
s=s[:begin]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_nativeUIV2_122.py','preserve_deck_nativeEntry_122.py').replace('02_有限视角重建优化路线_122原生界面.pptx','02_有限视角重建优化路线_122训练入口.pptx')
(H/'update_deck_nativeEntry_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_nativeUIV2_122.py').read_text(encoding='utf-8').replace('nativeUIV2-122','nativeEntry-122')
(H/'preserve_deck_nativeEntry_122.py').write_text(p,encoding='utf-8')
print('T3_DECK_BUILDER_READY')
