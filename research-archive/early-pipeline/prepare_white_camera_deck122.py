"""Prepare a small native slide52 edit using the existing presentation layout."""
from pathlib import Path
import json
H=Path(__file__).parent
assert not (H/'update_deck_whiteCamera122.mjs').exists()
s=(H/'update_deck_whiteGeometry122.mjs').read_text(encoding='utf-8-sig').replace('whiteGeometry122','whiteCamera122')
s=s.replace("build+'/whiteGrid122-deck-receipt.json'","build+'/whiteGeometry122-deck-receipt.json'")
s=s.replace('../clip03-fresh-geometry/face_feet_evidence.png','../clip03-camera-observability/background_face_feet.png')
a=s.index("replace('均衡特征分配");b=s.index('const images=',a)
s=s[:a]+"""replace('固定背景相机下的面部错位','背景相机约束对面部的影响');
replace('三排为面部原图','三排为背景、面部、鞋面留出投影，三列为200、232、280帧。');
replace('背景留出中位数','假设背景观测噪声为1px，面部相机敏感度约为鞋袜7倍。');
replace('面部最大13.05px','这是局部敏感度，尚不能认定为13.05px面部错位的唯一原因。');
"""+s[b:]
s=s.replace("alt:'原生面部及鞋面留出投影，蓝圈观测，粉色十字预测'","alt:'背景、面部、鞋面原生留出投影，蓝圈观测，粉色十字预测'")
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index('const candidatePath=',a)
notes=[
'2026-10-28 N12h。使用第一组YAML在4000×3000参考尺寸下的既有条件K，按原N4下采样与居中裁剪到1520×2704。尺寸字段说明标定参考坐标，真实去畸变输出K尚无独立导出记录。本轮不改内参、相机、三维点或二维观测。',
'在N12g完成结果的原216相机规范下只读线性化。40背景FIT族、207观测、414残差坐标。固定216位姿与216到232的单位基线，消去120个三维点变量后，相机35自由度数值满秩，列归一条件数141.10。未用heldout或面部建立信息矩阵，未加入掩盖空方向的正则。',
'假设每个背景拟合像素坐标存在独立1px标准差，传播至目标留出三角化预测的主轴标准差增益：背景86目标中位0.531687、最大1.081199；面部41目标中位2.675785、最大3.532888；鞋袜47目标中位0.377976、最大0.493526。面部与鞋袜中位比7.079。单位噪声只用于敏感度归一，实际噪声未知。这些数值不是实测相机误差、置信区间或逐点1px合格结论。',
'背景信息矩阵独立确定的最弱模式，在拟合残差总L2变化归一为1时，背景留出投影变化中位0.050934px、最大0.464654px，面部中位0.784479px、最大2.284266px，鞋袜中位0.056924px。此为局部线性响应，通过极小参数扰动有限差分核验。没有用面部误差选方向或拟合新相机，也没有把单位幅度假想相机输出为候选。',
'固定点支持存在深度不均衡。FIT地面深度中位9.804874个初始基线单位，视差中位27.639°。非地面固定点深度中位107.938670，视差1.278853°。这是诊断候选的相对几何，不是米制测量，也不保证物理K正确。',
'图中背景family282、面部family800、鞋面family1454均为冻结heldout族。三列200、232、280帧，每格128×128原生像素，原SIFT蓝圈与N12g不使用目标观测三角化所得粉十字。图不加文字，不修改训练图。背景按最小可覆盖三帧的familyID选取，面部及鞋面沿用N12g例图，未按残差选择。',
'结论：背景、鞋面低残差不足以保证人脸受约束。相机弱约束需要保留为风险，不能直接归因人物微动。本轮也未证明它足以解释全部13.05px面部误差。输出K、残余投影模型偏差、材料定位偏差与人物变化仍未完全分离。未生成新相机、点云或训练。',
'数值核验：解析Jacobian相对差1.21e-10；相机消元协方差与全联合SVD相对差3.57e-13；相机传递导数两步长差1.69e-9；原规范与184规范投影差9.09e-13px。2.65秒CPU，一次运行，输入哈希保持。数值控制通过不等于物理标定通过。',
'下一步预登记并核验既有原生图像和轨迹中，是否有处于人物同等深度、头部高度的可信固定场景物体及独立留出。若没有，明确证据缺口，不重复BA或用人脸拟合后断言人物微动。',
'来源：robust-portrait-122/video-update-20261027/clip03-camera-observability-plan.json，clip03-camera-observability/result.json、FIT-information.json、target-sensitivities.json、structure-support.json、figure-index.json、visual-review.json。以下保留历史诊断：'
]
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[b:]
s=s.replace('02_有限视角重建优化路线_122固定背景与面部误差.pptx','02_有限视角重建优化路线_122相机空间敏感度.pptx')
(H/'update_deck_whiteCamera122.mjs').write_text(s,encoding='utf-8')
(H/'preserve_deck_whiteCamera122.py').write_text((H/'preserve_deck_whiteGeometry122.py').read_text(encoding='utf-8-sig').replace('whiteGeometry122','whiteCamera122'),encoding='utf-8')
a=(H/'accept_white_geometry122.py').read_text(encoding='utf-8-sig').replace('whiteGeometry122','whiteCamera122').replace('clip03-fresh-geometry','clip03-camera-observability').replace('robust122-n12g','robust122-n12h').replace('N12g','N12h')
a=a.replace('N12h heading,4000basis,fresh7view geometry result,background/face/feet error table and3nativecrop rows actuallyvisible','N12h heading,4000basis,conditional unit-noise sensitivity table and3nativecrop rows actuallyvisible')
a=a.replace('new_camera_candidate=True','new_camera_candidate=False')
a=a.replace('N12e localphotometric diagnosis,N12f31roof review,N12h fresh7view fixedKcamera candidate and41face/47feet review completed. Backgroundmedian.332px butfaceheldout3.891px/max13.052 fails. ActualUI andsamePPT52completed;no camera/modelpromotion ortraining.','N12h once-only background information and174target transfer sensitivities reviewed. Face/feet unit-noise gain ratio7.079; no actual uncertainty or unique-cause claim. ActualUI andsamePPT52completed;no camera/modelmutation ortraining.')
a=a.replace('latest_white_geometry_PPT','latest_white_camera_sensitivity_PPT').replace('Actual N12h fixedKgeometry andface/feet evidence report','Actual N12h local camera sensitivity report')
(H/'accept_white_camera122.py').write_text(a,encoding='utf-8')
print('N12H_DECK_EDIT_PREPARED')
