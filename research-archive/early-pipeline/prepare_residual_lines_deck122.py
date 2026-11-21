"""Edit only existing slide52 using the verified Artifact Tool preservation workflow."""
from pathlib import Path
import json
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/residual-distortion-direction')
r=json.loads((O/'review.json').read_text(encoding='utf-8'));assert r['actual_review_complete']
s=(H/'update_deck_nativeSupervision_122.mjs').read_text(encoding='utf-8')
s=s.replace('nativeSupervision-122','residualLines-122').replace('native-supervision','residual-distortion-direction')
s=s.replace("root+'/review-inputs/report-comparison.png'","root+'/structure-comparison.png'")
s=s.replace("'/freshPrep-122-deck-receipt.json'","'/nativeSupervision-122-deck-receipt.json'")
s=s.replace('if(review.all_silhouettes_coarsely_reviewed!==464||review.actual_review_complete!==true)','if(review.actual_review_complete!==true||review.native_line_crops_reviewed!==13)')
start=s.index("replace('遮挡观测");end=s.index('const images=',start)
s=s[:start]+"""replace('白衣视频的原生图与分割草稿','残余畸变补偿的独立检验');
replace('第0、232、463帧，每组左为原图、右为分割。','三帧建筑原生裁片：第0、232、463帧。');
replace('464帧图像与时间已固定，完整相机仍待注册。','6条拟合、5条留出；留出误差未下降。');
replace('边界与遮挡仍待验收。面部误差未改善，尚未训练。','候选未采用。对应错配仍在，尚未启动新训练。');
"""+s[end:]
s=s.replace('height:width*541/1824','height:width*1100/4560')
s=s.replace('第0、232、463帧的原图与分割草稿；无图中文字，保留比例','第0、232、463帧原生建筑裁片，保持比例，图内无文字')
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index(".concat([previous]).join('\\n\\n');",start)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 D1残余畸变补偿与T6逐帧对应独立诊断。用户明确要求继续补偿方向；已实际执行新试验，不等同N6冬帽单径向参数结果。',
 *r['findings_zh'],
 '模型为processed_output_inverse_brown4：观测原生像素经固定K归一化后，xu=x(1+k1r²+k2r⁴)+2p1xy+p2(r²+2x²)，yu=y(1+k1r²+k2r⁴)+p1(r²+2y²)+2p2xy。此为输出像素到理想射线的逆模型，不能当成原始FISHEYE_8，也不能当成OpenCV前向系数。未修改训练原图、旧K、外参或原生渲染器。',
 '系数[k1,k2,p1,p2]=[0.05,-0.0329409091,0.0162457713,0.0010419650]。拟合6条、留出5条。拟合组等线权RMS1.138690→1.113886px，留出1.772676→1.783267px。预登记边界、留出和全域位移保护均失败，因此保留诊断而不发布为补偿相机。',
 '解析系数Jacobian与独立差分最大2.03e-7，空间Jacobian差1.72e-10，已知参数前后投影回环误差0。这些是实现数值控制，不是实拍标定。首次测量JSON时间戳函数漏调用导致写盘失败，已保存原脚本和失败证据，窄修复后13张原生审阅图逐字节一致；没有更改观测配置或重新搜索参数。',
 'T6使用单种子相邻原生帧双向LK，全464帧/87材料族，第一次数值拒绝即停，不重检测或吸附旧SIFT锚点。保留27221次尝试及27123个数值通过槽位；每帧至少10场景候选只说明局部数值连续性。',
 'T6非种子原SIFT与逐帧跟踪的最大差：场景1.53665、面部2.31566、衣物168.79320、鞋袜6.29493px。这些是两方法差，不是已知真实点误差。实际13页87行、6面部锚点条、8最大分歧条均已审阅。264帧family974的中/右原生裁片处于不同短裤印花，不能由共同镜头扭曲解释为同一材料点。',
 '当前图只展示第0/232/463帧原生上部建筑裁片，未重采样训练图或伪造补偿后人像。画面高度2704，已测线的y范围约12至818，不能宣称脚部畸变已约束。',
 '下一步继续分离残余映射、边缘/材料身份、相机外参和人物微动。在新的跨画面独立支持建立前，不加阶数硬压误差、不重跑同一候选、不启新高斯训练。真实面部13条留出最大13.04875764px仍未达标。原119预览保留。',
 '证据：robust-portrait-122/residual-distortion-direction/{plan,measurements,line-review,result,review}.json；native-continuity/{result,review}.json及原生裁片。其他51页保持不变。以下保留此前阶段记录。']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_nativeSupervision_122.py','preserve_deck_residualLines_122.py').replace('02_有限视角重建优化路线_122原生监督.pptx','02_有限视角重建优化路线_122残余补偿_数值核对.pptx')
s=s.replace('residualLines-122','residualLinesVerified-122').replace('122残余补偿_数值核对.pptx','122残余补偿_核验完成.pptx')
(H/'update_deck_residualLines_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_nativeSupervision_122.py').read_text(encoding='utf-8').replace('nativeSupervision-122','residualLinesVerified-122')
(H/'preserve_deck_residualLines_122.py').write_text(p,encoding='utf-8')
print('D1_DECK_BUILDER_READY')
