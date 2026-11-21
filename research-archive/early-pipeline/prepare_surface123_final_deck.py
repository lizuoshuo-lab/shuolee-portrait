from pathlib import Path
import json,hashlib
from PIL import Image
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
F=R/'winterhat-train-018000-review'
paths=[F/'frame_000067_body.png',F/'frame_000067_face.png',F/'frame_000112_face.png']
im=[Image.open(p).convert('RGB') for p in paths]
assert [x.height for x in im]==[480,240,240]
sheet=Image.new('RGB',(im[0].width+24+max(im[1].width,im[2].width),480),'white')
sheet.paste(im[0],(0,0));sheet.paste(im[1],(im[0].width+24,0));sheet.paste(im[2],(im[0].width+24,240))
out=R/'surface123-18000-figure.png';sheet.save(out)
(H/'surface123-18000-figure.json').write_text(json.dumps(dict(path=str(out),width=sheet.width,height=sheet.height,sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],image_content_unchanged=True,text_in_pixels=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
preserve=(H/'preserve_deck_surface123_6000.py').read_text(encoding='utf-8').replace('surface123-6000','surface123-18000')
(H/'preserve_deck_surface123_18000.py').write_text(preserve,encoding='utf-8')
code=(H/'update_deck_surface123_6000.mjs').read_text(encoding='utf-8')
code=code.replace('surface123-6000','surface123-18000').replace('preserve_deck_surface123_6000','preserve_deck_surface123_18000')
code=code.replace("build+'/residualFit-122-deck-receipt.json'","build+'/surface123-6000-deck-receipt.json'")
code=code.replace('winterhat-train-006000-review','winterhat-train-018000-review').replace('review.steps!==6000','review.steps!==18000').replace('123自由表面6000步','123自由表面18000步')
start=code.index("replace('拟合与遮挡使用同一畸变相机'")
end=code.index('const images=',start)
code=code[:start]+"""replace('自由衣物表面训练的6000步检查','自由衣物表面训练的18000步结果');
replace('20万点覆盖穿衣轮廓，侧脸和鞋边仍有重影。','29.4万高斯覆盖衣物，面部与鞋边重影仍未通过。');
replace('自动面部区域留出109/142次误差≥1像素，物理对应尚未验收。','112个相机与点云配套上架，高斯和点云模式均已实测。');
replace('左为全身，右为两种面部视角。每组依次为原图、6000步渲染和实际点中心。','左为全身，右为两种面部视角。每组依次为原图、18000步渲染和实际点中心。');
"""+code[end:]
code=code.replace("alt:'6000步同相机", "alt:'18000步同相机")
start=code.index('p.resolve(slide.id).speakerNotes.text=[')
end=code.index('const candidatePath=',start)
code=code[:start]+"""p.resolve(slide.id).speakerNotes.text=[
 '2026-10-28 本地表面自由点123，戴帽蓝衣新版视频已从头完成18000步。294054个最终高斯，初始200000个，112个真实注册的预登记训练帧。',
 review.summary,
 '本轮实际审阅全部6幅同相机全身、面部与点中心。原帧0的自动面部框落在帽子，不能作为有效人脸。另逐张查看36个10度间隔环绕渲染，背面明显空洞和破碎，没有背面实拍真值。',
 '最终标准PLY完整协方差和三阶SH逆变换通过，6个相机导出前后原生渲染差异在已登记格式阈值内。所有112个相机的完整K、位姿和原视频帧身份，经标准camera数组及绑定provenance联合回读核验一致。相机格式转换不代表物理标定正确。',
 '已在本机Shuolee查看器用打开文件选择实际ZIP，确认294054个高斯、三阶SH和112个SHA绑定相机，并实际选择原帧330查看Gaussian与points。右侧latest已指向本轮，下载包包含PLY和camera。未实测官方SIBR、独立Spark页面或公司00，未断言跨渲染器像素等价。',
 '冻留出自动面部区域142次观察中109次误差>=1像素，真实材料身份尚未逐点认证。训练结束仍不能声明重影修复或强鲁棒性。新的K仍为条件K，外参、对应、人物微动的唯一根因尚未确定。',
 '白T恤人物的独立93个相机、粗拟合与衣物初始表面已准备并审阅，继续同一队列，其他人物仍独立处理。沙发119作为旧对照保留。',
 '最终模型SHA256=814003a9411e0d6eeb9e4cd83b518b487e6e0a7be229e1811b8cc01f121301af。证据目录local-clothed-surface-123，winterhat-quality-review.json，winterhat-viewer-ui-review.json，winterhat-package-camera-parity.json。',
 '图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),
 '以下保留此前研究备注。',previous
].join('\\n\\n');
"""+code[end:]
(H/'update_deck_surface123_18000.mjs').write_text(code,encoding='utf-8')
print(out)
