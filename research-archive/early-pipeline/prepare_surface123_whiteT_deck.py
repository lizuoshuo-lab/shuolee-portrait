"""Prepare the already-reviewed WhiteT final evidence for the existing slide52."""
from pathlib import Path
import json,hashlib
from PIL import Image
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
q=json.loads((R/'whiteT-quality-review.json').read_text(encoding='utf-8'))
assert q['steps']==18000 and q['actual_same_camera_review'] and q['continuous_novel_views_reviewed']==36
assert q['local_viewer_UI']=='passed_actual_OpenFile_Gaussian_points'
F=R/'whiteT-train-018000-review'
paths=[F/'frame_000037_body.png',F/'frame_000019_face.png',F/'frame_000093_face.png']
im=[Image.open(p).convert('RGB') for p in paths]
assert [x.height for x in im]==[480,240,240]
sheet=Image.new('RGB',(im[0].width+24+max(im[1].width,im[2].width),480),'white')
sheet.paste(im[0],(0,0));sheet.paste(im[1],(im[0].width+24,0));sheet.paste(im[2],(im[0].width+24,240))
out=R/'surface123-whiteT-final-figure.png';sheet.save(out)
(H/'surface123-whiteT-final-figure.json').write_text(json.dumps(dict(path=str(out),width=sheet.width,height=sheet.height,
    sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
    image_content_unchanged=True,text_in_pixels=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
preserve=(H/'preserve_deck_surface123_18000.py').read_text(encoding='utf-8').replace('surface123-18000','surface123-whiteT-final')
(H/'preserve_deck_surface123_whiteT_final.py').write_text(preserve,encoding='utf-8')
code=(H/'update_deck_surface123_18000.mjs').read_text(encoding='utf-8')
code=code.replace('surface123-18000','surface123-whiteT-final').replace('preserve_deck_surface123_18000','preserve_deck_surface123_whiteT_final')
code=code.replace("build+'/surface123-6000-deck-receipt.json'","build+'/surface123-18000-deck-receipt.json'")
code=code.replace('winterhat-train-018000-review','whiteT-train-018000-review').replace('123自由表面18000步','123白T恤自由表面完成')
start=code.index("replace('自由衣物表面训练的6000步检查'")
end=code.index('const images=',start)
code=code[:start]+f"""replace('自由衣物表面训练的18000步结果','白T恤人物的自由表面训练结果');
replace('29.4万高斯覆盖衣物，面部与鞋边重影仍未通过。','{q['points']/10000:.1f}万高斯覆盖宽松衣物，面部多重轮廓仍未解决。');
replace('112个相机与点云配套上架，高斯和点云模式均已实测。','93个相机与点云配套上架，后续人物按同一方法继续。');
"""+code[end:]
start=code.index('p.resolve(slide.id).speakerNotes.text=[')
end=code.index('const candidatePath=',start)
notes=[
    f"2026-10-28 本地自由表面123，室外白T恤人物完成18000步。从头拟合及200000初始自由高斯，最终{q['points']}个高斯，93个真实注册的预登记训练帧，独立于此前戴帽蓝衣人物。",
    q['summary'],q['novel_summary'],
    '完整协方差/三阶SH与world/+Y导出已通过本轮格式检查；93个相机的完整K、位姿和原视频帧身份通过标准camera与绑定provenance联合回读。格式一致不代表物理标定通过。',
    '本机Shuolee查看器已通过打开文件载入实际scene.zip，实际审阅Gaussian与points并检查相机数量。未实测官方SIBR、独立Spark页面或公司00，未断言跨渲染器像素等价。',
    '固定留出自动面部ROI含54次观察，47次误差不小于1px，最大28.644px。六组预选原生纹理裁片中3组混入背景横幅、网格或树叶；自动ROI不能替代真实面部材料对应认证。没有按误差删点或改变本轮相机。',
    '戴帽蓝衣人物的18k结果已上架但重影质量未通过；其余人物依次从头训练。原沙发119保留作对照。本轮两个人物的结束不代表强鲁棒性验收。',
    '当前相机K仍为第一组YAML变换后的条件K；外参、残余畸变、对应歧义与人物微动的贡献未唯一分离。',
    '证据目录local-clothed-surface-123；whiteT-quality-review.json、whiteT-viewer-ui-review.json、whiteT-package-camera-parity.json。最终PLY SHA256='+q['source_sha256']
]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
(H/'update_deck_surface123_whiteT_final.mjs').write_text(code,encoding='utf-8')
commit=(H/'commit_surface123_final_deck.py').read_text(encoding='utf-8').replace('surface123-18000','surface123-whiteT-final')
(H/'commit_surface123_whiteT_deck.py').write_text(commit,encoding='utf-8')
print(out)
