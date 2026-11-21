from pathlib import Path
H = Path(__file__).parent
source = (H/'update_deck_observation123.mjs').read_text(encoding='utf-8')
changes = {
    "tag='smplify-observation123'": "tag='framewise-route123'",
    "b+'/smplify-conditioned123-deck-receipt.json'": "b+'/smplify-observation123-deck-receipt.json'",
    'smplifyx-reference-control/observation-contract': 'smplifyx-reference-control/framewise-canonical-route',
    "replace('SMPLify-X数值修复与观测冲突'": "replace('逐图贴合与跨视角不一致'",
    "replace('三视角完成拟合；重复手腕目标仍不一致'": "replace('SMPLify-X完成求解后，世界几何仍未一致'",
    "replace('面部中位1.53/0.97/0.94px'": "replace('三组面部极线中位7.09 / 29.61 / 30.17px'",
    "replace('三组左原图、右网格；左右腕重复检测差距中位'": "replace('原帧0与310。三列：原图、当前投影、另一视角极线。'",
    '02_有限视角重建优化路线_123跨视角一致性.pptx': '02_有限视角重建优化路线_123逐帧姿态与静态输出.pptx',
    'preserve_deck_observation123.py': 'preserve_deck_framewise123.py',
    "const height=576*review.figure_wh[1]/review.figure_wh[0];\nimage.frame={left:352,top:372-height/2,width:576,height};":
    "const imageScale=Math.min(576/review.figure_wh[0],352/review.figure_wh[1]);\nconst width=imageScale*review.figure_wh[0],height=imageScale*review.figure_wh[1];\nimage.frame={left:(1280-width)/2,top:372-height/2,width,height};",
}
for old, new in changes.items():
    assert old in source, old
    source = source.replace(old,new)
(H/'update_deck_framewise123.mjs').write_text(source,encoding='utf-8')
print('Prepared same-deck slide52 update; no deck authored yet.')
