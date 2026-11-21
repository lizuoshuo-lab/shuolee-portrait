from pathlib import Path
import json
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame')
review=json.loads((O/'slide-review.json').read_text(encoding='utf-8'))
assert json.loads((O/'visual-review.json').read_text(encoding='utf-8'))['actual_images_opened']==9
review['actual_images_reviewed']=True
(O/'slide-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source=(H/'update_deck_framewise123.mjs').read_text(encoding='utf-8')
changes={
"tag='framewise-route123'":"tag='shared-sequence123'",
"b+'/smplify-observation123-deck-receipt.json'":"b+'/framewise-route123-deck-receipt.json'",
"smplifyx-reference-control/framewise-canonical-route';":"smplifyx-reference-control/framewise-canonical-route/shared-three-frame';",
"replace('逐图贴合与跨视角不一致'":"replace('逐帧姿态与静态高斯输出'",
"replace('SMPLify-X完成求解后，世界几何仍未一致'":"replace('共享体型和高斯，各帧拟合姿态，优先保证面部'",
"replace('三组面部极线中位7.09 / 29.61 / 30.17px'":"replace('网格绑定数值检查通过。共享体型逐帧拟合尚未完成，高斯0步。'",
"replace('原帧0与310。三列：原图、当前投影、另一视角极线。'":"replace('原视频155帧：左原图，右SMPL-X叠加。仍有身体和脚部残差。'",
'02_有限视角重建优化路线_123逐帧姿态与静态输出.pptx':'02_有限视角重建优化路线_123共享体型三帧适配.pptx',
'preserve_deck_framewise123.py':'preserve_deck_shared_sequence123.py'}
for old,new in changes.items():
    assert old in source,old
    source=source.replace(old,new)
(H/'update_deck_shared_sequence123.mjs').write_text(source,encoding='utf-8')
print('Prepared single-slide edit of the existing52-slide deck.')
