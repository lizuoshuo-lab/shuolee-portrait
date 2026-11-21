from pathlib import Path
import json
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grass-exploratory')
p=O/'slide-review.json';q=json.loads(p.read_text(encoding='utf-8'));q['actual_images_reviewed']=True
q['visual_note']='Actual two-panel image inspected: side-view mesh lies outside subject; frontal mesh shifts head/limbs; aspect preserved and no image text.'
p.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=(H/'update_deck_grass_native123.mjs').read_text(encoding='utf-8')
s=s.replace("tag='grass-native123'","tag='grass-trial123'")
s=s.replace("/grass-preflight/generic-input'","/grass-exploratory'")
s=s.replace("'/alignment-initializer123-deck-receipt.json'","'/grass-native123-deck-receipt.json'")
s=s.replace("replace('通用二维与SMPL-X拟合链路'","replace('通用拟合的真实观察检查'")
s=s.replace("replace('原图与初值绑定及完整合成验证'","replace('草地人物94帧，配置保持一致'")
s=s.replace("replace('5个已知相机，160步'","replace('五官被头发遮住时'")
s=s.replace("replace('合成控制：上排初值'","replace('原始帧115、235、465'")
s=s.replace('02_有限视角重建优化路线_123草地原生观察.pptx','02_有限视角重建优化路线_123草地试训.pptx')
s=s.replace('preserve_deck_grass_native123.py','preserve_deck_grass_trial123.py')
(H/'update_deck_grass_trial123.mjs').write_text(s,encoding='utf-8')
print(H/'update_deck_grass_trial123.mjs')
