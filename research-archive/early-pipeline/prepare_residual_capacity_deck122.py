from pathlib import Path
import json
H = Path(__file__).parent
R = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
r = json.loads((R / 'native-residual-capacity/review.json').read_text(encoding='utf-8'))
s = (H / 'update_deck_residualCamera_122.mjs').read_text(encoding='utf-8')
s = s.replace('residualCamera-122', 'residualCapacity-122').replace('/native-residual-camera', '/native-residual-capacity')
s = s.replace("const figure=root+'/residual-control.png';", "const figure=root+'/r3-landscape-rgb.png';")
s = s.replace("build+'/gsplatDistortion-122-deck-receipt.json'", "build+'/residualCamera-122-deck-receipt.json'")
s = s.replace('review.checks_passed!==26', 'review.checks_passed!==19')
a, b = s.index("replace('gsplat 的"), s.index('const images=')
s = s[:a] + """replace('残余畸变的统一相机链路','完整畸变相机的全尺寸验证');
replace('同一 K、外参与畸变参数贯穿投影、训练与回读。','20万高斯，双视角更新与实际界面回放已核验。');
replace('合成训练与回读一致；实拍畸变系数仍待验证。','两路浮点回放差为0；实拍标定仍未通过。');
replace('26项集成、11项冻结检查通过；未启动人物训练。','原生1520×2704与2704×1520，完整K、外参与畸变。');
""" + s[b:]
s = s.replace('height:width*160/288', 'height:width*1520/2704')
s = s.replace('残余畸变合成控制：独立射线、CUDA及差值；两台训练相机。原生96×80，图内无文字', '20万完整协方差三阶SH高斯的2704×1520合成原生渲染，图内无文字')
a = s.index('p.resolve(slide.id).speakerNotes.text=')
b = s.index(".concat([previous]).join('\\n\\n');", a) + len(".concat([previous]).join('\\n\\n');")
notes = ['2026-10-28 R3全尺寸残余相机容量与R4实际原生界面回放。', *r['findings_zh'],
         '图为原生2704×1520合成椭球；保持比例，无图内文字。本轮仅一联合更新，不是人物训练、18000步稳定性或真实亚像素验收。',
         '证据：native-residual-capacity/result.json、review.json及native-residual-ui/review.json。以下保留历史备注。']
s = s[:a] + 'p.resolve(slide.id).speakerNotes.text=' + json.dumps(notes, ensure_ascii=False) + ".concat([previous]).join('\\n\\n');" + s[b:]
s = s.replace('preserve_deck_residualCamera_122.py', 'preserve_deck_residualCapacity_122.py').replace('02_有限视角重建优化路线_122残余相机链路.pptx', '02_有限视角重建优化路线_122畸变全尺寸回放.pptx')
(H / 'update_deck_residualCapacity_122.mjs').write_text(s, encoding='utf-8')
s = (H / 'preserve_deck_residualCamera_122.py').read_text(encoding='utf-8').replace('residualCamera-122', 'residualCapacity-122')
(H / 'preserve_deck_residualCapacity_122.py').write_text(s, encoding='utf-8')
print('R3_R4_DECK_READY')
