from pathlib import Path
import json
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-visibility-100';S=Path('D:/Projects/EngineLab/engine-home')
old='进一步用参考时刻的协方差椭球与MANO三角面求交筛选，未增加任何新的候选点。'
new='进一步检查参考时刻三角面上的欧氏最近点是否落在高斯的三倍标准差协方差椭球内，并结合距离和多视角支持筛选，未增加任何新的候选点。这是保守的最近点启发式，不是椭球与整个三角面的精确相交检验。'
for p in [L/'experiments/37_dynamic_human_reconstruction.md',L/'scripts/dynamic_human/publish_hand_visibility_knowledge.py']:
 s=p.read_text(encoding='utf8');assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf8')
p=L/'scripts/dynamic_human/audit_mano_surface_website.py';s=p.read_text(encoding='utf8').replace('dynamic-human-enerf-mano-surface-099','dynamic-human-enerf-hand-visibility-100').replace("['手部表面高斯与关节运动','1538个','透明度合成']","['手部表面高斯与关节运动','1937个','高斯颜色的光度梯度']").replace("['手部先验与关节表面高斯的图像检验','6000次更新','0.02694']","['旧表面遮挡与不透明度初始化','6000次更新','0.3466']")
(L/'scripts/dynamic_human/audit_hand_visibility_website.py').write_text(s,encoding='utf8')
print('Corrected nearest-surface heuristic description and prepared website checks.')
