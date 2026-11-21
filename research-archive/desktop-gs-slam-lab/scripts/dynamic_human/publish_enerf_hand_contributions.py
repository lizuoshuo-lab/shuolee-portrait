from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-contributions-086';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(O/'final-summary.json')['actual_panels_reviewed']==13
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md')
old='接下来的检查需同时看高斯对手部像素的**实际可见贡献**：中心不在手部掩码内的高斯，其投影覆盖范围仍可能跨入手部，仅按中心筛选更新对象可能遗漏这些贡献。这个待检验的问题不能直接当作模糊成因。'
new='还需看高斯对手部像素的**实际可见贡献**：中心不在手部掩码内的高斯，其投影覆盖范围仍可能跨入手部。当前39张训练画面的检查中，可更新高斯平均承担估计手部区域约81.5%的贡献，但这些贡献仍延伸到皮肤之外的碎片。故不能把失败主要归因于冻结范围；可更新的覆盖率、表面归属与局部损失是否提供有效梯度，仍是三个不同的问题。估计掩码本身也会漏掉手指，不能当作几何真值。'
assert old in txt;txt=txt.replace(old,new);p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md')
old='下一步转向可见贡献检查：原筛选使用中心投影与估计深度，而高斯覆盖的像素范围比中心更大。需先确定被冻结的高斯是否仍大量贡献错误手部像素，再决定是否调整局部优化范围。所有结论仍属于同一场景开发，完整背景与独立新场景验证未完成。'
new=r'''补充检查沿用13个训练机位的728、730、732帧及原手部掩码，将选中与被冻结高斯的指示属性一起进行透明度合成，保留原排序和遮挡。各组贡献之和与总透明度数值一致，因而测量的是实际渲染贡献，而非高斯中心是否落在掩码内。

![原照片、原重建、可更新高斯的可见贡献占比与估计手部掩码](../outputs/dynamic-human-enerf-hand-contributions-086/camera04-contribution.png)
> 图源：同上 · 机位04，728、730、732帧 · 第三列黑到白对应占比0至1；第四列为估计掩码。

39个机位时刻等权平均，可更新高斯承担估计手部区域81.5%的贡献；约13.0%的区域像素由冻结组贡献占多数，这些像素承载约12.4%的RGB绝对误差。该结果不支持“多数手部错误因为被冻结而无法修改”的解释。图中白色贡献仍延伸成皮肤之外的碎片，说明可更新范围覆盖手部，不等于其中已有正确的手指表面。04机位728帧的掩码还明显遗漏部分手掌，不能将掩码外的贡献全部判成错误。

13张、39个原生画面的实际查看及数组复核已完成；[侧面机位的同类对照](../outputs/dynamic-human-enerf-hand-contributions-086/camera13-contribution.png)、[贡献诊断](../outputs/dynamic-human-enerf-hand-contributions-086/final-summary.json)、[看图记录](../outputs/dynamic-human-enerf-hand-contributions-086/visual-review.json)保留在这里。此项没有训练新模型。下一步检查局部手部图像与其余画面对共享高斯参数的梯度是否稀释或冲突，再决定局部表面优化方式。所有结论仍属于同一场景开发，完整背景与独立新场景验证未完成。'''
assert old in txt;txt=txt.replace(old,new);p.write_text(txt,encoding='utf8')
files=['final-summary.json','visual-review.json','camera04-contribution.png','camera13-contribution.png'];p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');assert O.name+'/$'+'{name}' not in txt;lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-hand-fine-085/$'+'{name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+".map(name => [\x60outputs/"+O.name+"/$"+"{name}\x60, '手部可见贡献诊断']),\n");p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n',encoding='utf8');shutil.copy2(__file__,O/Path(__file__).name)
print('Existing section updated with measured contribution result;4resources allowlisted')
