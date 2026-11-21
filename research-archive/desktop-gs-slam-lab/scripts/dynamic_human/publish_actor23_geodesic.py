"""Publish inspected native evidence and one connected theoretical explanation."""
from pathlib import Path
import json, shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-geodesic-107';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
c=read(O/'comparison/summary.json');v=read(O/'visual-review.json');summary=read(O/'final-summary.json')
assert c['PNG_checks']==1584 and len(v['inspected'])>=12 and not summary['goal_complete']
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');assert '#### 控制节点的连接与高斯绑定' not in s
shutil.copy2(p,O/'knowledge-before.md')
needle='### 时间基函数与局部运动'
new=r'''#### 控制节点的连接与高斯绑定

局部运动中有两种不同的连接：节点之间的边决定哪些运动受共同的刚性约束，高斯到节点的权重决定哪些运动实际作用于该高斯。两者都只使用空间最近邻时，相邻的两条腿、手与衣物可能产生不合适的连接；但改变连接也不等于获得正确的人体关节。

设初始轮廓壳层的离散邻接图为 $\mathcal H$，$\Pi_{\mathcal H}(\mathbf x,\mathbf y)$ 为图上连接两点的路径集合。欧氏距离与沿该图的最短路径长度分别为

$$
d_{\mathrm E}(\mathbf x,\mathbf y)=\|\mathbf x-\mathbf y\|_2,
\qquad
d_{\mathcal H}(\mathbf x,\mathbf y)=
\min_{\pi\in\Pi_{\mathcal H}(\mathbf x,\mathbf y)}
\sum_{(\mathbf v_a,\mathbf v_b)\in\pi}\|\mathbf v_a-\mathbf v_b\|_2.
$$

两者都使用场景长度单位。沿图选取控制节点，可以减少穿过空隙的短连接。本地对照保留同一批节点，先只用 $d_{\mathcal H}$ 重建节点之间的邻接边，再额外用它选择每个高斯的四个控制节点 $\mathcal N_i$。后者的归一化权重为

$$
w_{ik}=\frac{\exp[-d_{\mathcal H}(\mathbf x_i,\mathbf g_k)^2/(2\sigma_i^2)]}
{\sum_{\ell\in\mathcal N_i}\exp[-d_{\mathcal H}(\mathbf x_i,\mathbf g_\ell)^2/(2\sigma_i^2)]},
\qquad \sigma_i=\frac{d_i^{(4)}}{2}.
$$

$d_i^{(4)}$ 是到第四近控制节点的路径距离，$\sigma_i$ 与位置使用同一长度单位，$w_{ik}$ 无量纲；恰好位于节点的高斯直接绑定该节点。由此，一组对照只改变运动约束，另一组同时改变高斯运动的混合权重。轮廓壳层及其碎片连接仍是几何近似，不能把这里的路径距离当作真实人体表面的测地距离。

![原照片与三种运动连接方式的下半身原生像素对照](../outputs/dynamic-human-actor23-geodesic-107/lower-body-review/camera00-frame000601-lower-body.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位00、帧601 · 原照片、欧氏近邻、仅改节点连接、同时改高斯绑定。

'''+v['knowledge_result_zh']+'\n\n[完整人物、手部与连续帧对照](../experiments/37_dynamic_human_reconstruction.md#局部运动连接与高斯绑定对照)保留本次开发验证的条件与结果。\n\n'
assert s.count(needle)==1;s=s.replace(needle,new+needle,1);p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');assert '### 局部运动连接与高斯绑定对照' not in s;shutil.copy2(p,O/'evidence-before.md')
s+='''

### 局部运动连接与高斯绑定对照

这组对照使用 actor2_3 的同一195张原生训练照片、相机时间偏移和初始高斯，每组固定15000次更新。384个控制节点的位置不变，节点之间仍有3072条有向边；沿初始壳层路径选邻居后，其中452条边被替换。第二组进一步改变高斯绑定，其中15355个初始高斯的邻居集合变化。高斯复制与分裂继承原有绑定。这里改变了邻接权重和空间支持，未增加关节标签或人体模板。

训练前已核对初始参数、照片顺序和背景采样；最终高斯数量可以不同。这是单随机种子的开发对照，之前查看过的验证图像不能再称为新的盲测。下列平均前景PSNR依次对应欧氏近邻、仅改节点连接、同时改高斯绑定：

'''
for key,label in [('validation_time','已知机位180张留出时刻'),('validation_camera_time','另外机位36张奇数帧'),('validation_camera','另外机位39张偶数帧')]:
    m=c['splits'][key]['foreground_psnr_db'];s+=f"{label}：**{m['dynamic_clock']:.2f}、{m['path_rigidity']:.2f}、{m['path_rigidity_binding']:.2f} dB**。\n\n"
s+=v['evidence_result_zh']+'\n\n'
s+='''![完整人物对照保留抬脚及相互遮挡上下文](../outputs/dynamic-human-actor23-geodesic-107/comparison/camera00-frame000601-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站实测 · 机位00、帧601 · 原照片、欧氏近邻、仅改节点连接、同时改高斯绑定。

![手部与物体上下文的同条件对照](../outputs/dynamic-human-actor23-geodesic-107/comparison/camera13-frame000613-crouching_hands_object_context.png)
> 图源：同上 · 机位13、帧613 · 原生像素，无锐化或补绘。

下半身窗口专门包含抬脚、膝盖和支撑脚，避免只截取图像底部而遗漏失败部位。它是固定上下文窗口，不是精确的人体语义标注。所有原生图像与局部面板均保留，指标沿用作者前景掩码，因此仍包含手持物及掩码误差。

[机位00连续回放](../outputs/dynamic-human-actor23-geodesic-107/continuity/camera00-native-continuity.mp4)与[机位13连续回放](../outputs/dynamic-human-actor23-geodesic-107/continuity/camera13-native-continuity.mp4)各包含25个连续帧，以5帧/秒慢放；前者全部为外观留出，后者包含训练帧和留出帧。视频为压缩观看副本。

[结果与适用范围](../outputs/dynamic-human-actor23-geodesic-107/final-summary.json)、[实际看图范围](../outputs/dynamic-human-actor23-geodesic-107/visual-review.json)。本轮未进行静态背景联合训练或OmniRe场景整合。
''';p.write_text(s,encoding='utf8')
files=['final-summary.json','visual-review.json','lower-body-review/camera00-frame000601-lower-body.png','comparison/camera00-frame000601-full-person.png','comparison/camera13-frame000613-crouching_hands_object_context.png','continuity/camera00-native-continuity.mp4','continuity/camera13-native-continuity.mp4']
p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert O.name+'/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if 'dynamic-human-actor23-independent-106/${name}' in line)+1;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+O.name+'/${name}`, \'局部运动连接与高斯绑定对照\']),\n');p.write_text(''.join(lines),encoding='utf8')
(O/'published-resources.json').write_text(json.dumps(dict(paths={O.name:files},licensed_models_exported=False),indent=2)+'\n',encoding='utf8');print('Theory and inspected native figures added to existing chapter;7explicit public resources.')
