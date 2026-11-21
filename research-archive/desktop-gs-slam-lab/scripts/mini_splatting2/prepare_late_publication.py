"""Prepare an image-first matched topology comparison without touching old reports."""
import ast
import hashlib
import json
from pathlib import Path
import time

S=Path(__file__).parent;ROOT=S.parents[1]/'outputs/mini-splatting2-mip-depth-001'
RUNS={'low':'native-resolution-cache-full-18000-s0','full':'native-late-full-18000-s0'}

def create(name,text,python=True):
    path=S/name;assert not path.exists()
    if python:ast.parse(text)
    path.write_text(text,encoding='utf-8');return path

def main():
    created=[]
    text=(S/'render_native_resolution_cache_paths.py').read_text()
    text=text.replace("NAMES={'Low-res training':'native-resolution-cache-low-18000-s0','Full-res training':'native-resolution-cache-full-18000-s0'}",
        "NAMES={'Original schedule':'native-resolution-cache-full-18000-s0','Late densification':'native-late-full-18000-s0'}")
    text=text.replace('native-resolution-cache-paths-001','native-late-paths-001')
    created.append(create('render_native_late_paths.py',text))
    text=(S/'build_resolution_cache_report.py').read_text()
    text=text.replace('def main():',f'RUNS={RUNS!r}\n\ndef main():',1)
    text=text.replace("f'native-resolution-cache-{arm}-18000-s0'","RUNS[arm]")
    text=text.replace("f'native-resolution-cache-{arm}-18000-s0/resolution-evaluation-v2'","(RUNS[arm]+'/resolution-evaluation-v2')")
    text=text.replace("f'native-resolution-cache-{arm}-18000-s0/resolution-evaluation-v2/summary.json'","(RUNS[arm]+'/resolution-evaluation-v2/summary.json')")
    text=text.replace("'native-resolution-cache-low-18000-s0/resolution-evaluation-v2'","'native-resolution-cache-full-18000-s0/resolution-evaluation-v2'")
    text=text.replace("    assert all(m['status']=='complete' for m in models.values())", "    assert all(m['status']=='complete' for m in models.values())\n    assert all(models[arm]['run']==name for arm,name in RUNS.items())")
    text=text.replace('native-resolution-cache-review-001','native-late-review-001').replace('native-resolution-cache-protocol-001','native-late-protocol-001')
    text=text.replace('native-resolution-cache-paths-001','native-late-paths-001').replace('native-resolution-assets','native-late-assets')
    text=text.replace('native_resolution_report.html','native_late_report.html').replace('native-resolution.html','native-late.html')
    text=text.replace('index-before-resolution.html','index-before-late.html').replace('最新对照：训练分辨率与人物细节','最新对照：高清阶段增密与人物细节')
    created.append(create('build_late_report.py',text))
    text=(S/'native_resolution_report.html').read_text(encoding='utf-8')
    replacements={
        'MiniGS2 · 训练分辨率与人物细节':'MiniGS2 · 高分辨率阶段增密',
        '保留视频的原始分辨率，人物能重建得更清楚吗？':'高分辨率阶段继续增密，人物细节是否改善？',
        '两组均从相同的稀疏点和相机开始训练。低分辨率图像直接由校正后的原分辨率图像缩小得到，避免混入视频解码或 JPEG 压缩差异。训练步数、运动遮罩和平均深度约束保持一致。':'两组都使用相同的 1514 × 2694 图像、相机、稀疏点和 18,000 步训练。原流程在 3,000 步结束增密；候选保留此前流程，在 3,000–5,000 步继续执行基于梯度的克隆、分裂与剪枝。运动遮罩、Mip 滤波和深度损失保持一致。',
        'native-spread.html':'native-resolution.html','上一轮深度分散对照':'上一轮分辨率对照',
        '低分辨率训练':'原增密时机','原分辨率训练':'延长增密窗口',
        '低分辨率模型':'原流程模型','原分辨率模型':'延长增密模型',
        'native-resolution-assets':'native-late-assets',
        'Mip 的三维滤波尺度随训练采样率改变，高斯增密与简化也会随输入变化。因此本轮衡量的是完整训练流程对分辨率的响应，不能把全部收益解释成纹理像素增加。评测保留各自训练时的三维滤波，仅按共同输出尺寸计算二维滤波。':'本轮仅延长普通增密与剪枝窗口，未延长积极克隆；仍在 8,000 步进行第二次简化。新增高斯使模型容量与耗时可能变化，因此不属于固定模型大小的比较。高分辨率阶段从训练相机取样，首轮简化后重置增密统计的模糊标记。评测保留训练时三维滤波，二维滤波由共同输出尺寸决定。',
    }
    for old,new in replacements.items():assert old in text,old;text=text.replace(old,new)
    text=text.replace('原增密时机 · 757 × 1347','原增密时机 · 1514 × 2694')
    created.append(create('native_late_report.html',text,False))
    # Reuse existing24frame display with the same frozen query poses/rectangles.
    text=(S/'append_resolution_observed_report.py').read_text()
    text=text.replace('native-resolution-observed-publication-001','native-late-observed-publication-001')
    text=text.replace('additional-frame-resolution-005','additional-frame-late-006')
    text=text.replace('native-resolution-observed-groups-001','native-late-observed-groups-001')
    text=text.replace('native-resolution.html','native-late.html').replace('native-resolution-assets','native-late-assets')
    text=text.replace('低分辨率训练','原增密时机').replace('原分辨率训练','延长增密窗口')
    text=text.replace('低分辨率模型','原流程模型').replace('原分辨率模型','延长增密模型')
    created.append(create('append_late_observed_report.py',text))
    (ROOT/'native-late-publication-sources-001.json').write_text(json.dumps(dict(status='prepared_before_candidate_results',created=time.time(),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in created},runs=RUNS,
        training_or_prior_reports_modified=False),indent=2));print([p.name for p in created])

if __name__=='__main__':main()
