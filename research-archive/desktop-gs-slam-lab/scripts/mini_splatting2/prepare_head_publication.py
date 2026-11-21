"""Prepare a separate image-first report before looking at head-candidate results."""
import ast
import hashlib
import json
import time
from pathlib import Path

S = Path(__file__).parent
ROOT = S.parents[1]/'outputs/mini-splatting2-mip-depth-001'


def main():
    generated = []
    for old, new in [('render_native_late_paths.py','render_native_head_paths.py'),
                     ('build_late_report.py','build_head_report.py'),
                     ('append_late_observed_report.py','append_head_observed_report.py'),
                     ('run_late_publication.py','run_head_publication.py'),
                     ('native_late_report.html','native_head_report.html')]:
        text = (S/old).read_text(encoding='utf-8')
        text = text.replace('native-late','native-head').replace('native_late','native_head')
        text = text.replace('additional-frame-late-006','additional-frame-head-007')
        text = text.replace('render_native_late_paths','render_native_head_paths')
        text = text.replace('build_late_report','build_head_report').replace('append_late_observed_report','append_head_observed_report')
        text = text.replace("'late-'+Path(script).stem", "'head-'+Path(script).stem")
        text = text.replace('frozen_before_late_results','frozen_before_head_results')
        text = text.replace('Original schedule versus late densification; both trained at1514x2694',
                            'Original normalized RGB L1 versus training-head coefficient2; both trained at1514x2694')
        text = text.replace('index-before-late.html','index-before-head.html')
        text = text.replace('Original schedule','Original L1').replace('Late densification','Head L1 x2')
        if new.endswith('.html'):
            old_intro = '两组都使用相同的 1514 × 2694 图像、相机、稀疏点和 18,000 步训练。原流程在 3,000 步结束增密；候选保留此前流程，在 3,000–5,000 步继续执行基于梯度的克隆、分裂与剪枝。运动遮罩、Mip 滤波和深度损失保持一致。'
            new_intro = '两组都使用相同的 1514 × 2694 图像、相机、稀疏点和 18,000 步训练。候选把训练照片中已核对的头部矩形区域的 RGB L1 权重设为 2，其他有效像素为 1，再按总权重归一化。Mip 滤波、SSIM、深度约束、运动遮罩和原始增密时机保持一致。'
            assert old_intro in text; text = text.replace(old_intro,new_intro)
            old_method = '本轮仅延长普通增密与剪枝窗口，未延长积极克隆；仍在 8,000 步进行第二次简化。新增高斯使模型容量与耗时可能变化，因此不属于固定模型大小的比较。高分辨率阶段从训练相机取样，首轮简化后重置增密统计的模糊标记。评测保留训练时三维滤波，二维滤波由共同输出尺寸决定。'
            new_method = '头部区域来自 51 张训练照片的固定人工矩形，包含部分颈部和背景，不是精确分割，也不读取评测照片。权重归一化会重新分配头部与其他像素的相对贡献；增密仍响应新的梯度，因此高斯数量可以改变。这不是固定容量的比较。中性权重 1 的损失与图像梯度通过逐位一致性检查；原内核的高斯参数梯度重复性波动仍未解决，不能声称整个训练数值严格等价。'
            assert old_method in text; text = text.replace(old_method,new_method)
            text = text.replace('本轮没有加入深度分散损失、头部加权、超分辨率或扩散生成图像。',
                                '本轮只增加头部 RGB L1 加权，没有加入深度分散损失、延长增密、超分辨率或扩散生成图像。')
            text = text.replace('高分辨率阶段继续增密，人物细节是否改善？','把更多优化权重分给头部，能改善真实人物细节吗？')
            text = text.replace('高分辨率阶段增密','头部区域加权')
            text = text.replace('href="native-resolution.html">上一轮分辨率对照','href="native-late.html">上一轮增密对照')
        replacements = {'高清阶段增密与人物细节':'头部区域加权与人物细节',
                        '原增密时机':'原始 RGB L1','延长增密窗口':'头部权重 ×2',
                        '原流程模型':'原始损失模型','延长增密模型':'头部加权模型'}
        for before, after in replacements.items(): text = text.replace(before,after)
        assert 'native-late-full' not in text and 'Late densification' not in text
        destination=S/new; assert not destination.exists()
        if new.endswith('.py'): ast.parse(text)
        destination.write_text(text,encoding='utf-8'); generated.append(destination)
    result=dict(status='prepared_before_candidate_results',created=time.time(),
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in generated},
                runs={'low':'native-resolution-cache-full-18000-s0','full':'native-head-full-18000-s0'},
                training_or_prior_reports_modified=False)
    target=ROOT/'native-head-publication-sources-001.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
