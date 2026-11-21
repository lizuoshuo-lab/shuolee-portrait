"""Publish completed, independently checked local renderer control into K18."""
import datetime, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(r'C:\Users\Administrator\Documents\New project\engine-home')
RUN = ROOT/'outputs/gaussian-global-gauge-renderer-controls-001'
AUDIT = ROOT/'outputs/gaussian-global-gauge-renderer-independent-raw-audit-001'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def main():
    assert sha(RUN/'metrics.json')=='b5019e63b46bb87371e41f3b9e2c6dfb81e1df4b98fb17314074283ce2daa4de'
    assert sha(AUDIT/'metrics.json')=='b83ee1a4370c15acc4142526f3b1eb5ad7d2d0764a883ddd4214fcd6920cbadc'
    actual, audit = read(RUN/'metrics.json'), read(AUDIT/'metrics.json')
    assert actual['render_calls']==2 and actual['GT_reads']==actual['optimizer_steps']==actual['new_training_steps']==0
    assert actual['fixed_input_before']==actual['fixed_input_after']
    assert audit['all_predeclared_numerical_gates_passed'] and audit['actual_float32_array_count']==6
    assert audit['maximum_independent_numeric_record_difference']==0
    assert all(g['passed'] for g in actual['comparison']['all_numerical_checks'])
    completion=read(RUN/'completion.json')
    for name,item in completion['all_saved_files'].items():
        p=RUN/name; assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
    review=RUN/'root_source_and_result_review.json'
    assert not review.exists()
    write(review,{'status':'passed_root_completed_renderer_and_independent_raw_array_review',
        'author_source_sha256':actual['source_sha256']['scripts/check_gaussian_global_gauge_renderer.py'],'protocol_sha256':actual['protocol_sha256'],
        'actual_metrics_sha256':sha(RUN/'metrics.json'),'independent_raw_audit_sha256':sha(AUDIT/'metrics.json'),
        'render_calls':2,'all_predeclared_numerical_gates_passed':True,'six_raw_arrays_independently_checked':True,
        'actual_images_visually_reviewed':['original.png','global_gauge.png'],
        'images_are_model_predictions_not_sensor_RGB':True,'original_raw_and_display_images_not_byte_identical':True,
        'saved_files_checked':len(completion['all_saved_files']),
        'scope':'One complete 23838-Gaussian frozen map, one fixed saved camera, one common global SE3 gauge transform. No GT or optimization, no accuracy/SH/local repair/loop closure claim.',
        'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    knowledge=ROOT/'knowledge/18_gaussian_map_pose_correction.md'
    text=knowledge.read_text(encoding='utf-8')
    text=text.replace('status: "synthetic_matrix_controls_and_independent_math_review_passed"',
                      'status: "CPU_math_and_fixed_map_GPU_render_controls_independently_checked"',1)
    text=text.replace('本页的实测来自一次确定性 float64 CPU 矩阵控制：', '本页第一组实测来自一次确定性 float64 CPU 矩阵控制：',1)
    text=text.replace('“负例通过”表示检出了预设错误，不表示错误路径可接受。',
        '“负例通过”表示检出了预设错误，不表示错误路径可接受。第二组实测使用已有完整地图做两次 GPU 渲染，结果和适用范围单独列在后文。',1)
    section='''## 实测：完整地图统一 gauge 的两次渲染

矩阵恒等式之外，还要检查实际光栅化路径。这里使用已完成的 desk V4 prior0 地图，保留全部 **23,838 个高斯**，固定保存的第 548 帧估计 w2c 与 160×120 内参。地图 checkpoint 的 SHA256 为 `64cda836…44d82`；原相机直接使用保存的 float32 字节，没有重新求逆或修正它的轻微非正交误差。

对全部高斯施加同一个世界变换：旋转 `Rz(23°) Ry(−11°) Rx(17°)`，平移 `[0.23,−0.17,0.09] m`。均值与四元数按上文规则更正，相机使用 `W_new = W_old inverse(Delta)`。log scales、颜色 logits、opacity logits 保持原字节；两次实际 GPU 渲染共享同一组激活后的尺度、常量 RGB 与 opacity 缓冲区。

唯一实际运行调用 gsplat 1.5.3 的 `RGB+ED` 两次，`packed=false`、near/far 为 0.1/6 m、黑色背景。没有优化步骤、GT 读取、原始传感器读取、新训练或失败重试。两次输出为：

![固定原地图与相机的模型预测](../outputs/gaussian-global-gauge-renderer-controls-001/original.png)

![共同世界变换之后的模型预测](../outputs/gaussian-global-gauge-renderer-controls-001/global_gauge.png)

这些都是模型预测图，不是传感器 RGB。展示 PNG 经过裁剪和 8 bit 量化；下表由未裁剪 float32 原始数组转 float64 后相减得到，不能靠“图片看起来一样”代替检查。

| 检查量 | 最大绝对差 | RMSE | 比较分母 |
|---|---:|---:|---|
| RGB，无量纲 | 1.06096e−5 | 8.63874e−7 | 57,600 个标量 |
| alpha，无量纲 | 7.62939e−6 | 5.39803e−7 | 19,200 个像素 |
| 共同正深度支持下的 ED，m | 1.29938e−5 | 5.17025e−7 | 19,200 个像素 |
| 共同覆盖下的 ED，m | 1.29938e−5 | 5.17277e−7 | 19,172 个像素 |

支持掩码要求 ED 有限且为正、`alpha>1e−6`；覆盖掩码要求 ED 有限且为正、`alpha>0.5`。有限 ED、正深度支持、覆盖三种掩码的 XOR 都是 **0 像素**；两种共同掩码均非空。深度最大差约为 **0.013 mm**，它是两次预测的差，不能解释为对真实场景的测深精度。

运行前固定的 11 项 GPU 门限全部通过：[原判据与配置](../outputs/gaussian-global-gauge-renderer-cpu-preparation-002/protocol.json)、[完整数值比较](../outputs/gaussian-global-gauge-renderer-controls-001/comparison.json)。RGB/alpha 的最大差门限为 1e−3、RMSE 门限为 1e−5；有限 ED XOR 必须为零，支持/覆盖 XOR 上限各 4；支持 ED 最大差/RMSE 上限为 0.01/1e−4 m，覆盖为 0.005/5e−5 m。没有根据结果放宽门限。

[控制源码](../scripts/check_gaussian_global_gauge_renderer.py)及[实际终态](../outputs/gaussian-global-gauge-renderer-controls-001/metrics.json)记录两次调用、原参数和相机的前后哈希与 CPU 几何检查；[completion](../outputs/gaussian-global-gauge-renderer-controls-001/completion.json)绑定八份归档文件。原始两份 NPZ 留在桌面实验目录，没有复制到网站。

[独立 SOURCE/CPU 审阅](../outputs/gaussian-global-gauge-renderer-independent-source-review-002/metrics.json)使用另一条 SciPy、Hamilton 乘法与通用逆矩阵路线；[其源码](../scripts/review_gaussian_global_gauge_renderer.py)没有执行作者控制。实际渲染后，[独立数组检查源码](../scripts/audit_gaussian_global_gauge_raw.py)读取六个原始数组，以 float64 相减与 `math.fsum` 重算 RMSE；[独立结果](../outputs/gaussian-global-gauge-renderer-independent-raw-audit-001/metrics.json)核对形状、字节哈希、所有分母、掩码和 11 个判据，重算记录与原记录差为 0。[主代理终态复核](../outputs/gaussian-global-gauge-renderer-controls-001/root_source_and_result_review.json)另行记录文件与两图核对。

这支持的是**同一个固定地图、相机、共同 SE3 变换下的 float32 渲染一致性**。只验证一个视角，没有证明逐锚点变形后的地图重叠一致、世界位置准确、SH 输运、回环、联合 BA 或完整 SLAM。全局 gauge 保持观测与修复不同轨迹段的漂移是两个问题。

'''
    assert '## 实测：完整地图统一 gauge 的两次渲染' not in text
    text=text.replace('## 源码和证据范围',section+'## 源码和证据范围',1)
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if '://' in link: continue
        p=(knowledge.parent/link).resolve();assert p.is_relative_to(ROOT) and p.is_file(),str(p)
        if p.suffix!='.md': assert p.stat().st_size<1048576,str(p)
    knowledge.write_text(text,encoding='utf-8',newline='\n')
    progress_path=ROOT/'manifests/progress.json'; progress=read(progress_path)
    assert not any(x['id']=='gaussian-global-gauge-renderer-controls-001' for x in progress['results'])
    progress['results'].append({'id':'gaussian-global-gauge-renderer-controls-001','title':'完整高斯地图共同世界变换的实际渲染控制',
        'status':actual['status'],'metrics':{'render_calls':2,'Gaussians':23838,'passed_numerical_gates':11,
        'comparison':actual['comparison'],'optimizer_steps':0,'GT_reads':0,'independent_raw_array_count':6},
        'artifacts':['knowledge/18_gaussian_map_pose_correction.md','outputs/gaussian-global-gauge-renderer-controls-001/metrics.json',
        'outputs/gaussian-global-gauge-renderer-controls-001/comparison.json','outputs/gaussian-global-gauge-renderer-independent-raw-audit-001/metrics.json'],
        'scope':'固定已有地图与单个估计相机，共同SE3世界变换的两次GPU预测比较；不证明世界精度、逐锚点地图修复、SH、回环、BA或完整SLAM。无训练、GT、重试。'})
    progress['current_stage']='completed_full_RGBD_controls_and_fixed_map_global_gauge_renderer_independent_checks'
    write(progress_path,progress)
    sync_path=SITE/'scripts/sync-gs-lab.mjs'; sync=sync_path.read_text(encoding='utf-8')
    extras=['scripts/check_gaussian_global_gauge_renderer.py','scripts/review_gaussian_global_gauge_renderer.py',
        'scripts/audit_gaussian_global_gauge_raw.py','outputs/gaussian-global-gauge-renderer-cpu-preparation-002/protocol.json',
        'outputs/gaussian-global-gauge-renderer-independent-source-review-002/metrics.json',
        'outputs/gaussian-global-gauge-renderer-controls-001/comparison.json',
        'outputs/gaussian-global-gauge-renderer-controls-001/completion.json',
        'outputs/gaussian-global-gauge-renderer-controls-001/root_source_and_result_review.json']
    additions=''.join('  ['+json.dumps(p,ensure_ascii=False)+', "完整地图共同世界变换的已核对证据"],\n' for p in extras if '"'+p+'"' not in sync and "'"+p+"'" not in sync)
    sync=sync.replace('const referencedExtras = new Map([\n','const referencedExtras = new Map([\n'+additions,1)
    sync_path.write_text(sync,encoding='utf-8',newline='\n')
    print(json.dumps({'status':'published_independently_checked_renderer_control_into_K18','knowledge18_sha256':sha(knowledge),'progress_results':len(progress['results']),'root_review_sha256':sha(review)}))

if __name__=='__main__': main()
