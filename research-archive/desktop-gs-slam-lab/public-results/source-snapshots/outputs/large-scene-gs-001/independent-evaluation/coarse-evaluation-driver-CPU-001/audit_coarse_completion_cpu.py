"""CPU-only terminal/artifact/metric audit and report for actual coarse117 eval.

Run after evaluate_coarse.py has really exited. No Torch import or CUDA query.
Preserve existing artifacts; refuse incomplete/invalid results without averaging.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001'
OUTROOT=BASE/'independent-evaluation'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def arr_digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(v,m):
    if not v:raise RuntimeError(m)
def save(path,v):
    require(not path.exists(),'Existing completion artifact: '+str(path))
    path.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation-run',required=True);a=p.parse_args()
    require(a.evaluation_run.replace('-','').isalnum(),'Invalid run')
    output=OUTROOT/a.evaluation_run
    outcome=read(output/'outcome.json');protocol=read(output/'protocol.json');metrics=read(output/'metrics.json');timing=read(output/'timings.json')
    renders=read(output/'per-frame-render-audit.json');rows=metrics['per_image'];names=protocol['test_names']
    require(outcome['status']==metrics['status']=='complete117_independent_local_coarse_flat_heldout' and outcome['cuda_used'] is True,'Actual GPU evaluation not complete')
    require(len(names)==len(rows)==len(renders)==117 and [r['name'] for r in rows]==[r['name'] for r in renders]==names and len(set(names))==117,'117 order/identity differs')
    require(timing['status']=='complete117_flat' and timing['valid_frames']==117 and all(r['status']=='finite_flat_RGB' for r in renders),'Partial render cannot be pooled')
    # Fresh native process observation is independent of outcome/status files.
    command="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    r=subprocess.run([str(ps),'-NoLogo','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    require(r.returncode==0,'Fresh terminal process query failed')
    processes=json.loads(r.stdout or '[]');processes=processes if isinstance(processes,list) else [processes]
    gpu_eval=[x for x in processes if 'evaluate_coarse.py' in (x.get('CommandLine') or '') and a.evaluation_run in (x.get('CommandLine') or '')]
    trainers=[x for x in processes if any(s in (x.get('CommandLine') or '') for s in ('train_coarse.py','train_single.py','train_post.py'))]
    require(not gpu_eval and not trainers,'Evaluator/official training Python still live; GPU release not verified')
    config_path=output/'bound-training-configuration.json';require(digest(config_path)==protocol['training_configuration_sha256'],'Bound configuration bytes mismatch')
    binding=read(output/'model-binding-CPU.json');require(binding['source_model_sha256']==digest(binding['source_model_path'])==metrics['model_binding']['source_model_sha256'],'Actual read-only source PLY changed')
    for row in read(output/'source-audit.json')['source_files']+read(output/'source-audit.json')['rasterizer_sources']:
        require(digest(row['snapshot'])==row['sha256']==digest(row['path']),'Source snapshot/current identity mismatch')
    for row in read(output/'source-audit.json')['modules']:require(digest(row['path'])==row['sha256'],'Native binary changed')
    snapshot=output/'source-snapshots/evaluation/evaluate_coarse.py';require(digest(snapshot)==outcome['driver_sha256'],'Actual evaluator source snapshot mismatch')
    z=np.load(protocol['evaluated_input_npz_path']);predz=np.load(output/'predictions.npz')
    require(digest(protocol['evaluated_input_npz_path'])==protocol['evaluated_input_npz_sha256'] and z['names'].tolist()==predz['names'].tolist()==names,'Independent input/prediction archive mismatch')
    checked=[];values=[]
    for i,(row,view) in enumerate(zip(rows,protocol['views'])):
        name=f'{i:03d}-{Path(view["name"]).stem}';folder=output/'flat'
        pred=np.load(folder/'float-frames'/f'{name}.npy');require(np.array_equal(pred,predz['rgb'][i]) and np.isfinite(pred).all() and pred.shape==(172,256,3) and pred.min()>=0 and pred.max()<=1,'Invalid actual float prediction')
        gt=z['rgb'][i];mask=z['mask'][i]
        require(arr_digest(mask)==view['mask_float32_sha256'] and np.array_equal(z['K'][i],np.asarray(view['K'])) and np.array_equal(z['c2w'][i],np.asarray(view['C2W'])),'Frozen mask/K/C2W differs')
        for key in ('source_RGB','source_mask','RGB','mask_png'):
            require(digest(view[key+'_path'])==view[key+'_sha256'],'Input original/derived file changed: '+key)
        png=(pred*255).astype(np.uint8)
        with Image.open(folder/'predicted'/f'{name}.png') as im:require(np.array_equal(np.asarray(im),png),'Actual prediction PNG differs')
        with Image.open(folder/'paired'/f'{name}.png') as im:require(np.array_equal(np.asarray(im),np.concatenate([gt,png],axis=1)),'Actual GT/pred pair differs')
        pe=pred*mask;ge=(gt.astype(np.float32)/255.*mask)*mask
        num=float(np.sum((pe.astype(np.float64)-ge.astype(np.float64))**2));den=gt.size;psnr=float(-10*np.log10(num/den))
        require(psnr==row['psnr_numpy_db'] and num==row['RGB_error_numerator'] and den==row['RGB_error_denominator']==132096,'Independent main NumPy PSNR differs')
        w=mask.astype(np.float64)**2;vden=3*float(w.sum());vnum=float(np.sum((pred.astype(np.float64)-gt.astype(np.float64)/255.)**2*w))
        diagnostic=row['symmetric_single_mask_valid_weight_diagnostic'];require(vnum==diagnostic['RGB_error_numerator'] and vden==diagnostic['RGB_error_denominator'],'Auxiliary symmetric valid-mask denominator differs')
        with Image.open(folder/'masked-paired'/f'{name}.png') as im:require(np.array_equal(np.asarray(im),np.concatenate([(ge*255).astype(np.uint8),(pe*255).astype(np.uint8)],axis=1)),'Actual masked pair differs')
        require(digest(row['masked_paired_path'])==row['masked_paired_sha256'] and digest(row['prediction_path'])==row['prediction_sha256'] and all(np.isfinite(row[k]) for k in ('psnr_numpy_db','ssim_torchmetrics','lpips_alex_torchmetrics')),'Invalid metric/file identity')
        values.append(psnr);checked.append(dict(name=row['name'],float_frame_sha256=digest(folder/'float-frames'/f'{name}.npy'),predicted_png_sha256=digest(folder/'predicted'/f'{name}.png'),paired_png_sha256=digest(folder/'paired'/f'{name}.png'),masked_pair_sha256=row['masked_paired_sha256'],independent_NumPy_PSNR_exact=True))
    require(float(np.mean(values))==metrics['psnr_numpy_mean_per_image_db'],'PSNR per-image mean differs')
    for group in metrics['groups']:
        group_rows=[x for x in rows if x['group']==group['group']]
        require(len(group_rows)==group['count'] and [x['name'] for x in group_rows]==group['names'],'Membership group/order differs')
        for source,key in (('psnr_numpy_db','psnr_mean_per_image_db'),('ssim_torchmetrics','ssim_mean'),('lpips_alex_torchmetrics','alex_lpips_mean')):require(float(np.mean([x[source] for x in group_rows]))==group[key],'Group metric mean differs')
    require(sorted(g['count'] for g in metrics['groups'])==[22,95],'Membership count differs')
    audit=dict(status='complete117_CPU_artifact_metric_and_terminal_audit_passed',observed_utc=datetime.now(timezone.utc).isoformat(),evaluation_run=a.evaluation_run,
        actual_eval_python_matching_processes=gpu_eval,actual_official_training_python_matching_processes=trainers,GPU_exclusive_stage_terminal_verified=True,CPU_audit_cuda_called=False,
        expected_count=117,actual_valid_count=117,actual_predicted_PNG=117,actual_RGB_pairs=117,actual_masked_pairs=117,actual_float_frames=117,
        source_model_sha256=binding['source_model_sha256'],driver_source_sha256=outcome['driver_sha256'],bound_configuration_sha256=digest(config_path),
        independent_input_npz_sha256=digest(protocol['evaluated_input_npz_path']),metrics_sha256=digest(output/'metrics.json'),outcome_sha256=digest(output/'outcome.json'),timings_sha256=digest(output/'timings.json'),checked_frames=checked,
        limitations=['Fresh process terminal evidence at this timestamp only.','Flat full-model rendering, no tree/cut/LOD/streaming.','Native rasterizer temporary memory outside Torch allocator remains unknown; synchronized whole-device boundaries are not a transient peak.','Local coarse background training is not equal-budget dense-training causal comparison; published SfM may use heldout photographs.'])
    save(output/'completion-audit.json',audit)
    lines=[f'# SmallCity 本地粗背景模型 117 张固定留出评测\n',f'本次完整完成 117/117。训练 `{binding["training_run"]}`，实际模型 SHA256 `{binding["source_model_sha256"]}`；{binding["total_G"]:,} 个高斯含一次原始前缀 {binding["skybox_prefix_G"]:,} 天空，无追加。源码 SH1，{binding["requested_iterations"]:,} 次循环、{binding["optimizer_update_calls"]:,} 次优化调用按原源码控制流及最终回执推得，非另行计数。\n',
      '采用未修改作者 flat/direct renderer，默认抗锯齿、近面 z>0.2 和深度前向保留；无树、无 LOD。关闭曝光，117 张测试曝光从未拟合。固定 256×172 RGB、真实软 mask、原全局相机 frame；主口径 GT×mask²、预测×mask，误差分母每图为 3WH=132096。辅助对称 mask 指标使用 3Σmask²，单独保存。\n',
      '| 图像组 | 张数 | 平均逐图 PSNR / dB | SSIM | Alex LPIPS |\n|---|---:|---:|---:|---:|',
      f'| 全部固定留出 | 117 | {metrics["psnr_numpy_mean_per_image_db"]:.6f} | {metrics["ssim_mean_per_image"]:.6f} | {metrics["lpips_alex_mean_per_image"]:.6f} |']
    for g in metrics['groups']:lines.append(f'| {g["group"]} | {g["count"]} | {g["psnr_mean_per_image_db"]:.6f} | {g["ssim_mean"]:.6f} | {g["alex_lpips_mean"]:.6f} |')
    t=timing['stages']['official_flat_render_seconds'];whole=max(r['whole_device_after']['whole_device_used_bytes'] for r in renders)
    lines.extend(['\n95/22 分组仅表示作者公开分块相机列表成员资格，不是空间内外，也不证明非成员只有粗背景覆盖。\n',
      f'3 次热身后逐帧 CUDA 同步，原作者 flat 渲染 p50 {t["p50"]*1000:.4f} ms、p95 {t["p95"]*1000:.4f} ms，包含深度前向/屏幕梯度缓冲与抗锯齿，不含解码、传输、静态属性缓存、I/O 或指标。全模型常驻；Torch peak allocated {timing["peak_torch_allocated"]/(1024**3):.4f} GiB / reserved {timing["peak_torch_reserved"]/(1024**3):.4f} GiB；同步渲染后 whole-device 最大已用边界 {whole/(1024**3):.4f} GiB，不能称原生瞬时峰值。\n',
      '粗模型为固定 SfM 位置、SH1 的背景训练；不与密集训练按相同预算作因果比较。117 留出限定于 GS 光度优化和曝光；公开 SfM 标定、位置、颜色可能使用全部照片。与作者预训练城市的 12 张 reconstruction-view pilot 口径不同，不能直接作训练收益比较。\n',
      '每图 float、预测、原 RGB 配对、mask 后配对、指标、分组、原始输入/源码/配置/模型回执已保存。CPU 独立复算全部逐图 PSNR 和辅助误差分母、核验图像/输入/源模型哈希；实际查询评测及官方训练 Python 已终止，GPU 已释放。'])
    report=output/'report.zh-CN.md';require(not report.exists(),'Existing report');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('status','actual_valid_count','GPU_exclusive_stage_terminal_verified','source_model_sha256')},ensure_ascii=False))
if __name__=='__main__':main()
