"""Actual coarse117/geometry8 CPU metric reference and future real1404 wrapper.

All tensors/module parameters explicitly CPU. No CUDA query/call/rendering,
downloads, process control, or changes to source coarse/evaluation artifacts.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,hashlib,importlib.metadata,importlib.util,json,os,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];OUTROOT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
COARSE=OUTROOT/'small-city-local-coarse-003-flat-heldout-001'
GEOMETRY=OUTROOT/'small-city-boundary-selection-CPU-002/selection.json'
INPUT_SHA='60ab56f705c6db347b00f821b08a44e49c0d2dca4f0f06e9a978b9c73c8bce25'
METRICS_SHA='17035e5e1f83577962c933d3a924a5f39d8cf84438d2a862f9d69847f68442dc'
OPTIONAL=OUTROOT/'city-hierarchy-posterior-auditor-CPU-001/recompute_perceptual_cpu.py'
OPTIONAL_SHA='4b318c7362c35cce5087605ac7c1902fb0697c67185c039e3150676cfd3a3181'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(v,m):
    if not v:raise RuntimeError(m)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def cpu_memory():
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE
    p=ctypes.WinDLL('psapi',use_last_error=True);p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];p.GetProcessMemoryInfo.restype=wintypes.BOOL
    c=Counters();c.cb=ctypes.sizeof(c)
    if not p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=int(c.WorkingSetSize),peak_working_set_bytes=int(c.PeakWorkingSetSize),private_usage_bytes=int(c.PrivateUsage),peak_commit_bytes=int(c.PeakPagefileUsage),scope='Actualcurrent-process Windows counters; lifetime peak includes imports/weights/inputs/metric states, no GPU memory query')
def initialize_cpu(weights,threads):
    for w in weights:require(Path(w['path']).is_file() and digest(w['path'])==w['sha256'],'Actualfixedweight missing/changed; no download')
    os.environ['TORCH_HOME']=str(ROOT/'.cache/large_gs_torch')
    import torch
    require(importlib.metadata.version('torch')=='2.7.1+cu128' and importlib.metadata.version('torchmetrics')=='1.7.1' and importlib.metadata.version('torchvision')=='0.22.1+cu128','FixedCPUmetricversions differ')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    def reject_download(*args,**kwargs):raise RuntimeError('Downloads prohibited inCPU reference')
    torch.hub.download_url_to_file=reject_download
    require((Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth').resolve()==Path(weights[0]['path']).resolve(),'ActualTorch cache is not fixed author metric checkpoint')
    return torch
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-run',required=True);p.add_argument('--scope',choices=['geometry8','coarse117','own1404'],required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--evaluation-run');p.add_argument('--posterior-audit');a=p.parse_args()
    require(a.audit_run.replace('-','').isalnum() and len(a.audit_run)<=100 and 1<=a.threads<=8,'InvalidCPUrun/thread budget')
    output=OUTROOT/a.audit_run;output.mkdir(exist_ok=False);start=time.perf_counter();outcome=dict(status='running_CPU_reference',started_utc=now(),CUDA_called=False,downloads=False,source_SHA=digest(__file__),scope=a.scope,threads=a.threads,batch_size=1,actual1404_completed=False)
    save(output/'outcome.json',outcome);(output/'CPU-reference-source.py').write_bytes(Path(__file__).read_bytes())
    try:
        metrics=read(COARSE/'metrics.json');require(digest(COARSE/'metrics.json')==METRICS_SHA,'OriginalGPUmetric bytes changed')
        if a.scope=='own1404':
            require(a.evaluation_run and a.posterior_audit,'Future1404 requiresactual evaluation andposterior')
            prior=read(a.posterior_audit);require(prior['status']=='actual1404_complete_independent_CPU_posterior_audit_passed' and prior['actual_verified_frames']==1404 and prior['evaluation_run']==a.evaluation_run,'Realcompleted1404 unavailable; coarse cannot substitute')
            require(digest(OPTIONAL)==OPTIONAL_SHA,'Frozenoptional CPU recalculator changed')
            initialize_cpu(metrics['weight_files'],a.threads)
            spec=importlib.util.spec_from_file_location('fixed_1404_CPU_recalculator',OPTIONAL);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            ownrun=a.audit_run+'-actual1404';sys.argv=[str(OPTIONAL),'--audit-run',ownrun,'--evaluation-run',a.evaluation_run,'--posterior-audit',a.posterior_audit];module.main()
            outcome.update(status='completed_future_actual1404_CPU_perceptual_wrapper',actual_perceptual_output=str(OUTROOT/ownrun),frozen_optional_source_SHA=OPTIONAL_SHA,actual1404_posterior_completion_provided_by_separate_audit=True)
        else:
            require(not a.evaluation_run and not a.posterior_audit,'Coarse modecannot take dense substitution')
            proto=read(COARSE/'protocol.json');proof=read(COARSE/'completion-audit.json');require(proof['status']=='complete117_CPU_artifact_metric_and_terminal_audit_passed' and proof['metrics_sha256']==METRICS_SHA,'Actualcoarse completion unavailable')
            require(digest(proto['evaluated_input_npz_path'])==INPUT_SHA==proof['independent_input_npz_sha256'],'Actualfrozen117inputs differ')
            with np.load(proto['evaluated_input_npz_path'],allow_pickle=False) as z:gt=z['rgb'].copy();masks=z['mask'].copy();names=z['names'].tolist()
            require(names==proto['test_names'] and len(names)==117,'Actual117order differs')
            geometry=read(GEOMETRY);require(digest(GEOMETRY)=='6b865fca8a94c90a3b8e69902c6dd67bb93bee0cf85b7d447e472ee98bdfde88','Fixedgeometry002 selection changed')
            selected=geometry['selected_names'] if a.scope=='geometry8' else names
            config=dict(scope=a.scope,selected_names=selected,selection_rule='Frozen geometry002 8 names; no RGB/metric ranking' if a.scope=='geometry8' else 'All117 canonical frozen names, originalorder',threads=a.threads,interop_threads=1,batch_size=1,source_metrics_SHA=METRICS_SHA,source_input_SHA=INPUT_SHA,source_protocol_SHA=digest(COARSE/'protocol.json'),source_completion_SHA=digest(COARSE/'completion-audit.json'),geometry_selection_SHA=digest(GEOMETRY),ssim_protocol=metrics['ssim_protocol'],lpips_protocol=metrics['lpips_protocol'],weights=metrics['weight_files'],mask='GTm2/predm, unchangedfloat32mask',CUDA_called=False,actual1404_completed=False)
            save(output/'configuration.json',config)
            torch=initialize_cpu(metrics['weight_files'],a.threads)
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            ssim=StructuralSimilarityIndexMeasure(data_range=1.).to('cpu').eval();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to('cpu').eval()
            require(all(t.device.type=='cpu' for module in (ssim,lpips) for t in list(module.parameters())+list(module.buffers())),'Metricparameter/buffer not CPU')
            reported={r['name']:r for r in metrics['per_image']};checked={r['name']:r for r in proof['checked_frames']};rows=[];metric_start=time.perf_counter();memory_initial=cpu_memory()
            for name in selected:
                j=names.index(name);path=COARSE/'flat/float-frames'/f'{j:03d}-{Path(name).stem}.npy'
                require(path.is_file() and digest(path)==checked[name]['float_frame_sha256'],'ActualGPUfloatframe missing/changed')
                pred=np.load(path,allow_pickle=False);require(pred.dtype==np.float32 and pred.shape==(172,256,3) and np.isfinite(pred).all(),'Actualpredictions invalid')
                pe=pred*masks[j];ge=(gt[j].astype(np.float32)/255.*masks[j])*masks[j]
                pt=torch.from_numpy(pe).permute(2,0,1)[None].to('cpu');target=torch.from_numpy(ge).permute(2,0,1)[None].to('cpu');require(pt.device.type==target.device.type=='cpu','InputnotCPU')
                began=time.perf_counter()
                with torch.no_grad():sv=float(ssim(pt,target));lv=float(lpips(pt,target))
                ssim.reset();lpips.reset();elapsed=time.perf_counter()-began;require(np.isfinite(sv+lv),'ActualCPUperceptualnonfinite')
                old=reported[name];sd=sv-old['ssim_torchmetrics'];ld=lv-old['lpips_alex_torchmetrics'];memory=cpu_memory()
                rows.append(dict(name=name,ordinal=j,actual_float_frame_path=str(path),actual_float_frame_sha256=digest(path),CPU_SSIM=sv,CPU_AlexLPIPS=lv,reported_GPU_SSIM=old['ssim_torchmetrics'],reported_GPU_AlexLPIPS=old['lpips_alex_torchmetrics'],CPU_minus_GPU_SSIM=sd,CPU_minus_GPU_AlexLPIPS=ld,SSIM_float32_ULP_delta=abs(sd)/float(np.spacing(np.float32(old['ssim_torchmetrics']))),LPIPS_float32_ULP_delta=abs(ld)/float(np.spacing(np.float32(old['lpips_alex_torchmetrics']))),CPU_metric_seconds=elapsed,memory=memory))
                save(output/'actual-per-image.json',rows);print(json.dumps(dict(name=name,count=len(rows),seconds=elapsed,SSIM_delta=sd,LPIPS_delta=ld)),flush=True);del pt,target
            require(len(rows)==(8 if a.scope=='geometry8' else 117),'ActualCPUreference incomplete')
            result=dict(status='actual_coarse_CPU_perceptual_reference_completed_not1404',actual_views=len(rows),scope=a.scope,selected_names=selected,per_image=rows,
                CPU_SSIM_mean=float(np.mean([r['CPU_SSIM'] for r in rows])),GPU_SSIM_mean=float(np.mean([r['reported_GPU_SSIM'] for r in rows])),CPU_AlexLPIPS_mean=float(np.mean([r['CPU_AlexLPIPS'] for r in rows])),GPU_AlexLPIPS_mean=float(np.mean([r['reported_GPU_AlexLPIPS'] for r in rows])),
                max_abs_SSIM_delta=max(abs(r['CPU_minus_GPU_SSIM']) for r in rows),max_abs_AlexLPIPS_delta=max(abs(r['CPU_minus_GPU_AlexLPIPS']) for r in rows),max_SSIM_float32_ULPs=max(r['SSIM_float32_ULP_delta'] for r in rows),max_AlexLPIPS_float32_ULPs=max(r['LPIPS_float32_ULP_delta'] for r in rows),
                total_CPU_metric_seconds=sum(r['CPU_metric_seconds'] for r in rows),elapsed_metric_loop_including_IO_seconds=time.perf_counter()-metric_start,initial_memory=memory_initial,final_memory=cpu_memory(),
                timing_scope='CPU metrics forward+perimage reset; masking/conversion/filesaves/inputhashes excluded fromperimage timer, includedinloop elapsed; moduleimports/weightsetup onlyincludedinwholeprocess elapsed',
                numeric_comparison='Actualfloat32CPU/CUDA backend deltas andULPs measured; notbitwiseequivalence oranalyticalerrorbound; no automatic arbitrarytolerance substitutedforreportedGPU metrics',threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),batch_size=1,CUDA_called=False,downloads=False,actual1404_completed=False)
            save(output/'metrics-CPU-reference.json',result);outcome.update(status=result['status'],actual_views=len(rows))
        outcome.update(ended_utc=now(),elapsed_seconds=time.perf_counter()-start,final_memory=cpu_memory());save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as error:
        outcome.update(status='failed_preserved_CPU_reference',error=str(error),traceback=traceback.format_exc(),ended_utc=now(),elapsed_seconds=time.perf_counter()-start,Torch_imported='torch' in sys.modules);save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
