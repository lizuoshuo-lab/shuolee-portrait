"""Optional genuine CPU SSIM/AlexLPIPS recalculation after real1404 audit.

No CUDA calls/renderers/training; source GPU floats and masks must already pass
the independent posterior audit. CPU/backend numeric differences are reported,
not hidden by an arbitrary tolerance or substituted into original results.
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUTROOT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(v,m):
    if not v:raise RuntimeError(m)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-run',required=True);p.add_argument('--evaluation-run',required=True);p.add_argument('--posterior-audit',required=True);a=p.parse_args()
    require(a.audit_run.replace('-','').isalnum() and a.evaluation_run.replace('-','').isalnum(),'Invalidrun')
    output=OUTROOT/a.audit_run;output.mkdir(exist_ok=False);start=time.perf_counter();outcome=dict(status='running_CPU_perceptual',CUDA_called=False,actual1404_posterior_completion_not_established_by_this_script=True)
    save(output/'outcome.json',outcome);(output/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    try:
        prior=read(a.posterior_audit);require(prior['status']=='actual1404_complete_independent_CPU_posterior_audit_passed' and prior['actual_verified_frames']==1404 and prior['evaluation_run']==a.evaluation_run,'Actual1404 completed posterior audit unavailable; no perceptual substitute')
        source=OUTROOT/a.evaluation_run;require(digest(source/'metrics.json')==prior['source_overall_metrics_sha256'],'Actualoverall metrics changed afterposterior audit')
        proto=read(source/'protocol.json')
        with np.load(proto['independent_input_npz_path'],allow_pickle=False) as z:gt=z['rgb'].copy();masks=z['mask'].copy();names=z['names'].tolist()
        require(names==proto['test_names'] and len(names)==117,'Actual117 inputorder differs')
        weights=read(source/'merged-post/tau3/metrics.json')['weight_files']
        for w in weights:require(Path(w['path']).is_file() and digest(w['path'])==w['sha256'],'Missing/changed local fixed metricweights; no downloads')
        import torch
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        require(importlib.metadata.version('torch')=='2.7.1+cu128' and importlib.metadata.version('torchmetrics')=='1.7.1','Fixed CPUmetricruntime differs')
        require((Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth').is_file(),'LocalAlexNetcheckpoint missing; prohibit download')
        ssim=StructuralSimilarityIndexMeasure(data_range=1.).to('cpu').eval();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).to('cpu').eval();rows=[]
        for model in ('merged-unoptimized','merged-post'):
            for setting in ('leaves','tau1','tau3','tau5','tau6','tau15'):
                folder=source/model/setting;reported=read(folder/'per-image-metrics.json');require([r['name'] for r in reported]==names,'Actualmetricorder differs')
                for j,r in enumerate(reported):
                    pred=np.load(folder/'float-frames'/f'{j:02d}-{Path(names[j]).stem}.npy',allow_pickle=False)
                    pe=pred*masks[j];ge=(gt[j].astype(np.float32)/255.*masks[j])*masks[j]
                    pt=torch.from_numpy(pe).permute(2,0,1)[None].to('cpu');target=torch.from_numpy(ge).permute(2,0,1)[None].to('cpu')
                    with torch.no_grad():sv=float(ssim(pt,target));lv=float(lpips(pt,target))
                    ssim.reset();lpips.reset();require(np.isfinite(sv+lv),'ActualCPUmetricnonfinite')
                    rows.append(dict(model=model,setting=setting,name=names[j],CPU_SSIM=sv,CPU_AlexLPIPS=lv,reported_GPU_SSIM=r['ssim_torchmetrics'],reported_GPU_AlexLPIPS=r['lpips_alex_torchmetrics'],CPU_minus_GPU_SSIM=sv-r['ssim_torchmetrics'],CPU_minus_GPU_AlexLPIPS=lv-r['lpips_alex_torchmetrics']))
                    del pt,target
                save(output/'actual-perceptual-progress.json',dict(status='actual_partial_CPU_results_no_overall_completion',count=len(rows),rows=rows))
        require(len(rows)==1404,'ActualCPUperceptualcount incomplete')
        result=dict(status='actual1404_CPU_SSIM_AlexLPIPS_recalculation_completed',device='CPU',weights=weights,per_image=rows,max_abs_CPU_GPU_SSIM_delta=max(abs(r['CPU_minus_GPU_SSIM']) for r in rows),max_abs_CPU_GPU_AlexLPIPS_delta=max(abs(r['CPU_minus_GPU_AlexLPIPS']) for r in rows),fixed_mask_chain='GTm2/predm unchanged',interpretation='ActualCPU/backend deltas reported. No fixedtolerance fabricated and originalGPUmetrics notmodified; this recomputationalone isnotposteriorcompletion.',CUDA_called=False)
        save(output/'perceptual-recalculation.json',result);outcome.update(status=result['status'],elapsed_seconds=time.perf_counter()-start,source_SHA=digest(__file__))
        save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True)
    except Exception as error:
        outcome.update(status='failed_preserved_CPU_perceptual_no_substitute',error=str(error),traceback=traceback.format_exc(),Torch_imported='torch' in sys.modules,source_SHA=digest(__file__));save(output/'outcome.json',outcome);raise
if __name__=='__main__':main()
