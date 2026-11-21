"""Independent NumPy/stdlib raw reader; no author compare, renderer or GT import."""
import argparse, hashlib, json, math, traceback
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/'outputs/gaussian-global-gauge-renderer-cpu-preparation-002'
RUN = ROOT/'outputs/gaussian-global-gauge-renderer-controls-001'
OUT = ROOT/'outputs/gaussian-global-gauge-renderer-independent-raw-audit-001'
CPU = ROOT/'outputs/gaussian-global-gauge-renderer-raw-audit-preparation-002'
AUTHOR_SHA = 'f525ffec8f78d0ccd80ac0f565bd04a204092fbf86ab51b3940c47c1bbb3e86f'
PROTOCOL_SHA = '732abff449b619a36b2b907630c26b0f7b36ff9a7834c8456cc5a2b71317939f'
def need(v, s):
    if not v: raise ValueError(s)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, v):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(v, f, indent=2, allow_nan=False)
def record(a):
    a=np.ascontiguousarray(a); return dict(shape=list(a.shape),dtype=str(a.dtype),raw_bytes=a.nbytes,raw_bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest())
def error(a,b,mask=None):
    d=np.subtract(a,b,dtype=np.float64)
    if mask is not None: d=d[mask]
    d=d.reshape(-1); need(np.isfinite(d).all(),'Finite raw differences')
    return dict(scalar_count=len(d),max_abs=float(np.max(np.abs(d))) if len(d) else None,RMSE=math.sqrt(math.fsum(float(x)*float(x) for x in d)/len(d)) if len(d) else None)
def comparison(a,b,cfg):
    need(all(np.isfinite(p[k]).all() for p in (a,b) for k in ('RGB','alpha')),'Finite RGB/alpha')
    f0,f1=np.isfinite(a['ED']),np.isfinite(b['ED'])
    s0,s1=(f0&(a['ED']>0)&(a['alpha']>cfg['ED_support_alpha_strictly_greater'])),(f1&(b['ED']>0)&(b['alpha']>cfg['ED_support_alpha_strictly_greater']))
    c0,c1=(f0&(a['ED']>0)&(a['alpha']>cfg['coverage_alpha_strictly_greater'])),(f1&(b['ED']>0)&(b['alpha']>cfg['coverage_alpha_strictly_greater']))
    masks={f'{key}_{suffix}':int(v.sum()) for key,vs in [('finite',(f0,f1)),('positive_supported',(s0,s1)),('covered',(c0,c1))] for suffix,v in zip(('original','gauged'),vs)}
    masks.update(finite_ED_XOR_pixels=int((f0^f1).sum()),positive_supported_XOR_pixels=int((s0^s1).sum()),positive_supported_common_pixels=int((s0&s1).sum()),covered_XOR_pixels=int((c0^c1).sum()),covered_common_pixels=int((c0&c1).sum()))
    result={name:error(a[k],b[k],mask) for name,k,mask in [('RGB','RGB',None),('alpha','alpha',None),('all_finite_ED_common','ED',f0&f1),('positive_supported_ED_common','ED',s0&s1),('covered_ED_common','ED',c0&c1)]};result['masks']=masks
    values={'RGB_max_abs':result['RGB']['max_abs'],'RGB_RMSE':result['RGB']['RMSE'],'alpha_max_abs':result['alpha']['max_abs'],'alpha_RMSE':result['alpha']['RMSE'],'finite_ED_mask_XOR_pixels':masks['finite_ED_XOR_pixels'],'positive_supported_ED_mask_XOR_pixels':masks['positive_supported_XOR_pixels'],'covered_ED_mask_XOR_pixels':masks['covered_XOR_pixels']}
    for prefix,key in [('common_supported','positive_supported_ED_common'),('common_covered','covered_ED_common')]:
        values[prefix+'_ED_max_abs_m']=result[key]['max_abs'];values[prefix+'_ED_RMSE_m']=result[key]['RMSE']
    need(set(values)==set(cfg['GPU_limits']) and len(values)==11,'All11 predeclared gates')
    result['all_numerical_checks']=[dict(name=k,actual=v,limit=cfg['GPU_limits'][k],passed=v is not None and math.isfinite(v) and v<=cfg['GPU_limits'][k]) for k,v in values.items()]
    return result
def matches(actual,stored):
    need(len(stored['all_numerical_checks'])==len(actual['all_numerical_checks'])==11 and len({g['name'] for g in stored['all_numerical_checks']})==11,'All11 original gate records present without duplicates')
    need(actual['masks']==stored['masks'],'Exact raw mask counts/XOR')
    maximum=0.
    for key in ('RGB','alpha','all_finite_ED_common','positive_supported_ED_common','covered_ED_common'):
        need(actual[key]['scalar_count']==stored[key]['scalar_count'],'Complete scalar denominators')
        for field in ('max_abs','RMSE'):
            x,y=actual[key][field],stored[key][field];need((x is None)==(y is None),'Null iff empty mask')
            if x is not None:
                diff=abs(x-y);need(diff<=16*max(math.ulp(x),math.ulp(y)),'Independent fsum numerical record differs');maximum=max(maximum,diff)
    need(actual['all_numerical_checks']==stored['all_numerical_checks'] or all(x['name']==y['name'] and x['limit']==y['limit'] and x['passed']==y['passed'] for x,y in zip(actual['all_numerical_checks'],stored['all_numerical_checks'])),'All11 original gate decisions retained')
    return maximum
def controls(cfg):
    a={'RGB':np.zeros((2,3,3),np.float32),'ED':np.ones((2,3),np.float32),'alpha':np.ones((2,3),np.float32)}
    need(all(g['passed'] for g in comparison(a,a,cfg)['all_numerical_checks']),'Identical synthetic arrays')
    b={k:v.copy() for k,v in a.items()};b['alpha'][:]=np.float32(.5);r=comparison(b,b,cfg)
    need(r['masks']['covered_common_pixels']==0 and not all(g['passed'] for g in r['all_numerical_checks']),'Strict .5 and empty masks cannot pass')
    b['alpha'][:]=np.float32(1e-6);r=comparison(b,b,cfg);need(r['masks']['positive_supported_common_pixels']==0,'Strict float32 alpha1e-6 threshold')
    b={k:v.copy() for k,v in a.items()};b['ED'][0,0]=np.nan;r=comparison(a,b,cfg)
    need(r['masks']['finite_ED_XOR_pixels']==1 and not all(g['passed'] for g in r['all_numerical_checks']),'Finite XOR rejects missing ED')
    b={k:v.copy() for k,v in a.items()};b['RGB'][:]=.01;need(not all(g['passed'] for g in comparison(a,b,cfg)['all_numerical_checks']),'Predeclared RGB limits reject perturbation')
    need(error(np.array([1.,2.]),np.zeros(2))['RMSE']==math.sqrt(2.5),'Independent fsum RMSE fixed arithmetic')
    return dict(status='passed_tiny_synthetic_array_controls',controls=6,actual_run_reads=0,GPU_calls=0)
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--audit',action='store_true');p.add_argument('--source-sha');p.add_argument('--terminal-sha');args=p.parse_args()
    need(sha(PREP/'protocol.json')==PROTOCOL_SHA and sha(ROOT/'scripts/check_gaussian_global_gauge_renderer.py')==AUTHOR_SHA,'Frozen protocol/source')
    protocol=read(PREP/'protocol.json');cfg=protocol['configuration']
    if args.prepare:
        need(args.source_sha is None and args.terminal_sha is None,'Preparation never reads actual run');CPU.mkdir(exist_ok=False);(CPU/'source_script.py').write_bytes(Path(__file__).read_bytes())
        save(CPU/'CPU_controls.json',controls(cfg));print(json.dumps(dict(status='SOURCE_CPU_prepared_no_actual_run_reads',source_sha256=sha(__file__),CPU_controls_sha256=sha(CPU/'CPU_controls.json'),GPU_calls=0)));return
    need(args.source_sha==sha(__file__) and args.terminal_sha,'Exact root-reviewed reader and actual terminal SHA');OUT.mkdir(exist_ok=False);(OUT/'source_script.py').write_bytes(Path(__file__).read_bytes())
    watched={};
    def watch(path,expected=None):
        d=sha(path);need(expected is None or d==expected,'Consumed file hash');watched[Path(path)]=d
    try:
        terminal=RUN/('failure.json' if (RUN/'failure.json').exists() else 'metrics.json');watch(terminal,args.terminal_sha);state=read(terminal)
        watch(PREP/'protocol.json',PROTOCOL_SHA);watch(ROOT/'scripts/check_gaussian_global_gauge_renderer.py',AUTHOR_SHA);watch(Path(__file__),args.source_sha)
        need(state['protocol_sha256']==PROTOCOL_SHA and state['source_sha256']==protocol['source_sha256'] and state['configuration']==cfg and state['GT_reads']==state['optimizer_steps']==state['new_training_steps']==0,'Terminal declared source/config/scope')
        need(state['status'] in ('completed_global_gauge_render_control_within_predeclared_float32_bounds','failed_global_gauge_render_control_preserved_no_retry'),'Actual terminal status, never active output')
        arms=state['requested_arms'];payloads=[];need(len(arms)==2 and [a['name'] for a in arms]==['original_frozen_map_render','global_gauge_map_camera_render'],'Two predeclared original/gauged arms only')
        for index,arm in enumerate(arms):
            if arm['status']!='render_completed': continue
            name='original.npz' if index==0 else 'global_gauge.npz';need(arm['prediction_path']==name,'Fixed two raw prediction paths');watch(RUN/name,arm['prediction_sha256'])
            with np.load(RUN/name,allow_pickle=False) as z:
                need(set(z.files)=={'RGB','ED','alpha'},'Exactly3 raw arrays per render');arrays={k:z[k].copy() for k in z.files}
            for k,v in arrays.items():need(v.dtype==np.float32 and v.shape==((120,160,3) if k=='RGB' else (120,160)) and record(v)==arm['raw_array_payloads'][k],'Raw float32 shapes/bytes/hash')
            payloads.append(arrays)
        result=dict(status='passed_independent_raw_file_and_numeric_record_audit',reader_source_sha256=sha(__file__),author_source_sha256=AUTHOR_SHA,protocol_sha256=PROTOCOL_SHA,original_terminal_sha256=args.terminal_sha,original_terminal_status=state['status'],original_failure_preserved=terminal.name=='failure.json',original_exception={k:state.get(k) for k in ('exception_type','exception_message','traceback')},actual_raw_prediction_count=len(payloads),actual_float32_array_count=3*len(payloads),GPU_calls=0,GT_reads=0,optimizer_steps=0)
        if len(payloads)==2:
            need(state['render_calls']==2,'Exactly2 render attempts');r=comparison(*payloads,cfg);result['independent_comparison']=r
            if 'comparison' in state:result['maximum_independent_numeric_record_difference']=matches(r,state['comparison'])
            if (RUN/'comparison.json').exists():watch(RUN/'comparison.json');matches(r,read(RUN/'comparison.json'))
            result['all_predeclared_numerical_gates_passed']=all(g['passed'] for g in r['all_numerical_checks'])
        else:need(terminal.name=='failure.json','Only retained original failure permits partial renders');result['all_predeclared_numerical_gates_passed']=None
        if terminal.name=='metrics.json':need(len(payloads)==2 and result['all_predeclared_numerical_gates_passed'] and state['fixed_input_before']==state['fixed_input_after']==protocol['fixed_input'],'Completed original control input/postguard/numerical states')
        need(all(sha(path)==d for path,d in watched.items()),'All read files unchanged');result['consumed_files_unchanged']=len(watched)
        save(OUT/'metrics.json',result);print(json.dumps(dict(status=result['status'],metrics_sha256=sha(OUT/'metrics.json'),arrays=result['actual_float32_array_count'],original_failure_preserved=result['original_failure_preserved'],GPU_calls=0)))
    except BaseException as e:save(OUT/'failure.json',dict(status='independent_readonly_raw_audit_failure_retained',error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),GPU_calls=0));raise
if __name__=='__main__':main()
