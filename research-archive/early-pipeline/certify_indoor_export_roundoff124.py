"""One read-only export diagnosis, independent of the preserved failed gate."""
import inspect
from pathlib import Path
import diagnose_indoor_circle16_export124 as original
from native_export_roundoff124 import certify
from run_official_gaussians123 import save,sha,now

def main():
    out=original.ROOT/'export-roundoff-certificate124-v2'
    out.mkdir(exist_ok=False)
    snap=original.ROOT/'gaussian-run/step-018000'
    hashes={str(p):sha(p) for p in (snap/'checkpoint.pt',snap/'world.ply',snap/'bound-local.ply',snap/'binding.npz')}
    plan=dict(updated=now(),state='registered_readonly_diagnosis',source_hashes=hashes,
        original_strict_gate=dict(mean_abs_lt=2e-6,max_abs_lt=.005,change_allowed=False),
        check='All centers, scales, complete covariance, opacity and SH0; same CUDA ordered candidates and unique cutoff contributor for every original-gate exception.',
        scope='Explain known reference-camera roundoff; not a new scientific quality gate or pixel-identical export claim.',
        training_resumed=False,optimizer_updates_added=0,source_changed=False,
        helper_sha256=sha(Path(__file__).with_name('native_export_roundoff124.py')))
    plan['criterion_correction']='v1 incorrectly assumed exp(log(s)) always has relative error below8eps, independent of log magnitude. v2 checks exact raw standard log encoding instead. The complete covariance bound and all original strict RGB thresholds are unchanged; v1 failure retained.'
    save(out/'plan.json',plan)
    src=inspect.getsource(original.main)
    prefix=src[src.index('    lock='):src.index('        delta=')]
    prefix=prefix.replace("out=ROOT/'export-failure-diagnostic'","out=ROOT/'export-roundoff-certificate124-v2'")
    prefix=prefix.replace('out.mkdir(exist_ok=False)','assert out.exists()')
    tail='''        result=certify(model,plain,cam,cam,torch,render,pipe,standard_scale_encoding=True)
        identity=certify(model,model,cam,cam,torch,render,pipe)
        assert identity['passed'] and identity['strict_image_parity']['max_abs']==0
        result.update(updated=now(),source_hashes=hashes,identity_control=identity['strict_image_parity'],training_resumed=False,optimizer_updates_added=0,original_strict_gate_overridden=False)
        save(out/'result.json',result)
    for path,digest in hashes.items():assert sha(path)==digest
    print('EXPORT_ROUNDOFF_DIAGNOSIS',result,flush=True)
'''
    ns=dict(original.__dict__);ns['certify']=certify
    exec(compile('def run():\n'+prefix+tail,'<readonly-roundoff-diagnosis>','exec'),ns)
    ns['run']()

if __name__=='__main__':main()
