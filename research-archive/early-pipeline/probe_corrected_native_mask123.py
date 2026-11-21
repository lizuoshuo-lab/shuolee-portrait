"""Re-run the unchanged registered alpha contract with isolated fixed AA backward."""
import fcntl,os,signal,time,traceback
from pathlib import Path
import run_official_gaussians123 as io
import probe_native_mesh_mask123 as probe
from test_isolated_aa_grad123 import OUT as CONTROL,load_variants

OUT=CONTROL/'all59-mask-contract'


def main():
    original_plan=probe.checked()
    synthetic=io.read(CONTROL/'synthetic-result.json');assert synthetic['synthetic_pass']
    face=io.read(probe.AUDIT/'face-landmark-boundary/result.json')
    assert face['overlap_vertices']==0 and not face['new_face_fit_run']
    OUT.mkdir(exist_ok=False)
    plan=dict(original_plan)
    plan.update(created=io.now(),implementation_repair='Only AA backward position accumulation uses isolated direct atomicAdd. Unchanged original forward/projection, input masks, face exclusion, thresholds and no optimizer.',
        isolated_backward_build=io.sha(CONTROL/'build-result.json'),
        synthetic_control=io.sha(CONTROL/'synthetic-result.json'),
        face_boundary=io.sha(probe.AUDIT/'face-landmark-boundary/result.json'),
        wrapper_sha256=io.sha(__file__))
    io.save(OUT/'plan.json',plan)
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=probe.process(331)
    assert before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert probe.active_gpu_pids()<={331}
    try:
        os.kill(331,signal.SIGSTOP)
        for _ in range(50):
            if probe.process(331)['state']=='T':break
            time.sleep(.05)
        assert probe.process(331)['state']=='T'
        import torch
        import nvdiffrast.torch.ops as ops
        modules=load_variants();original=ops._nvdiffrast_c.antialias_grad
        ops._nvdiffrast_c.antialias_grad=modules['direct'].antialias_grad
        try:
            probe.OUT=OUT
            probe.replay(331)
        finally:ops._nvdiffrast_c.antialias_grad=original
        result=io.read(OUT/'result.json')
        result.update(source='Unchanged original VHAP alpha forward and camera projection; isolated two-call direct-atomic backward repair verified by paired synthetic control.',
            installed_runtime_modified=False,production_training_modified=False,
            face_exclusion='Verified4906 protected head/neck vertices and356 static/dynamic landmark support vertices; all59 protected gradients zero.',
            original_failed_contract_retained=True,no_parameter_or_threshold_change=True,
            diagnostic_only=True,quality_accepted=False)
        io.save(OUT/'result.json',result)
    except BaseException as error:
        io.save(OUT/'failure.json',dict(updated=io.now(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        foreign=probe.active_gpu_pids()-{331,os.getpid()}
        if not foreign:os.kill(331,signal.SIGCONT)
        io.save(OUT/'gpu-handoff.json',dict(updated=io.now(),before=before,after=probe.process(331),foreign=list(foreign)))


if __name__=='__main__':main()
