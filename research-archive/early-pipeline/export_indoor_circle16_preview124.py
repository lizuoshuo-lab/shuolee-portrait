"""Publishable diagnostic preview; original strict-save failure stays failed.

Reuse rigid PLY/camera export and actual native readback. This one run may use
the separately reviewed standard-scale roundoff diagnosis for its source PLY.
It does not promote reconstruction quality or change the original training gate.
"""
import difflib,inspect
from pathlib import Path
import export_cross_subject_sh0124 as shared

SUBJECT='indoor-circle16-paired18k124'

def main():
    root=shared.ROOT/SUBJECT;snap=root/'gaussian-run/step-018000'
    rt=shared.rt;cert=root/'export-roundoff-certificate124-v2/result.json'
    evidence=rt.read(cert)
    assert evidence['passed'] and not evidence['strict_image_parity']['passed']
    assert evidence['properties']['scale_encoding_exact']
    for p,h in evidence['source_hashes'].items():assert rt.sha(p)==h
    rt.save(root/'preview-export-decision124.json',dict(updated=rt.now(),
        scope='Diagnostic preview with exact standard log-scale encoding and separately explained one-pixel alpha-cutoff discontinuity. Original source strict parity remains failed.',
        certificate=str(cert),certificate_sha256=rt.sha(cert),
        original_process_exit_changed=False,training_restarted=False,optimizer_updates_added=0,
        quality_accepted=False,default_model_promotion=False))
    old=shared.old
    source=inspect.getsource(old.prepare)
    adapted=source.replace("    assert parity['mean_abs'] < 2e-6 and parity['max_abs'] < .005",'''    certificate=rt.read(ROOT/'export-roundoff-certificate124-v2/result.json')
    assert certificate['passed'] and not certificate['strict_image_parity']['passed']
    for source_path,digest in certificate['source_hashes'].items():assert rt.sha(source_path)==digest''')
    adapted=adapted.replace('native_source_ply_parity=dict(parity, passed=True)',
        "native_source_ply_parity=dict(parity, passed=False), native_source_roundoff_diagnosis=certificate, diagnostic_preview_only=True")
    assert adapted!=source
    namespace=dict(old.prepare.__globals__)
    # Set the original globals now; the compiled preparation uses this snapshot.
    row=rt.read(root/'data/training-inputs.json')['records'][0]
    R,_=shared.exp.rigid_rotation64(shared.np.diag([1.,-1.,-1.])@shared.np.asarray(row['w2c'])[:3,:3])
    namespace.update(ROOT=root,R=R,paths=lambda step:(root/'gaussian-run'/f'step-{step:06d}',root/'delivery'/f'step-{step:06d}'))
    # Resolve rotate_ply dynamically after the shared SH0 adapter installs it.
    namespace['rotate_ply']=lambda *a,**k:old.rotate_ply(*a,**k)
    exec(compile(adapted,'<diagnostic-preview-prepare>','exec'),namespace)
    old.prepare=namespace['prepare']
    text=inspect.getsource(shared.main)
    before="    assert progress['state']=='bounded_control_completed',progress"
    after="    assert subject=='indoor-circle16-paired18k124' and progress['state']=='training_budget_reached_snapshot_export_failed' and progress['step']==18000 and progress['pid'] is None,progress"
    assert before in text;text=text.replace(before,after)
    env=dict(shared.__dict__);exec(compile(text,'<completed-budget-preview-export>','exec'),env)
    env['main'](SUBJECT,18000)
    out=root/'delivery/step-018000'
    (out/'preview-export-interface.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),adapted.splitlines(True))))
    valid=rt.read(out/'validation.json')
    assert valid['format_passed'] and not valid['native_source_ply_parity']['passed']
    valid['quality_accepted']=False;valid['strict_source_native_parity_passed']=False
    rt.save(out/'validation.json',valid)
    rt.save(out/'source-roundoff-diagnosis.json',evidence)
    context=rt.read(out/'context.json')
    context['rendererContract']['nativeImageEquivalence']='Source strict max-image gate failed at one pixel due to verified float32 scale roundtrip at original alpha cutoff. Full geometry checked; standard world PLY to +Y PLY and serialized camera checked separately. Spark is a diagnostic preview.'
    context['diagnosticPreviewOnly']=True;rt.save(out/'context.json',context)
    print('INDOOR16_DIAGNOSTIC_PREVIEW_READY',valid['points'],flush=True)

if __name__=='__main__':main()
