"""Prepare isolated publication scripts and document the rejected evaluation/runtime trials."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001';SCRIPTS=Path(__file__).parent


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    builder=(SCRIPTS/'build_resolution_report.py').read_text(encoding='utf-8')
    for old,new in [('native-resolution-{arm}-18000-s0','native-resolution-cache-{arm}-18000-s0'),
                    ('native-resolution-low-18000-s0','native-resolution-cache-low-18000-s0'),
                    ('native-resolution-review-001','native-resolution-cache-review-001'),
                    ('native-resolution-protocol-001','native-resolution-cache-protocol-001'),
                    ('native-resolution-paths-001','native-resolution-cache-paths-001')]:
        assert old in builder,old
        builder=builder.replace(old,new)
    target=SCRIPTS/'build_resolution_cache_report.py';assert not target.exists();target.write_text(builder,encoding='utf-8')
    publisher=(SCRIPTS/'run_resolution_publication.py').read_text(encoding='utf-8')
    for old,new in [('native-resolution-publication-queue','native-resolution-cache-publication-queue'),
                    ('native-resolution-low-18000-s0','native-resolution-cache-low-18000-s0'),
                    ('native-resolution-full-18000-s0','native-resolution-cache-full-18000-s0'),
                    ('native-resolution-paths-001','native-resolution-cache-paths-001'),
                    ('render_native_resolution_paths.py','render_native_resolution_cache_paths.py'),
                    ('build_resolution_report.py','build_resolution_cache_report.py'),
                    ('native-resolution-queue.json','native-resolution-cache-queue.json'),
                    ("('resolution-'+path.stem", "('resolution-cache-'+path.stem")]:
        assert old in publisher,old
        publisher=publisher.replace(old,new)
    target=SCRIPTS/'run_resolution_cache_publication.py';assert not target.exists();target.write_text(publisher,encoding='utf-8')
    audit=ROOT/'native-resolution-evaluator-correction-001/after.json';assert not audit.exists()
    dump(audit,dict(status='implementation_corrected_new_evaluations_pending',created=time.time(),
        cause='Repeated sys.path.insert(0) reversed intended precedence and loaded legacy rasterization kernels in evaluator v1 and first memory probe',
        fix='Explicit native Mip dependency precedence, loaded-library hashes, stored3Dfilter equality, common-camera RGB equality',
        excluded=[dict(path='native-resolution-low-18000-s0/resolution-evaluation',reason='Legacy-kernel v1 invalid for native Mip metrics'),
                  dict(path='native-resolution-probe-001',reason='Geometry/reference checks retained; memory and render probe not a valid native Mip runtime benchmark')],
        original_training_library_audit='training-runtime.json',
        original_training_used_correct_mip_kernel=True,new_training_checks_loaded_libraries_per_run=True,
        old_low_training_preserved=True,original_full_training_interrupted_due_to_gpu_paging=True,
        active_queue='native-resolution-cache-queue.json',
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SCRIPTS/'evaluate_resolution_stage.py',SCRIPTS/'run_resolution_cache_stage.py',SCRIPTS/'build_resolution_cache_report.py']}))
    runtime=ROOT/'native-resolution-cache-runtime/manifest.json';r=json.loads(runtime.read_text());r.update(status='reference_cache_validated_training_transition_in_progress',
        validation='validation.json',original_joint_backward_preserved=True);dump(runtime,r)
    rejected=ROOT/'native-resolution-memory-runtime/review.json';assert not rejected.exists()
    dump(rejected,dict(status='rejected',created=time.time(),evidence='validation-attempt002.json',
        reason='Sequential backward changed scale/rotation gradients beyond the declared equivalence threshold',
        action='Do not train official comparisons with repo-depth-memory; repo-depth-cache restores the original joint backward'))
    manifest=LAB/'manifests/mini-splatting2-optimization.json';m=json.loads(manifest.read_text(encoding='utf-8'))
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');shutil.copy2(manifest,ROOT/f'manifest-before-resolution-cache-{stamp}.json')
    q=json.loads((ROOT/'native-resolution-cache-queue.json').read_text())
    m.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),confirmed_user_method='Mip-Splatting',local_viewer_models=15,active_job=[q.get('current','native-resolution-cache-queue')])
    m.setdefault('native_queues',{}).update({n:json.loads((ROOT/n).read_text()) for n in ['native-resolution-queue.json','native-resolution-publication-queue.json','native-resolution-cache-queue.json']})
    m['resolution_stage']=dict(status=q['status'],accepted=False,queue=str(ROOT/'native-resolution-cache-queue.json'),protocol=str(ROOT/'native-resolution-cache-protocol-001.json'),
        reference_cache_validation=str(ROOT/'native-resolution-cache-runtime/validation.json'),
        original_full_run_interrupted=True,legacy_v1_metrics_excluded=True,sequential_backward_rejected=True,
        current_training_session=89155)
    m['next']=['Complete full-resolution transition smoke; gate memory before long training.',
        'Finish matched low/full18k runs with unchanged joint loss and compare both at common resolutions.',
        'Visually inspect and repeat a useful candidate; keep independent high-pitch capture validation open.']
    dump(manifest,m)
    dump(ROOT/'source-resolution-rectified-002/training-status.json',dict(status='cache_transition_smoke',
        current_queue=str(ROOT/'native-resolution-cache-queue.json'),frozen_inventory_preserved=True,
        input_geometry_checks_retained=True,probe001_memory_result_excluded=True,
        original_full_run_interrupted='GPU paging; no completed model',v1_evaluation_excluded=True))
    dump(ROOT/f'progress-checkpoint-{stamp}.json',dict(created=m['updated_at'],goal='active',resolution_stage=m['resolution_stage'],
        queues=m['native_queues'],models=15,unrelated_windows_asr_pid=18064,
        spread_package=m.get('spread_package'),next=m['next']))
    print(json.dumps(dict(status=q['status'],current=q.get('current'),scripts_prepared=True,original_results_preserved=True)))


if __name__=='__main__':main()
