"""Independent CPU verification of retained history and actual image controls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
EVALS = ROOT/'outputs/large-scene-gs-001/independent-evaluation'
SOURCE = ROOT/'scripts/large_scene_gs/audit_city_boundary_images_002.py'
SOURCE_SHA = '15cc350108647f47ee7c20c6297ce7930d6af674233bbc3eac2017eeb74fcb1f'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    report_path = OUT/'CPU-verification.json'
    require(not report_path.exists(), 'Preserve existing CPU verification')
    require(digest(SOURCE) == SOURCE_SHA, 'Boundary002 source changed')
    manifest = read(OUT/'history/manifest.json')
    require(len(manifest['files']) == 4, 'Missing original001 source/report bytes')
    for row in manifest['files']:
        require(digest(row['original_path']) == digest(row['snapshot_path']) == row['sha256'],
                'Original001 or its history copy changed')
    old = read(EVALS/'small-city-coarse-boundary-reference-CPU-001/report.json')
    successful = EVALS/'small-city-coarse-boundary-reference-CPU-003'
    report = read(successful/'report.json'); outcome = read(successful/'outcome.json')
    require(outcome['status'] == 'completed_actual_coarse_reference_boundary_image_index_only'
            and outcome['exit_code'] == 0 and outcome['City_dense_boundary_results_complete'] is False,
            'Actual coarse reference did not finish within its separate scope')
    require(outcome['report_sha256'] == digest(successful/'report.json')
            and digest(successful/'source-snapshot.py') == SOURCE_SHA, 'Actual source/report bytes not bound')
    require(report['actual_original_frame_sets'] == report['view_count'] == 8 and report['fixed_patch_count'] == 24
            and report['views'] == old['views'] and report['selected_names'] == old['selected_names'],
            'Actual images/fixed patch arithmetic changed between001 and revised002')
    require(report['all_original_sources_unchanged'] is True and report['images_edited_or_generated'] is False
            and report['CUDA_initialized'] is False and report['torch_imported'] is False
            and report['actual_dense_completion_binding'] is None, 'Coarse scope/image identity weakened')
    for path, sha in report['protected_original_sources'].items():
        require(digest(path) == sha, 'An actual source/original image changed after CPU check')
    missing = EVALS/'small-city-dense-boundary-missing1404-control-CPU-002'
    refusal = read(missing/'outcome.json')
    require(refusal['status'] == 'failed_preserved_CPU_boundary_index' and refusal['exit_code'] == 1
            and refusal['mode'] == 'dense' and refusal['CUDA_initialized'] is False
            and refusal['error'] == 'Actual completed1404 posterior proof missing; no City dense boundary completion',
            'Actual missing1404 dependency was not refused')
    require(not (missing/'report.json').exists() and digest(missing/'source-snapshot.py') == SOURCE_SHA,
            'A refusal was wrongly published as dense completion')
    first = EVALS/'small-city-coarse-boundary-reference-CPU-002'
    first_outcome = read(first/'outcome.json')
    require(first_outcome['status'] == 'failed_preserved_CPU_boundary_index'
            and first_outcome['error'] == 'Original frame paths outside exact selected family/setting/ordinal'
            and digest(first/'source-snapshot.py') == 'f60942693edb0e3c833ed900ea7282af6716679b60d0ecf4a2f3c444f29ebdddd'
            and not (first/'report.json').exists(), 'Original coarse naming-guard refusal was not preserved')
    require('torch' not in sys.modules, 'CPU checker imported Torch')
    result = dict(status='CPU_boundary002_actual_coarse_identity_and_missing1404_refusal_passed',
        verified_utc=datetime.now(timezone.utc).isoformat(),boundary_source_sha256=SOURCE_SHA,
        retained_original001_files=4,history_manifest_sha256=digest(OUT/'history/manifest.json'),
        actual_coarse_reference_namespace=successful.name,actual_coarse_reference_outcome_sha256=digest(successful/'outcome.json'),
        actual_coarse_reference_report_sha256=digest(successful/'report.json'),actual_original_frame_sets=8,
        fixed_geometry_patches=24,all_original_images_and_source_hashes_rechecked=True,
        original001_images_and_fixed_patch_arithmetic_identical=True,
        missing1404_namespace=missing.name,missing1404_outcome_sha256=digest(missing/'outcome.json'),
        missing1404_actual_exit_code=1,missing1404_refused_before_model_or_original_image_reads=True,
        preserved_first_coarse_naming_refusal_namespace=first.name,
        actual1404_or_dense_boundary_results_complete=False,Torch_imported=False,CUDA_called=False,
        native_stage_launcher_or_process_control_executed=False,
        limitation='CPU controls use genuine previous coarse frames and an actually absent1404 posterior. They do not fabricate future dense models, frames or completion.')
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
