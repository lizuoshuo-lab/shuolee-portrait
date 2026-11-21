"""Aggregate completed, hash-verified official Truck training receipts."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/large-scene-gs-001'
RUNS = BASE/'official-training'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    target=RUNS/'truck-four-chunks-summary-001.json'
    if target.exists():
        raise RuntimeError('Retain existing summary; no overwrites')
    manifest=BASE/'official-inputs/truck-four-chunks-001/manifest.json'
    input_hash=digest(manifest)
    records=[]
    for chunk, name in [(None,'truck-coarse-001'),('0_0','truck-chunk00-001'),
                        ('0_1','truck-chunk01-001'),('1_0','truck-chunk10-001'),('1_1','truck-chunk11-001')]:
        directory=RUNS/name
        outcome=read(directory/'outcome.json')
        config=read(directory/'configuration.json')
        audit_path=directory/'model-audit.json' if chunk is not None else BASE/'hierarchy-build/truck-coarse-001-CPU-audit.json'
        audit=read(audit_path)
        if outcome['status']!='completed' or outcome['exit_code']!=0 or config['chunk']!=chunk:
            raise RuntimeError('Stage is not complete or has wrong chunk identity')
        if config['input_manifest_sha256']!=input_hash or config['requested_iterations']!=30000:
            raise RuntimeError('Changed input identity or budget')
        if set(config['training_names'])&set(config['frozen_global_test_names']):
            raise RuntimeError('Train/test leakage')
        if chunk is not None and (not audit['all_saved_parameters_finite'] or not audit['nonzero_quaternions']):
            raise RuntimeError('Invalid saved parameters')
        if chunk is None and not audit['frozen_xyz_tail_bitwise_equal']:
            raise RuntimeError('Coarse positions failed historical independent check')
        ply=Path(outcome['saved_model_directory'])/'point_cloud.ply'
        if digest(ply)!=outcome['point_cloud_sha256']:
            raise RuntimeError('Saved PLY bytes changed')
        v=PlyData.read(ply)['vertex'].data
        if not all(np.isfinite(v[field]).all() for field in v.dtype.names) or np.any(np.linalg.norm(np.stack([v[f'rot_{j}'] for j in range(4)],axis=1),axis=1)<=0):
            raise RuntimeError('Current saved model finite/quaternion check failed')
        launcher=directory/'launcher-source.py'
        if launcher.exists() and digest(launcher)!=config.get('launcher_source_sha256'):
            raise RuntimeError('Launcher snapshot changed')
        if chunk is not None and not launcher.exists():
            raise RuntimeError('Missing chunk launcher snapshot')
        peak=outcome.get('logged_peak_allocated_bytes')
        if peak is None:
            logged=[int(value) for value in re.findall(r'(?:Peak memory[=:]|peak memory:)\s*(\d+)',(directory/'training.log').read_text(encoding='utf-8',errors='replace'))]
            peak=max(logged) if logged else None
        records.append(dict(run=name,chunk=chunk,training_images=len(config['training_names']),
            requested_iterations=30000,optimizer_updates=outcome['optimizer_update_calls'],
            gaussian_count_including_context_and_sky=outcome['gaussian_count'],
            elapsed_seconds=outcome['elapsed_seconds'],peak_logged_allocated_bytes=peak,
            memory_evidence='Original outcome or explicit regex of retained official log; no external/native peak inference',training_log_sha256=digest(directory/'training.log'),
            outcome_sha256=digest(directory/'outcome.json'),configuration_sha256=digest(directory/'configuration.json'),
            point_cloud_sha256=outcome['point_cloud_sha256'],launcher_sha256=digest(launcher) if launcher.exists() else None,
            launcher_snapshot_scope='Recorded exact chunk wrapper' if launcher.exists() else 'Historical coarse run predates wrapper snapshot; source/config retained, wrapper cannot be reconstructed as an exact run snapshot',
            model_audit_sha256=digest(audit_path),model_audit_path=str(audit_path),
            current_saved_parameters_finite=True,current_nonzero_quaternions=True))
    chunks=records[1:]
    result=dict(status='five_full_stages_completed_hash_verified_not_yet_consolidated',
        script_sha256=digest(__file__),input_manifest_sha256=input_hash,runs=records,
        chunk_total_optimizer_updates=sum(r['optimizer_updates'] for r in chunks),
        chunk_total_elapsed_seconds=sum(r['elapsed_seconds'] for r in chunks),
        chunk_max_logged_peak_allocated_bytes=max(r['peak_logged_allocated_bytes'] for r in chunks),
        chunk_sum_saved_G_not_unique=sum(r['gaussian_count_including_context_and_sky'] for r in chunks),
        total_full_stage_optimizer_updates=sum(r['optimizer_updates'] for r in records),
        total_full_stage_elapsed_seconds=sum(r['elapsed_seconds'] for r in records),
        Gaussian_count_limit='Each block includes the shared 100000 skybox and background context. Counts cannot be summed as unique scene Gaussians.',
        comparison_limit='Official fixed initialization, rasterizer, density schedule, seed0 and 119996 chunk updates differ from gsplat seed42/30000 baseline. Workflow comparison, not pure partition causal ablation.',
        snapshot_limit='Official final iteration saves after backward and exits before the last optimizer step; requested I implies I-1 updates.',
        memory_limit='Peak is largest recorded Torch allocated value in official logs; reserved/native/OS peaks are not measured; per-stage peaks are not additive.')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','chunk_total_optimizer_updates','chunk_total_elapsed_seconds','chunk_sum_saved_G_not_unique']}))


if __name__=='__main__':
    main()
