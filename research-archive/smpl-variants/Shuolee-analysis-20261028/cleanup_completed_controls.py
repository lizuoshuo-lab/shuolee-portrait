"""Remove intermediate point-cloud snapshots only after final results and evidence exist."""
from pathlib import Path
import datetime,hashlib,json
base=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029').resolve()
evidence=Path('/mnt/d/Projects/Shuolee-analysis-20261028').resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
removed=[];retained=[]
for name,final in [('gray-head-motion',6000),('gray-dynamic-face',18000),('gray-dynamic-face-fan',18000)]:
    run=(base/name).resolve();assert base in run.parents
    result=json.loads((run/'result.json').read_text());assert result['state']=='completed' and result['steps']==final
    last=run/f'step-{final:06d}/canonical.ply';assert last.is_file();retained.append(dict(path=str(last),sha256=sha(last)))
    if name.endswith('fan'):
        assert json.loads((evidence/'dynamic-face-fan-all-views/report.json').read_text())['frames']==95
        assert json.loads((run/'export/validation.json').read_text())['format_passed']
    for step in [1000,6000,12000]:
        if step>=final:continue
        snapshot=run/f'step-{step:06d}';path=snapshot/'canonical.ply'
        if not path.exists():continue
        assert not path.is_symlink() and run in path.resolve().parents
        metadata=json.loads((snapshot/'metadata.json').read_text());digest=sha(path)
        assert digest==metadata['canonical_ply_sha256']
        assert list(snapshot.glob('*_render.png'))
        removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest,
            reason='Intermediate point cloud; final model, complete logs, per-step renders and checksum retained'))
        path.unlink()
gallery=Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图').resolve();fig=gallery/'修复对照'
page=(gallery/'修复进展.html').read_text()
for step in [6000,12000]:
    for label,file in [('头部渲染对照','head-comparison.png'),('完整画面对照','full-comparison.png')]:
        path=fig/f'fan_{step}步_{label}.png';source=evidence/f'dynamic-face-fan-render-comparison/step-{step:06d}'/file
        if not path.exists():continue
        assert gallery in path.resolve().parents and not path.is_symlink()
        assert f'修复对照/{path.name}"' not in page
        digest=sha(path);assert digest==sha(source)
        removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest,retained_copy=str(source),reason='Duplicate intermediate figure'))
        path.unlink()
receipt=dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),removed=removed,retained_final_models=retained,
    bytes_removed=sum(x['bytes'] for x in removed),source_videos_removed=False,original_fits_removed=False,quality_accepted=False)
(evidence/'intermediate-ply-cleanup.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(removed_files=len(removed),bytes_removed=receipt['bytes_removed'],final_models_retained=len(retained))))
