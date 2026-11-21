from pathlib import Path
import json,hashlib,shutil,datetime
base=Path('/opt/gs-slam-lab/msv2-video15-20261105/runs/video15-face-composite4x').resolve(strict=True)
assert base.name=='video15-face-composite4x' and base.parent.name=='runs'
assert json.loads((base/'state.json').read_text())['state']=='interrupted'
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try: raw=proc.read_bytes()
 except (FileNotFoundError,PermissionError,ProcessLookupError): continue
 if b'video15-face-composite4x' in raw: raise RuntimeError('A live process still references the superseded run')
selected=[]
for level in (1,2):
 candidate=base/f'swinir4x/rgb-cache-{level}'
 resolved=candidate.resolve(strict=True)
 assert resolved.is_relative_to(base) and resolved.parent==base/'swinir4x'
 assert not candidate.is_symlink() and resolved.name==f'rgb-cache-{level}'
 files=list(resolved.iterdir())
 assert len(files)==65 and all(f.is_file() and f.suffix=='.npy' and not f.is_symlink() and f.resolve().parent==resolved for f in files)
 selected.append(dict(path=str(resolved),files=len(files),bytes=sum(f.stat().st_size for f in files)))
protected=[base/n for n in ('job.json','state.json','plan.json','swinir4x/superresolution.json','swinir4x/training-inputs.json')]
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
for item in selected: shutil.rmtree(item['path'])
assert all(not Path(item['path']).exists() for item in selected)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
receipt=dict(state='removed_regenerable_superseded_caches',finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),removed=selected,freed_bytes=sum(x['bytes'] for x in selected),freed_gib=sum(x['bytes'] for x in selected)/1024**3,protected_provenance_sha256=before,originals_masks_SR_targets_models_retained=True,previously_rejected_fixture_delete_not_retried=True)
Path('/mnt/d/00/记录/2026-11-06_MSV2_video15_子母图联合训练/cleanup-result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'state':receipt['state'],'freed_gib':receipt['freed_gib'],'D_free_gib':shutil.disk_usage('/mnt/d').free/1024**3}))
