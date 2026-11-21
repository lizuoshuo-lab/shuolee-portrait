from pathlib import Path
import shutil,json,hashlib
P=Path('/mnt/d/00/工作/人体重建')
sources=[
('/opt/gs-slam-lab/shuolee-ga-raster/source','src/avatar_core/submodules/diff-gaussian-rasterization'),
('/opt/gs-slam-lab/mini-splatting2-native-001/repo/submodules/simple-knn','src/avatar_core/submodules/simple-knn'),
('/opt/gs-slam-lab/dynamic-human/repos/pytorch3d','third_party/pytorch3d'),
('/opt/gs-slam-lab/dynamic-human/repos/nvdiffrast','third_party/nvdiffrast')]
records=[]
excluded={'.git','build','dist','__pycache__','.pytest_cache','docs','tests','website','packaging','.github','examples','samples'}
for source,dest in sources:
 src=Path(source);dst=P/dest
 for p in src.rglob('*'):
  rel=p.relative_to(src)
  if not p.is_file() or any(x in excluded or x.endswith('.egg-info') for x in rel.parts):continue
  if p.suffix.lower() in {'.so','.o','.a','.pyc','.pth','.pt','.png','.jpg','.jpeg','.gif','.mp4','.npy','.npz','.pkl','.zip'}:continue
  d=dst/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
  records.append(dict(path=d.relative_to(P).as_posix(),bytes=d.stat().st_size,sha256=hashlib.sha256(d.read_bytes()).hexdigest(),kind='native_build_source_snapshot'))
f=P/'provenance/source-manifest.json';doc=json.loads(f.read_text())
by_path={r['path']:r for r in doc['files']}
for r in records:by_path[r['path']]=r
for r in by_path.values():
 p=P/r['path'];actual=hashlib.sha256(p.read_bytes()).hexdigest()
 if actual!=r['sha256']:r['upstream_sha256_before_portable_patch']=r['sha256'];r['sha256']=actual;r['bytes']=p.stat().st_size
doc['files']=list(by_path.values());doc['native_source_note']='Build sources copied from the actual installed local runtime. Commit provenance is recorded only where available; file hashes identify these exact snapshots.'
f.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'copied_native_files':len(records),'bytes':sum(x['bytes'] for x in records)}))
