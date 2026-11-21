"""Validate and unpack one completed author archive downloaded by the user.

Do not accept a browser partial; never follow links or execute archive contents.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse, gzip, hashlib, json, os, shutil, tarfile, time

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-enerf-full-023'
ROOT=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted')
EXPECTED={'actor1_4.tar.gz':12029956569,'actor2_3.tar.gz':11789404504,'actor5_6.tar.gz':11779739564}
LIMIT=70*1024**3

def now():return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def target_path(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or '\\' in name or any(v in ('..','') or ':' in v for v in p.parts):
        raise ValueError('Unsafe archive path')
    q=root.joinpath(*p.parts).resolve()
    if not q.is_relative_to(root.resolve()):raise ValueError('Archive path escapes destination')
    return q

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);a=ap.parse_args()
    archive=Path(a.archive).resolve()
    if archive.name not in EXPECTED:raise ValueError('Select one completed author actor*.tar.gz file')
    if archive.stat().st_size!=EXPECTED[archive.name]:raise ValueError('File size differs from official folder metadata; partial preserved')
    dest=ROOT/archive.name.removesuffix('.tar.gz')
    receipt=OUT/('extraction-'+dest.name+'.json')
    if receipt.exists() or dest.exists():raise FileExistsError('Preserve prior extraction; inspect before any retry')
    ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ROOT).free<LIMIT+10*1024**3:raise RuntimeError('Less than80GiB free for bounded extraction')
    archive_hash=digest(archive);dest.mkdir();entries=[];seen=set();expanded=0;tick=time.monotonic()
    state=OUT/('extraction-'+dest.name+'-state.json')
    def status(label,**fields):
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(dict(status=label,pid=os.getpid(),updated_utc=now(),**fields),indent=2)+'\n');os.replace(temp,state)
    status('extracting',archive=str(archive),destination=str(dest))
    try:
        with gzip.open(archive,'rb') as gz:
            with tarfile.open(fileobj=gz,mode='r|') as tar:
                for item in tar:
                    path=target_path(dest,item.name)
                    if item.isdir():path.mkdir(parents=True,exist_ok=True);continue
                    if not item.isfile() or item.issym() or item.islnk():raise ValueError('Non-regular archive member')
                    if path in seen:raise ValueError('Duplicate destination in archive')
                    seen.add(path);expanded+=item.size
                    if expanded>LIMIT or len(seen)>200000:raise RuntimeError('Extraction exceeds declared budget')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    h=hashlib.sha256();written=0
                    with tar.extractfile(item) as src,path.open('xb') as dst:
                        for block in iter(lambda:src.read(2**20),b''):
                            dst.write(block);h.update(block);written+=len(block)
                    if written!=item.size:raise ValueError('Truncated archive member')
                    entries.append(dict(path=path.relative_to(dest).as_posix(),bytes=written,sha256=h.hexdigest()))
                    if time.monotonic()-tick>15:
                        status('extracting',files=len(entries),expanded_bytes=expanded);tick=time.monotonic()
            # Read to gzip EOF so trailer CRC/length failures cannot hide after tar's end marker.
            for _ in iter(lambda:gz.read(2**20),b''):pass
        if archive.stat().st_size!=EXPECTED[archive.name] or digest(archive)!=archive_hash:
            raise RuntimeError('Archive changed during extraction')
        roots=[str(p.parent) for p in dest.rglob('intri.yml') if (p.parent/'extri.yml').is_file()]
        result=dict(status='archive_size_gzip_crc_and_all_regular_members_verified_real_image_audit_pending',
            archive=str(archive),archive_sha256=archive_hash,official_checksum_available=False,
            destination=str(dest),files=len(entries),expanded_bytes=expanded,entries=entries,
            calibration_roots=roots,local_only=True,completed_utc=now(),
            citations=['https://zju3dv.github.io/4k4d/','https://github.com/zju3dv/EasyVolcap'])
        receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        status('extraction_complete_input_audit_pending',files=len(entries),calibration_roots=roots)
        print(json.dumps({k:result[k] for k in ['status','files','expanded_bytes','calibration_roots']},indent=2))
    except BaseException as e:
        status('extraction_failed_partial_preserved',error=repr(e),files=len(entries),expanded_bytes=expanded);raise

if __name__=='__main__':main()
