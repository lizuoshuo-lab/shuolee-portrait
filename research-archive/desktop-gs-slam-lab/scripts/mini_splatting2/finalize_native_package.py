"""Audit and revise this not-yet-delivered package, retaining prior deliveries."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile
from package_native_stage import LAB, ROOT, README


def main():
    receipt_path=ROOT/'native-package-receipt.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    assert receipt['status']=='packaged_not_transferred'
    stage=Path(receipt['local_stage'])/'MiniGS2_Native'
    assert stage.resolve().is_relative_to(ROOT.resolve())
    package=Path(receipt['package'])
    assert package.name=='MiniGS2_Native_20261019.zip'
    (stage/'使用说明.txt').write_text(README,encoding='utf-8')
    models=[]
    for meta_path in sorted(stage.glob('model-*.json')):
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        ply=stage/meta['model'];blob=ply.read_bytes()
        sha=hashlib.sha256(blob).hexdigest()
        assert sha==meta['sha256'],meta_path
        assert len(blob)==meta['modelBytes'],meta_path
        header=blob.split(b'end_header',1)[0].decode('ascii')
        count=int(next(l for l in header.splitlines() if l.startswith('element vertex ')).split()[-1])
        assert count==meta['gaussianCount'] and meta['shDegree']==3
        models.append(dict(name=meta_path.name,sha256=sha,gaussians=count))
    assert len(models)==10
    original=LAB/'outputs/mini-splatting2-computer00-001/viewer/models/TRAINED_18000/point_cloud.ply'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='0751327e63353b2c1cf9db302f8f5861d72571879aaade8e791a32abfc823eb8'
    depth=ROOT/'native-depthmask-summary.json'
    data=json.loads(depth.read_text(encoding='utf-8-sig'))
    depth.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    q=ROOT/'native-depthmask-queue.json';queue=json.loads(q.read_text(encoding='utf-8-sig'))
    queue['visual_review']=data['visual_review'];q.write_text(json.dumps(queue,indent=2),encoding='utf-8')
    assert "?query.get('model'):'native-motion-mip-depthmask'" in (stage/'viewer.js').read_text(encoding='utf-8')
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in stage.rglob('*'):
            if p.is_file():z.write(p,Path('MiniGS2_Native')/p.relative_to(stage))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        count=len(z.namelist())
        assert len(set(z.namelist()))==count
    receipt.update(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=package.stat().st_size,
                   sha256=hashlib.sha256(package.read_bytes()).hexdigest(),files=count,
                   audit='10 PLY hashes, byte lengths, Gaussian counts and SH3 checked; original PLY unchanged',
                   default_model='native-motion-mip-depthmask',model_audit=models)
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['package','bytes','sha256','crc','files','model_count','status']}))


if __name__=='__main__':main()
