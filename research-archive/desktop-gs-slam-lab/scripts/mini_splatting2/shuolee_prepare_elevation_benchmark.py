"""Freeze and stage one synchronized instant with disjoint camera-height rings."""
import hashlib
import io
import json
from pathlib import Path
import time
import pyarrow.parquet as pq
import requests
from PIL import Image
from shuolee_audit_diffuman_cameras import OUT, REPO


def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    audit=json.loads((OUT/'metadata-audit.json').read_text(encoding='utf-8'))
    scene='0023_06';frame='000075'
    record=next(r for r in audit['results'] if r['scene']==scene)
    roles={'train_horizontal':[c for c,a in record['camera_elevations'].items() if abs(a)<5],
        'development_middle':[c for c,a in record['camera_elevations'].items() if 10<a<25],
        'reserved_upper':[c for c,a in record['camera_elevations'].items() if a>30]}
    assert all(len(v)==16 for v in roles.values()) and len(set(sum(roles.values(),[])))==48
    folder=OUT/'elevation-benchmark';folder.mkdir(exist_ok=True)
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='frozen_before_rgb_download',
        dataset=REPO,revision=audit['revision'],scene=scene,frame=frame,roles=roles,
        selection='Single temporal midpoint and all three height rings, selected from calibration only; no RGB inspection.',
        data_role='Controlled human static-instant reconstruction; no claim of street-scene or dynamic reconstruction completion.',
        test_policy='Upper16cameraRGB/masks must not enter training, initialization, pseudo-label generation, or hyperparameter selection. Evaluate only after candidate/method is frozen.',
        initialization='Use training-camera images only for triangulation or a common input-independent initialization; do not extract the author sparse_pcd.',
        cameras='Known calibrated camera poses permitted for render evaluation. Camera-up based elevation, not gravity.',
        independent_scope='New data for this project. Not guaranteed absent from any pretrained prior training corpus.',
        source_license=audit['license'],input_image_quality_and_sync_verified=False,accepted=False)
    protocol_path=folder/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text(encoding='utf-8'))
        assert all(previous[k]==protocol[k] for k in ['dataset','revision','scene','frame','roles'])
        protocol=previous
    else:dump(protocol_path,protocol)
    inventory=requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{audit["revision"]}?recursive=true',timeout=30)
    inventory.raise_for_status();entries={r['path']:r for r in inventory.json()}
    status=dict(status='downloading',scene=scene,frame=frame,completed=[])
    status_path=folder/'status.json';dump(status_path,status)
    for kind,column,extension in [('images','image','webp'),('fmasks','fmask','png')]:
        name=f'{kind}/{scene}.parquet';entry=entries[name]
        assert entry['size']<600_000_000
        target=OUT/(kind+'-'+scene+'.parquet')
        if not target.exists():
            partial=target.with_suffix('.part')
            assert not partial.exists(),'Inspect interrupted download instead of replacing it'
            url=f'https://huggingface.co/datasets/{REPO}/resolve/{audit["revision"]}/{name}'
            response=requests.get(url,stream=True,timeout=(30,90));response.raise_for_status()
            received=0;last=time.time()
            with partial.open('xb') as f:
                for block in response.iter_content(2**20):
                    f.write(block);received+=len(block)
                    assert received<=entry['size']
                    if time.time()-last>10:
                        status.update(file=name,bytes=received,total=entry['size']);dump(status_path,status)
                        print(json.dumps(status),flush=True);last=time.time()
            assert received==entry['size']
            partial.rename(target)
        assert target.stat().st_size==entry['size']
        actual=digest(target)
        if entry.get('lfs',{}).get('oid'):assert actual==entry['lfs']['oid']
        found={}
        for batch in pq.ParquetFile(target).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if str(row['frame']).zfill(6)!=frame:continue
                cam=str(row['camera']).zfill(2)
                role=next(k for k,v in roles.items() if cam in v)
                dest=folder/role/kind/cam/f'{frame}.{extension}';dest.parent.mkdir(parents=True,exist_ok=True)
                data=row[column]
                with Image.open(io.BytesIO(data)) as im:
                    dimensions=im.size;im.verify()
                if dest.exists():assert dest.read_bytes()==data
                else:dest.write_bytes(data)
                found[cam]=dict(role=role,size=list(dimensions),sha256=hashlib.sha256(data).hexdigest())
        assert len(found)==48
        status['completed'].append(dict(kind=kind,file=name,sha256=actual,bytes=target.stat().st_size,cameras=found))
        dump(status_path,status)
    status.update(status='staged_not_visualized',training_started=False,test_images_inspected=False,
        author_sparse_points_extracted=False,frame=frame,roles=roles)
    dump(status_path,status);print(json.dumps(dict(status=status['status'],scene=scene,frame=frame,roles=roles)),flush=True)


if __name__=='__main__':main()
