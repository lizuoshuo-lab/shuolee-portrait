"""Inspect author-released camera metadata before selecting any test photographs."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pyarrow.parquet as pq
import requests

LAB=Path(__file__).resolve().parents[2]
HOME=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
OUT=HOME/'data/public-validation/diffuman4d-example'
REPO='krahets/diffuman4d_example'


def get(url):
    r=requests.get(url,timeout=45);r.raise_for_status();return r


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    info=get('https://huggingface.co/api/datasets/'+REPO).json();revision=info['sha']
    readme=get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/README.md').text
    (OUT/'SOURCE.txt').write_text(f'https://huggingface.co/datasets/{REPO}\nRevision: {revision}\n'+readme,encoding='utf-8')
    sys.path.insert(0,str(LAB/'scripts/dynamic_human'))
    from audit_enerf_input import read_cameras
    results=[]
    for name in ['0013_01','0023_06']:
        f=OUT/(name+'.parquet')
        if not f.exists():
            data=get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/cameras/{name}.parquet').content
            assert len(data)<1_000_000
            f.write_bytes(data)
        rows=pq.read_table(f).to_pylist();assert len(rows)==1
        row=rows[0];folder=OUT/name;folder.mkdir(exist_ok=True)
        for field,file in [('cam_evc_intri','intri.yml'),('cam_evc_extri','extri.yml'),('cam_ns','transforms.json')]:
            (folder/file).write_text(row[field],encoding='utf-8')
        cams=read_cameras(folder);C=[];forward=[];up=[]
        for c in cams.values():
            R=np.asarray(c['w2c'])[:3,:3]
            C.append(c['camera_center_world']);forward.append(R[2]);up.append(-R[1])
        C=np.array(C);forward=np.array(forward);u=np.mean(up,axis=0);u/=np.linalg.norm(u)
        P=np.eye(3)[None]-forward[:,:,None]*forward[:,None,:]
        target=np.linalg.lstsq(P.sum(0),np.einsum('nij,nj->i',P,C),rcond=None)[0]
        ray=C-target;angle=np.degrees(np.arcsin(np.clip(ray@u/np.linalg.norm(ray,axis=-1),-1,1)))
        transforms=json.loads(row['cam_ns'])
        results.append(dict(scene=name,cameras=len(cams),elevation_range=[float(angle.min()),float(angle.max())],
            elevation_span=float(np.ptp(angle)),camera_elevations={n:float(a) for n,a in zip(cams,angle)},
            optical_axis_intersection=target.tolist(),up_from_cameras=u.tolist(),
            parquet_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            transform_keys=list(transforms),example_camera_metadata=transforms.get('frames',[None])[0],
            camera_keys=list(cams),sparse_point_cloud_extracted=False))
    result=dict(status='metadata_audited',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),repo=REPO,revision=revision,
        license=info.get('cardData',{}).get('license'),sources=['https://github.com/zju3dv/Diffuman4D',f'https://huggingface.co/datasets/{REPO}'],
        image_content_inspected=False,results=results,
        limits='Camera-up and axis-intersection based angles, not gravity. No synchronization or image-quality acceptance yet. Author sample is not automatically independent of pretrained priors.')
    (OUT/'metadata-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
