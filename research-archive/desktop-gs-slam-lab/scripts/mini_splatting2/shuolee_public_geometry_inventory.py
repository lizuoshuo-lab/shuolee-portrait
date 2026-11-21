"""Read calibration metadata to qualify existing public validation options."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LAB/'scripts/dynamic_human'))
from audit_enerf_input import read_cameras,digest


def main():
    root=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted')
    out=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee/current/public-validation-inventory.json'
    assert not out.exists()
    results=[]
    for scene in ['actor1_4','actor2_3','actor5_6']:
        folder=root/scene;cams=read_cameras(folder/'optimized')
        centers=[];forward=[];up=[]
        for c in cams.values():
            R=np.asarray(c['w2c'])[:3,:3]
            centers.append(c['camera_center_world']);forward.append(R[2]);up.append(-R[1])
        C=np.array(centers);f=np.array(forward);u=np.mean(up,axis=0);u/=np.linalg.norm(u)
        P=np.eye(3)[None]-f[:,:,None]*f[:,None,:]
        target=np.linalg.lstsq(P.sum(0),np.einsum('nij,nj->i',P,C),rcond=None)[0]
        directions=C-target
        elevation=np.degrees(np.arcsin(np.clip(directions@u/np.linalg.norm(directions,axis=-1),-1,1)))
        points=[]
        for index,(name,c) in enumerate(cams.items()):
            files=sorted((folder/'images'/name).glob('*.jpg'))
            # Header only: do not inspect selected evaluation image content.
            with Image.open(files[0]) as im:width,height=im.size
            points.append(dict(camera=name,frames=len(files),first=files[0].stem,last=files[-1].stem,
                size=[width,height],center=c['camera_center_world'],elevation_deg=float(elevation[index]),
                depth_toward_optical_axis_intersection=float((target-C[index])@f[index])))
        results.append(dict(scene=scene,cameras=points,
            calibration_sha256={n:digest(folder/'optimized'/n) for n in ['intri.yml','extri.yml']},
            optical_axis_intersection=target.tolist(),up_from_average_camera= u.tolist(),
            elevation_range=[float(elevation.min()),float(elevation.max())],elevation_span=float(np.ptp(elevation)),
            photographic_content_inspected=False,prior_use='All three sequences occur in existing project scripts; do not automatically label an entire sequence unseen.'))
    result=dict(status='metadata_inventory_complete_not_final_test_frozen',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        datasets=results,limitations=['Elevation is relative to the average camera-up direction and fitted optical-axis intersection, not gravity or measured head center.',
        'Numerical calibration and shared frame numbers do not prove subframe synchronization. Existing project records include timing discrepancies.',
        'Freeze a genuinely unused temporal/view block only after auditing historical train/evaluation usage. Released surfs geometry cannot initialize a held-out experiment without provenance.'],
        next='Audit existing sequence usage and capture synchronization before a native-scale held-out-geometry validation.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['scene','elevation_range','elevation_span']} for r in results],indent=2))


if __name__=='__main__':main()
