"""Rank public AV2 human tracks by observed world motion; no model evaluation."""
from pathlib import Path
import json,collections
import numpy as np
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003/street-preflight'
def transform(row):
    m=np.eye(4);m[:3,:3]=Rotation.from_quat([row[k] for k in ['qx','qy','qz','qw']]).as_matrix();m[:3,3]=[row[k] for k in ['tx_m','ty_m','tz_m']];return m

def main():
    poses=feather.read_table(OUT/'city_SE3_egovehicle.feather').to_pylist();poses.sort(key=lambda r:r['timestamp_ns']);times=np.array([p['timestamp_ns'] for p in poses],np.int64)
    posemap={p['timestamp_ns']:transform(p) for p in poses}
    tracks=collections.defaultdict(list);max_dt=0
    for row in feather.read_table(OUT/'annotations.feather').to_pylist():
        if row['category']!='PEDESTRIAN':continue
        t=row['timestamp_ns'];j=int(np.argmin(np.abs(times-t)));max_dt=max(max_dt,int(abs(times[j]-t)))
        city=posemap[int(times[j])]@np.array([row['tx_m'],row['ty_m'],row['tz_m'],1.]);tracks[row['track_uuid']].append((row,city[:3]))
    result=[]
    for uid,records in tracks.items():
        records.sort(key=lambda v:v[0]['timestamp_ns']);world=np.array([r[1] for r in records]);distances=np.array([np.linalg.norm([r[0]['tx_m'],r[0]['ty_m']]) for r in records]);count=np.array([r[0]['num_interior_pts'] for r in records]);duration=(records[-1][0]['timestamp_ns']-records[0][0]['timestamp_ns'])/1e9
        displacement=float(np.linalg.norm(world[-1]-world[0]));nearest=int(distances.argmin())
        result.append(dict(track_uuid=uid,observations=len(records),duration_s=duration,world_endpoint_displacement_m=displacement,median_lidar_points=float(np.median(count)),minimum_ego_range_m=float(distances.min()),closest_timestamp_ns=records[nearest][0]['timestamp_ns'],timestamps_ns=[r[0]['timestamp_ns'] for r in records],moving_candidate=len(records)>=20 and displacement>=1.0))
    result.sort(key=lambda r:(not r['moving_candidate'],r['minimum_ego_range_m']))
    output=dict(status='metadata_screening_only',scene='00a6ffc1-6ce9-3bc3-a060-6006e9893a1a',source='https://argoverse.s3.amazonaws.com/',license='CC BY-NC-SA 4.0',terms='https://www.argoverse.org/about.html#terms-of-use',attribution='© 2021 Argo AI, LLC',selection='at least 20 annotation observations and at least 1m endpoint displacement after compensating ego-motion; rank by minimum ego-range; camera visibility still requires image inspection',maximum_annotation_pose_time_offset_ns=max_dt,pedestrian_tracks=len(result),moving_candidates=sum(r['moving_candidate'] for r in result),ranked_tracks=result)
    (OUT/'human-track-screening.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in output.items() if k!='ranked_tracks'},ensure_ascii=False,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='timestamps_ns'} for r in result[:5]],indent=2))

if __name__=='__main__':main()
