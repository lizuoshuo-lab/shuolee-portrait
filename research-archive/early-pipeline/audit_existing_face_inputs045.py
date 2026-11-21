from pathlib import Path
import numpy as np,json,hashlib,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
F=B/'current/novel-view-face/head-structure/fan68-cpu'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=read(B/'current/gaussianavatars-smplx-from-scratch-001/data/inputs.json')
raw={r['frame']:r for r in inputs['records']}
p=read(F/'protocol.json');n=np.load(F/'landmarks.npz')
records=[]
for r in p['records']:
    frame=r['frame'];a=np.asarray(n[frame+'_crop_xy']);H=np.asarray(r['H_native_to_crop'])
    def inv(q):
        x=np.c_[q,np.ones(len(q))]@np.linalg.inv(H).T
        return x[:,:2]/x[:,2:]
    direct=inv(a);center=inv(a+.5)-.5
    src=Path(r['source_path']);crop=Path(r['crop_path'])
    records.append(dict(frame=frame,source_matches_current59=(sha(src)==r['source_sha256']==raw[frame]['source_sha256']),
      crop_matches_protocol=sha(crop)==r['crop_sha256'],
      direct_formula_max_px=float(np.max(np.abs(direct-n[frame+'_native_xy_direct_H']))),
      center_formula_max_px=float(np.max(np.abs(center-n[frame+'_native_xy_pillow_center_assumption']))),
      convention_distance_px=float(np.linalg.norm(direct-center,axis=-1).max()),
      raw_detection_accepted_for_fitting=read(F/(frame+'.json'))['accepted_for_fitting']))
tracker=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/VHAP_smplx/vhap/model/tracker_smplx.py')
out=dict(created=datetime.datetime.now().astimezone().isoformat(),state='raw_sources_and_formula_inventory_complete_visual_admission_pending',
    raw_frames=len(records),current_frames=59,missing_frames=[k for k in raw if k not in {r['frame'] for r in records}],
    records=records,raw_landmark_sha256=sha(F/'landmarks.npz'),reference_code_sha256=sha(tracker),
    body_updates=0,gaussian_updates=0,old_personalized_state_loaded=False,accepted_for_fitting=False,
    note='Only raw detection/source identity and declared transforms checked. Does not establish FAN pixel convention, visibility, landmark accuracy, or compatibility with51 model points. No old fitted face parameters read.')
assert all(r['source_matches_current59'] and r['crop_matches_protocol'] for r in records)
dest=B/'current/smplx-observable-landmark-audit-044/face-input-inventory.json'
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ['state','raw_frames','current_frames','accepted_for_fitting','body_updates']},ensure_ascii=False))
print('maximum convention difference',max(r['convention_distance_px'] for r in records))
