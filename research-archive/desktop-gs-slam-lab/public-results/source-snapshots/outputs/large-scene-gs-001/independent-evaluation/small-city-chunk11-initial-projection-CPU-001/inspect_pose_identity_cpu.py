from pathlib import Path
import importlib.util,json,numpy as np
p=Path(__file__).parent/'audit_projection_cpu.py';s=importlib.util.spec_from_file_location('initial_projection_readonly',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
manifest=m.read(m.INPUT/'manifest.json');config=m.read(m.TRAIN/'configuration.json');records=m.camera_records(m.LOCK/'creator-camera-archives/1_1/sparse/0/images.bin')
with np.load(Path(manifest['protocol_source'])/'global-cameras.npz',allow_pickle=False) as z:names=z['names'].tolist();poses=z['c2w'].copy();K=z['K'].copy()
rows=[]
for name in config['training_names'][:10]:
    rec=records[name];Rt=np.eye(4);Rt[:3,:3]=m.rot(rec['qvec']);Rt[:3,3]=rec['tvec'];c2w=np.linalg.inv(Rt);j=names.index(name)
    rows.append(dict(name=name,qvec=rec['qvec'].tolist(),source_tvec=rec['tvec'].tolist(),original_C2W=c2w.tolist(),global_C2W=poses[j].tolist(),C2W_max_abs_difference=float(np.max(np.abs(c2w-poses[j]))),global_K=K[j].tolist()))
m.write(Path(__file__).parent/'pose-identity-inspection.json',rows)
print(json.dumps(rows[0],indent=2))
