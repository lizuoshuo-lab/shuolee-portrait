"""Read saved official foot vertices/targets; never fit, infer, or alter inputs."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import json
import time
import numpy as np
from review_full_sequence123 import read, sha, relocate

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REF=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
F=REF/'official-first-cycle/full94-shared-shape'
OUT=REF/'official-first-cycle/shared-shape-control/foot-binding-audit'
SITE=relocate('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/lib/python3.10/site-packages/smplx')

def save(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir()
    start=time.monotonic()
    files=[F/'data/training-inputs.json',REF/'official/smplifyx/utils.py',SITE/'vertex_ids.py',SITE/'vertex_joint_selector.py',Path(__file__)]
    hashes={str(p):sha(p) for p in files}
    save(OUT/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
         scope='Verify saved current94-frame foot joints equal installed official foot vertices and native-K projection. Reuse frozen detector targets; no residual-based exclusion.',
         frames=94,cpu_seconds=90,fit_updates=0,detector_calls=0,GPU_calls=0,
         limitation='A correct channel/vertex/projection binding does not establish detector anatomical accuracy, camera truth, or shoe-surface agreement.'))
    node=next(n for n in ast.parse((SITE/'vertex_ids.py').read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='vertex_ids' for t in n.targets))
    vertex_ids=ast.literal_eval(node.value)['smplx']
    names=['LBigToe','LSmallToe','LHeel','RBigToe','RSmallToe','RHeel']
    ids=np.array([vertex_ids[x] for x in names])
    official=REF/'official/smplifyx/utils.py'
    fn=next(n for n in ast.parse(official.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='smpl_to_openpose')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(official),'exec'),scope)
    mapping=scope['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    assert np.array_equal(mapping[19:25],np.arange(60,66))
    rows=read(F/'data/training-inputs.json')['records'];assert len(rows)==94
    results=[];max_xyz=max_uv=0.
    with np.load(F/'fit/fitted.npz',allow_pickle=False) as fit:
        targets=fit['observed_native_keypoints'].copy();assert list(fit['frame_ids'].astype(str))==[r['frame'] for r in rows]
    for i,row in enumerate(rows):
        assert time.monotonic()-start<90
        path=F/'frames'/row['frame']/'winner-camera-space.npz'
        with np.load(path,allow_pickle=False) as z:
            assert str(z['frame_id'].item())==row['frame']
            assert np.array_equal(z['K_integer_centers'],row['K_integer_centers'])
            vertices=z['vertices_camera'][0,ids];joints=z['joints_camera'][0,19:25]
            max_xyz=max(max_xyz,float(np.max(np.abs(vertices-joints))))
            p=vertices.astype(np.float64)@np.asarray(row['K_integer_centers']).T
            uv=p[:,:2]/p[:,2:]
            max_uv=max(max_uv,float(np.max(np.abs(uv-z['joints_projected_native'][0,19:25]))))
            err=np.linalg.norm(uv-targets[i,19:25,:2],axis=1)
        results.append(dict(frame_id=row['frame'],source_sha256=sha(path),
            feet=[dict(name=name,source_coco_channel=17+j,official_slot=19+j,smplx_joint=60+j,vertex=int(ids[j]),
                 target_xy=targets[i,19+j,:2].tolist(),model_xy=uv[j].tolist(),
                 confidence=float(targets[i,19+j,2]),error_px=float(err[j])) for j,name in enumerate(names)]))
    assert max_xyz==0. and max_uv<.001
    assert all(sha(Path(p))==h for p,h in hashes.items())
    save(OUT/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),state='completed_readonly',
         frames=94,actual_mesh_vertex_binding_passed=True,official_map_passed=True,
         maximum_xyz_difference=max_xyz,maximum_native_projection_difference_px=max_uv,rows=results,
         fit_updates=0,GPU_calls=0,detector_calls=0,physical_truth_verified=False,
         conclusion='The saved current official foot channels use the correct official vertices and same native K; this does not explain remaining detector-fit or barefoot/shoe mismatch.',
         seconds=time.monotonic()-start))
    print(json.dumps(dict(frames=94,max_xyz=max_xyz,max_uv_px=max_uv,seconds=time.monotonic()-start)))

if __name__=='__main__':main()
