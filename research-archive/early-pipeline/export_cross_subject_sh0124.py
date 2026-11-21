"""Reuse rigid export/readback; specialize the existing exporter to SH0."""
import argparse, ast, difflib, fcntl, inspect, json, os, sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src','/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import export as exp
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS
import export_demo_body_snapshot123 as old
ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')

def main(subject,step):
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=ROOT/subject;progress=rt.read(root/'progress.json')
    assert progress['state']=='bounded_control_completed',progress
    rt.configure(ASSETS)
    old.ROOT=root;row=rt.read(root/'data/training-inputs.json')['records'][0]
    old.R,_=exp.rigid_rotation64(np.diag([1.,-1.,-1.])@np.asarray(row['w2c'])[:3,:3])
    source=inspect.getsource(exp.rotate_ply)
    adapted=source.replace("+[f'f_rest_{i}' for i in range(45)]",'')
    adapted=adapted.replace('Incomplete standard degree-3 Gaussian PLY.','Incomplete standard SH0 Gaussian PLY.')
    begin=adapted.index('    rng=np.random.default_rng');end=adapted.index('    # Normals are unused')
    adapted=adapted[:begin]+"    if any(k.startswith('f_rest_') for k in names):raise RuntimeError('SH0 export expected, do not drop coefficients silently.')\n    basis_error=0.0  # Degree0 is invariant; all DC floats remain unchanged.\n"+adapted[end:]
    scope=dict(exp.rotate_ply.__globals__);exec(compile(adapted,'<same-rigid-export-degree0>','exec'),scope)
    old.rotate_ply=scope['rotate_ply']
    old.prepare(step)
    out=root/'delivery'/f'step-{step:06d}'
    (out/'degree0-export.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),adapted.splitlines(True))))
    context=rt.read(out/'context.json');context.update(label=f'{subject} · SMPLify-X直接SH0 · {step:,}步',world='World coordinates rigidly rotated to reference-camera +Y-up preview; not measured gravity or physical scale')
    rt.save(out/'context.json',context)
    src=inspect.getsource(old.replay)
    src=src.replace("    cmd, state, _ = process(suspended_pid)\n    assert 'train_demo_body_upstream_offset123.py train' in cmd and state == 'T'\n",'')
    src=src.replace('source, portable = GaussianModel(3), GaussianModel(3)','source, portable = GaussianModel(0), GaussianModel(0)')
    src=src.replace("        assert process(suspended_pid)[1] == 'T', 'Trainer resumed during export replay'\n",'')
    src=src.replace('        direction = -center/np.linalg.norm(center)',"        camera_eye=np.asarray(rt.read(out/'cameras.json')[0]['position'])\n        direction=(camera_eye-center)/np.linalg.norm(camera_eye-center)")
    ns=dict(old.replay.__globals__);exec(compile(src,'<completed-run-native-rigid-readback>','exec'),ns);ns['replay'](step,None)
    valid=rt.read(out/'validation.json');valid['coordinate_space']=context['world'];rt.save(out/'validation.json',valid)
    a=exp.PlyData.read(root/'gaussian-run'/f'step-{step:06d}'/'world.ply')['vertex'].data
    b=exp.PlyData.read(out/'point_cloud.ply')['vertex'].data
    assert a.dtype.names==b.dtype.names and len(a)==len(b)
    same={k:a[k].tobytes()==b[k].tobytes() for k in ['f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2']}
    assert all(same.values())
    rt.save(out/'sh0-field-check.json',dict(fields=list(b.dtype.names),unchanged_fields=same,sha256=rt.sha(out/'point_cloud.ply'),points=len(b),passed=True))
    print('CROSS_PERSON_NATIVE_EXPORT_PASSED',subject,step,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('subject');ap.add_argument('--step',type=int,default=18000);args=ap.parse_args();main(args.subject,args.step)
