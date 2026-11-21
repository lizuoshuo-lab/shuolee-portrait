"""Prove duplicate wrist semantics and quantify conflicting saved observations."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import numpy as np
import audit_gaussianavatars_params123 as a

ROOT=a.REC/'cross-subject-sh0-124'
OUT=ROOT/'selected-observation-audit124'


def main():
    assert not a.lp(OUT/'duplicate-wrist-result.json').exists()
    utils=a.REC/'smplifyx-reference-control/official/smplifyx/utils.py'
    source=a.lp(utils).read_text(encoding='utf-8')
    fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='smpl_to_openpose')
    scope={'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(utils),'exec'),scope)
    mapping=scope['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
    assert mapping[4]==mapping[46]==21 and mapping[7]==mapping[25]==20
    audit=a.read(OUT/'result.json');records=[]
    hashes={str(utils):a.sha(utils),str(OUT/'result.json'):a.sha(OUT/'result.json')}
    for subject,folder in [('grass','grass-protected-refine124'),('winterhat','winterhat-selected-view-pose124')]:
        path=ROOT/folder/'selected-fit.npz';hashes[str(path)]=a.sha(path)
    a.save(OUT/'duplicate-wrist-plan.json',dict(updated=datetime.now(timezone.utc).isoformat(),
        scope='Read-only original wrist index aliases on the same16frames. No fit, removal, threshold change or automatic acceptance.',
        source_hashes=hashes,budget=dict(frames=16,CPU_seconds=120,GPU_fits=0)))
    for subject,folder in [('grass','grass-protected-refine124'),('winterhat','winterhat-selected-view-pose124')]:
        path=ROOT/folder/'selected-fit.npz'
        with np.load(a.lp(path),allow_pickle=False) as z:
            ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'].copy();proj=z['joints_projected_native'].copy()
        for i,f in enumerate(ids):
            height=next(r['height_px'] for r in audit['records'] if r['subject']==subject and r['frame']==f)
            for side,body,hand in [('right',4,46),('left',7,25)]:
                exact=float(np.linalg.norm(proj[i,body]-proj[i,hand]));assert exact<1e-5
                distance=float(np.linalg.norm(obs[i,body,:2]-obs[i,hand,:2]))
                records.append(dict(subject=subject,frame=f,side=side,body_index=body,hand_index=hand,
                    smplx_joint=int(mapping[body]),body_observed_xy=obs[i,body,:2].tolist(),hand_observed_xy=obs[i,hand,:2].tolist(),
                    body_confidence=float(obs[i,body,2]),hand_confidence=float(obs[i,hand,2]),
                    observed_gap_px=distance,gap_over_height=distance/height,
                    best_possible_unweighted_max_error_lower_bound_px=distance/2,
                    fitted_alias_projection_difference_px=exact))
    assert all(a.sha(Path(p))==digest for p,digest in hashes.items())
    report=dict(updated=datetime.now(timezone.utc).isoformat(),state='duplicate_wrist_conflicts_quantified',
        source_hashes=hashes,records=records,
        proven='Original SMPL-X wrist joint21 is used at OpenPose body4 and right hand46; joint20 at body7 and left hand25. Actual fitted projections coincide.',
        interpretation='These are repeated measurements of one3D joint, not separate body and hand anatomy. A large target separation is an internally inconsistent2D input; no single projected wrist can equal both targets.',
        limits='Relative confidence is not calibrated visibility. Conflict alone does not say which target is correct or establish the whole cause of a pose error. Lower bound is unweighted geometry, not the robust training loss.',
        changed_selection=False,changed_training=False,quality_accepted=False)
    a.save(OUT/'duplicate-wrist-result.json',report)
    for r in sorted(records,key=lambda r:r['observed_gap_px'],reverse=True)[:12]:print(r)


if __name__=='__main__':main()
