"""Separate upstream fitted joints from the broader existing quality screen."""
import ast,re
import numpy as np
from pathlib import Path
import audit_gaussianavatars_params123 as a
from quality_budget124 import verify
from select_quality_frames124 import POLICY

OUT=a.REC/'cross-subject-sh0-124/grass-selected-view-refine124'
BASE=OUT.parent/'grass-selected-view-pose124'
H=Path(__file__).resolve().parent

def main():
    assert not a.lp(OUT/'screen-contract-audit.json').exists()
    official=a.REC/'smplifyx-reference-control/official'
    cfg=a.lp(official/'cfg_files/fit_smplx.yaml').read_text(encoding='utf-8')
    ignored=[int(s) for s in re.findall(r'^\s+-\s+(\d+)\s*$',cfg.split('joints_to_ign:')[1].split('data_folder:')[0],re.M)]
    tree=ast.parse((H/'official_smplifyx_entry123.py').read_text())
    assigns=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].value,ast.Name)
        and n.targets[0].value.id=='jw']
    assert len(assigns)==1
    active_ignored=ast.literal_eval(assigns[0].targets[0].slice.elts[1])
    assert active_ignored==ignored==[1,9,12] and ast.literal_eval(assigns[0].value)==0
    old=a.read(BASE/'selection-reviewed.json');records=[]
    with np.load(a.lp(BASE/'selected-fit.npz')) as z:
        err=np.linalg.norm(z['joints_projected_native']-z['observed_native_keypoints'][:,:,:2],axis=2)
        conf=z['observed_native_keypoints'][:,:,2]
    body=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    for i,r in enumerate(old['records']):
        full=body[conf[i,body]>0];active=np.array([j for j in full if j not in ignored])
        j=int(full[np.argmax(err[i,full])]);h=r['person_box_height_px']
        records.append(dict(frame=r['frame'],maximum_body_joint=j,maximum_is_ignored_by_fit=j in ignored,
            full_body_max_over_height=float(err[i,full].max()/h),active_body_max_over_height=float(err[i,active].max()/h),
            ignored_hip_errors_px={str(k):float(err[i,k]) for k in [9,12]},original_screen_flags=r['screen_flags']))
    a.save(OUT/'screen-contract-audit.json',dict(upstream_cfg_ignored=ignored,actual_adapter_ignored=active_ignored,
        records=records,screen_threshold=POLICY['body_max_over_height'],training_changed=False,
        decision='The existing hard screen measures broader detected-body consistency than the optimized joint set. Keep both scopes visible; ignored hip errors are not zeroed or hidden and the3percent gate remains unchanged this round. Being excluded from a fit alone does not prove that a validation point is invalid.',
        source_hashes={str(p):a.sha(p) for p in [official/'cfg_files/fit_smplx.yaml',H/'official_smplifyx_entry123.py',BASE/'selected-fit.npz',H/'select_quality_frames124.py']}))
    a.save(OUT/'budget-selection-plan.json',dict(method='Existing quality and pose gates unchanged. First prefer one best per angular bin; fill any remaining count from eligible frames by angular diversity and deterministic score/id tie-break. Missing directions remain disclosed.',
        reason='User requested about8 good frames and preferably spread views; equal bins were previously an unintended hard cap on count.',
        target=8,thresholds_changed=False,no_bad_frame_backfill=True,one_source_fit_required=True,
        original_review_only_frames_excluded=True,automatic_training=False,
        tests=verify(),source_sha256=a.sha(H/'quality_budget124.py')))
    print('SCREEN_SCOPE_AUDITED_THRESHOLDS_UNCHANGED_BUDGET_SELECTOR_PASSED')

if __name__=='__main__':main()
