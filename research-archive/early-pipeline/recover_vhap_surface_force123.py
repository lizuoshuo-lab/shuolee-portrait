"""Recover saved audit arrays after a final JSON numpy-int crop failure.

No GPU rerun; unavailable scalar losses and non-offset gradients stay unknown.
"""
import json
import numpy as np
from pathlib import Path
import audit_gaussianavatars_params123 as a


def stats(v):
    x=np.asarray(v,dtype=float)
    return dict(minimum=float(x.min()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(x.max()))


def main():
    root=a.REC/'demo-vhap-initialization'
    out=root/'vhap-surface-force-audit'
    plan=a.read(root/'vhap-surface-force-plan.json')
    failure=a.read(out/'failure.json')
    assert 'int64' in failure['error'] and "save(OUT/'result.json',result)" in failure['traceback']
    source=a.H/'audit_vhap_surface_forces123.py'
    assert a.sha(source)==plan['source_sha256']
    with np.load(a.lp(out/'offset-gradients.npz'),allow_pickle=False) as z:
        assert list(z['frame_ids'])==plan['roster']
        gradients={k:z[k].astype(np.float64) for k in z.files if k!='frame_ids'}
    assert all(g.shape==(59,10475,3) and np.isfinite(g).all() for g in gradients.values())
    log=(a.H/'audit-vhap-surface-forces.log').read_text(encoding='utf-8',errors='replace')
    records=[]
    for line in log.splitlines():
        begin=line.find('{"frame":')
        if begin >= 0:
            records.append(json.loads(line[begin:]))
    assert records and len({r['frame'] for r in records})==len(records)
    assert all(r['frame'] in plan['roster'] for r in records)
    mask=gradients['silhouette'];mn=np.linalg.norm(mask.reshape(59,-1),axis=1)
    summary={}
    for k,g in gradients.items():
        n=np.linalg.norm(g.reshape(59,-1),axis=1)
        co=np.sum(g*mask,axis=(1,2))/np.maximum(n*mn,1e-30)
        summary[k]=dict(offset_gradient_l2=stats(n),cosine_with_mask=stats(co),
            opposite_mask_frames=int((co<0).sum()),mean_gradient_l2=float(np.linalg.norm(g.mean(0))),
            coherence=float(np.linalg.norm(g.sum(0))/max(n.sum(),1e-30)),scalar_loss=None)
    assert np.max(np.abs(mask-gradients['silhouette_rest']-gradients['silhouette_far_missing']))<2e-4
    images=[]
    for frame in plan['visual_roster']:
        p=out/(frame+'_native_raw_target_mesh_mismatch.png')
        images.append(dict(frame=frame,path=str(p),sha256=a.sha(p)))
    result=dict(state='complete_saved_gradients_recovered',frames=59,optimizer_updates=0,source_sha256=a.sha(source),
        gradient_npz_sha256=a.sha(out/'offset-gradients.npz'),failure_preserved=str(out/'failure.json'),
        scalar_losses_available=False,root_and_body_gradient_stats_available=False,
        identity_checks='Original source performs parameter digest equality and all input hashes before writing the successfully saved gradient NPZ; final failure was image crop numpy.int64 serialization.',
        seconds_to_last_frame=records[-1]['seconds'],summary=summary,records=records,images=images,
        surviving_scalar_log_frames=len(records),missing_scalar_log_frames=[f for f in plan['roster'] if f not in {r['frame'] for r in records}],
        partial_log_far_missing_fraction=stats([r['far_missing_fraction'] for r in records]),
        far_gradient_max_abs=float(np.abs(gradients['silhouette_far_missing']).max()),
        RGB_gradient_present_all59=bool(np.all(np.linalg.norm(gradients['photo'].reshape(59,-1),axis=1)>0)),
        actual_visual_review=False,quality_accepted=False,training_started=False,
        limitation='Local derivatives at a rejected checkpoint; no unique cause or accepted shape. WSL failure stderr overwrote part of the shared output prefix; incomplete scalar logs are explicitly marked. The NPZ retains all59 offset gradients. No second GPU run to recover missing scalar logs.')
    a.save(out/'result-recovered.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('images','records')},ensure_ascii=False))


if __name__=='__main__':main()
