"""Read-only all59 same-step rendered-image comparison; not independent geometry."""
from datetime import datetime,timezone
from pathlib import Path
import os
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a

if os.name!='nt':
    a.REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')

ROOT=a.REC/'demo-vhap-initialization'
OUT=ROOT/'body-gaussian-smplify-direct'
SNAP=OUT/'gaussian-run/step-001000'
CONTROL=ROOT/'body-gaussian-root-sh-control'


def win(s):return s if os.name!='nt' else s.replace('/mnt/c/','C:/').replace('/mnt/d/','D:/')
def read_image(p):return np.asarray(Image.open(a.lp(p)).convert('RGB'))
def stats(v):
    v=np.asarray(v);return dict(min=float(v.min()),median=float(np.median(v)),mean=float(v.mean()),max=float(v.max()))


def main():
    rows=a.read(OUT/'data/training-inputs.json')['records']
    controls={r['frame']:r for r in a.read(CONTROL/'data/training-inputs.json')['records']}
    review=a.read(SNAP/'review/result.json');boxes={r['frame']:r['images'] for r in review['records']}
    records=[]
    for row in rows:
        other=controls[row['frame']]
        assert row['source_sha256']==other['source_sha256'] and row['mask_sha256']==other['mask_sha256']
        assert np.array_equal(row['K'],other['K']) and np.array_equal(row['w2c'],other['w2c'])
        raw=read_image(win(row['image']))
        mask=np.asarray(Image.open(win(row['mask'])).convert('L'))>127
        direct=read_image(SNAP/(row['frame']+'_gaussian.png'))
        baseline=read_image(CONTROL/'gaussian-run/step-001000'/(row['frame']+'_gaussian.png'))
        assert raw.shape==direct.shape==baseline.shape==(1520,2704,3)
        de=np.abs(direct.astype(np.float32)-raw).mean(-1)/255
        be=np.abs(baseline.astype(np.float32)-raw).mean(-1)/255
        regions={}
        for item in boxes[row['frame']]:
            x0,y0,x1,y1=item['crop_integer_xyxy'];m=mask[y0:y1,x0:x1]
            if m.any():
                d=float(de[y0:y1,x0:x1][m].mean());b=float(be[y0:y1,x0:x1][m].mean())
                regions[item['region']]=dict(direct=d,vhap_root_sh=b,relative_change=d/b-1 if b else None,pixels=int(m.sum()))
        records.append(dict(frame=row['frame'],regions=regions,foreground_rgb_MAE=dict(direct=float(de[mask].mean()),vhap_root_sh=float(be[mask].mean())),
            outside_mask_rgb_energy=dict(direct=float(direct[~mask].mean()/255),vhap_root_sh=float(baseline[~mask].mean()/255))))
    summary={}
    for region in ('body','face','left_hand','right_hand','feet'):
        values=[r['regions'][region] for r in records if region in r['regions']]
        changes=[r['relative_change'] for r in values]
        summary[region]=dict(frames=len(values),direct=stats([r['direct'] for r in values]),
            vhap_root_sh=stats([r['vhap_root_sh'] for r in values]),relative_change=stats(changes),
            improved_frames=int((np.asarray(changes)<0).sum()))
    run=a.read(OUT/'gaussian-run/result.json');gate=a.read(OUT/'gaussian-run/geometry-progress.json')
    data=dict(updated=datetime.now(timezone.utc).isoformat(),state='1000_step_direct_route_reviewed',frames=59,
        snapshot_step=1000,optimizer_updates=run['optimizer_updates'],training_and_59_export_seconds=run['seconds'],
        upstream_schedule_steps=600000,mesh_offset_lr=1e-6,initial_and_current_points=20908,summary=summary,records=records,
        geometry_gate=gate,comparability='Same59 pixels, masks, nativeK, view identities, step and seed. Route also adds mandatory face/skeleton exclusion; not a pure initializer-only ablation.',
        measurement='Mean absolute RGB error inside each original mask/crop; 0..1 RGB. Training views only; not independent heldout geometry or subpixel face acceptance.',
        actual_review=dict(all59_body_overviews=True,all59_native_face_hand_foot_crop_sheets=True,
            findings=['The direct SMPLify-X route renders a recognizable clothed person in all59 training views.',
                'Face and sweater remain blurred at1000; some grasping fingers fragment or disappear.',
                'Side-view shoes/ankles retain displaced or overlapping geometry; protected original face/hand/foot fitting is not repaired by this Gaussian stage.',
                'The inserted pocket hand is occluded; its hidden geometry cannot be accepted from RGB.',
                'Clothing centers can move outside the fitted body, but the mesh moves less than0.9mm; clothing-surface mesh acceptance has not been achieved.']),
        direct_route_viable_for_further_research=True,quality_accepted=False,model_promoted=False,
        new_unseen_views_tested=False,end_to_end_15min_accepted=False,
        timing_boundary='75-second figure is the Gaussian short trial and59-view export only. Reused SMPLify-X fit previously took2299.67seconds, excluding mask/preprocessing; full pipeline is not15minutes.',
        next='Keep direct SMPLify-X as the next experimental input path. Before long training, quantify remaining frozen limb/foot fitting errors and rendered-surface versus mesh gap; do not infer good geometry from low full-image RGB loss or raise learning rates blindly. No VHAP restart, teacher-center attraction or automatic threshold sweep.',
        source_hashes={str(p):a.sha(p) for p in [OUT/'plan.json',OUT/'smplify-gaussian-interface-receipt.json',SNAP/'result.json',SNAP/'current-meshes.npz',SNAP/'review/result.json']})
    a.save(OUT/'quality-review.json',data)
    a.save(SNAP/'review/visual-review.json',dict(updated=data['updated'],**data['actual_review'],quality_accepted=False))
    print({region:dict(relative_change_median=v['relative_change']['median'],improved=v['improved_frames'],frames=v['frames']) for region,v in summary.items()})


if __name__=='__main__':main()
