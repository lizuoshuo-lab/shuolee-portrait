"""A new local calibration protocol, explicitly separate from appearance.

This does not revise previously published train/test claims. Camera00/06/12
are calibration-visible here, but their later appearance-test images remain
unused for initialization, appearance or motion optimization.
"""
from pathlib import Path
import json,copy
from datetime import datetime,timezone
from audit_enerf_input import digest
from enerf_native_loader import EnerfNativeDataset
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-local-time-029';out.mkdir(exist_ok=True);dest=out/'calibration-only-manifest.json';assert not dest.exists()
source=LAB/'outputs/dynamic-human-enerf-full-023/actor1_4-native-input-audit.json';original=json.loads(source.read_text());spec=copy.deepcopy(original);anchors=[9,29,19];frames=sorted({t+d for t in anchors for d in range(-3,4)})
rows=[dict(r,split='train') for r in original['images'] if r['frame'] in frames];assert len(rows)==18*21
root=Path(original['root'])
for r in rows:
    assert digest(root/r['path'])==r['sha256'];assert digest(root/r['mask_path'])==r['mask_sha256']
spec.update(status='local_calibration_sources_verified',purpose='local_temporal_calibration_only',appearance_training_permitted=False,selected_frames=frames,image_count=len(rows),heldout_cameras=[],heldout_frames=[],images=rows,split_counts=dict(train=len(rows),heldout_view=0,heldout_time=0),fit_anchor_frames=anchors[:2],independent_check_anchor_frames=anchors[2:],original_manifest_sha256=digest(source),split_scope='NEW exploratory protocol. All18 cameras calibrated using6..12,16..22,26..32; no appearance images from33..59 used for calibration. Camera00,06,12 are calibration-visible, later appearance-heldout, NOT globally unseen cameras.',initialization='FORBIDDEN for Gaussian initialization/appearance optimization',source_geometry='Not used',observed_utc=datetime.now(timezone.utc).isoformat())
dest.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8')
try:EnerfNativeDataset(dest,'train')
except ValueError as e:assert 'Calibration-only' in str(e)
else:raise AssertionError('Calibration-only guard missing')
protocol=dict(status='local_calibration_protocol_frozen_before_fit',fit_frames=[9,29],check_frame=19,calibration_raw_frames=frames,all18_camera_calibration_visible=True,appearance_test_cameras=['00','06','12'],appearance_common_frame_window=[34,58],appearance_window_condition='May shrink endpoints before training if estimated offset would require calibration pixels or source outside audited0..59. Do not expand without auditing.',acceptance_gate='Connected fit graph; independent anchor19 mean pair moving median residual at least20percent lower than zero; at least75percent of changed pairs with>=12check matches improve. Otherwise retain hypothesis as rejected diagnostic and do not launch retimed sequence.',scope='Development protocol; earlier0..59 images already inspected in prior experiments, not an untouched final benchmark. First pair only tests novel appearance viewpoints over continuous motion; separate temporal holdout remains required.',no_calibration_photos_for_appearance=True,source_sha256=digest(source))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=protocol['status'],images=len(rows))))
