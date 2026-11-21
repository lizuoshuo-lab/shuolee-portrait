"""Freeze a DISJOINT calibration segment; prohibited as appearance data."""
from pathlib import Path
import json,copy,argparse
from PIL import Image
from audit_enerf_input import digest
from enerf_native_loader import EnerfNativeDataset
LAB=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser();p.add_argument('--out',default=str(LAB/'outputs/dynamic-human-enerf-time-calibration-028'));p.add_argument('--anchors',type=int,nargs=3,default=[109,129,149]);p.add_argument('--selection-receipt');a=p.parse_args();source=LAB/'outputs/dynamic-human-enerf-full-023/actor1_4-native-input-audit.json';out=Path(a.out);out.mkdir(parents=True,exist_ok=True);dest=out/'calibration-only-manifest.json'
assert not dest.exists()
original=json.loads(source.read_text());spec=copy.deepcopy(original);root=Path(spec['root']);frames=[t+d for t in a.anchors for d in range(-3,4)];assert set(frames).isdisjoint(original['selected_frames']);rows=[]
for cam in sorted(spec['cameras']):
    for t in frames:
        p=root/'images'/cam/f'{t:06d}.jpg';m=root/'bgmtv2'/cam/p.name
        for f in [p,m]:
            with Image.open(f) as im:im.load();assert im.size==(1920,1080)
        rows.append(dict(camera=cam,frame=t,path=p.relative_to(root).as_posix(),width=1920,height=1080,sha256=digest(p),split='train',mask_path=m.relative_to(root).as_posix(),mask_sha256=digest(m),mask_semantics='Author background matting; timing feature selection only'))
spec.update(status='separate_temporal_calibration_source_files_verified',purpose='separate_temporal_calibration_only',appearance_training_permitted=False,selected_frames=frames,image_count=len(rows),heldout_cameras=[],heldout_frames=[],images=rows,split_counts=dict(train=len(rows),heldout_view=0,heldout_time=0),fit_anchor_frames=[109,129],independent_check_anchor_frames=[149],original_appearance_manifest_sha256=digest(source),split_scope='All18 cameras in disjoint calibration segment106..152. Camera00,06,12 images in0..59 remain appearance-heldout, but these cameras are now calibration-visible. No image from0..59 used to fit this protocol.',initialization='FORBIDDEN for Gaussian initialization, geometry fitting or appearance training',source_geometry='Not used',unresolved=['Empirical relative frame index; not hardware timestamp calibration','Offset stationarity across time','Moving-point match identity and shutter blur'])
spec['fit_anchor_frames']=a.anchors[:2];spec['independent_check_anchor_frames']=a.anchors[2:]
spec['split_scope']=f'All18 cameras in disjoint calibration windows around{a.anchors}. Camera00,06,12 images in0..59 remain appearance-heldout, but these cameras are now calibration-visible. No image from0..59 used to fit this protocol.'
if a.selection_receipt:spec['anchor_selection_sha256']=digest(a.selection_receipt)
dest.write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8')
try:EnerfNativeDataset(dest,'train')
except ValueError as e:assert 'Calibration-only' in str(e)
else:raise AssertionError('Calibration-only guard failed')
print(json.dumps(dict(manifest=str(dest),images=len(rows),frames=frames,appearance_loader_guard='passed')))
