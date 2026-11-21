from pathlib import Path
import json,numpy as np,cv2
from PIL import Image
from audit_enerf_input import read_cameras,digest
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-4k4d-mask-control-039';DATA=Path('/opt/gs-slam-lab/4k4d/actor5_6-mask039')
SRC=LAB/'outputs/dynamic-human-enerf-mask-quality-038/train15';MASK=SRC/'hybrid-closed-core'
report=json.loads((OUT/'input-preparation.json').read_text());before=OUT/'input-preparation-before-rgb-report-correction.json'
if not before.exists():before.write_text(json.dumps(report,indent=2)+'\n')
records=[]
for r in report['records']:
    c=r['camera'];p=DATA/'images'/c/'000750.png';assert digest(p)==r['derived_rgb_sha256']
    original=Path('/mnt/d/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6')
    assert digest(original/'images'/c/'000750.jpg')==r['source_rgb_sha256']
    assert digest(original/'bgmtv2'/c/'000750.jpg')==r['source_matte_sha256']
    image=np.array(Image.open(p));assert image.shape==(1080,1920,3)
    if c not in ['00','06','12']:
        source=np.array(Image.open(SRC/f'camera{c}-source.png'));diff=np.abs(image.astype(np.int16)-source.astype(np.int16))
        info=dict(max_abs_u8=int(diff.max()),changed_values=int(np.count_nonzero(diff)),mean_abs_u8=float(diff.mean()))
        r['segmentation_to_official_rgb_difference']=info
        # Differences from double-vs-float32 calibration are sparse subpixel remapping effects.
        assert info['mean_abs_u8']<.02 and info['changed_values']/image.size<.01
        mask=np.array(Image.open(DATA/'bgmtv2'/c/'000750.png'))
        assert mask.shape==(1080,1920) and set(np.unique(mask))=={0,255}
        assert np.array_equal(mask,np.array(Image.open(MASK/f'camera{c}-mask.png')))
        records.append(dict(camera=c,**info,mask_pixels=int((mask>0).sum())))
report['note']='Initial diagnostic mistakenly compared BGR to RGB; corrected report only, no inputs changed. Official-to-derived loader roundtrip already verified exact.'
(OUT/'input-preparation.json').write_text(json.dumps(report,indent=2)+'\n')
audit=dict(status='original_source_hashes_unchanged_15_candidate_masks_exact_native_coordinates',records=records,heldout_masks='Unmodified official decoded released mattes; heldoutRGB never used in candidate masks',goal_complete=False)
(OUT/'independent-input-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
