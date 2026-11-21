"""Reuse original-pixel review; compare seven common source frames honestly."""
import inspect,sys
import review_indoor_circle16_gaussian124 as base
from train_winterhat_half8_sh0124 import OUT

BASELINE=OUT.parent/'winterhat-selected-sh0-pipeline124'
scope=dict(base.__dict__,OUT=OUT,BASELINE=BASELINE)
code=inspect.getsource(base.main)
edits=[("Path('/mnt/d/00/记录/008')","Path('/mnt/d/00/记录/001')"),
    ("oldmaps = {r['frame']:r for r in read(BASELINE/'pixel-mapping.json')['records']}","oldmaps = {}  # Old eight-view winterhat uses original whole-raster pixels."),
    ('== 16','== 8'),
    ("old = original_pixel_render(BASELINE,step,oldrows[frame],oldmaps[frame]) if frame in oldrows else None",
     "old = Image.open(BASELINE/'gaussian-run'/f'step-{step:06d}'/(frame+'_gaussian.png')).convert('RGB') if frame in oldrows else None\n        if old is not None:assert old.size==raw.size"),
    ("crops = {'body':crop_box(mask.getbbox(),raw.size,30)}", "crops = {'body':crop_box(mask.getbbox(),raw.size,30)}\n        bb=crops['body'];crops['feet']=crop_box([bb[0],bb[1]+.72*(bb[3]-bb[1]),bb[2],bb[3]],raw.size)"),
    ('室内一圈16帧124','冬帽半圈8帧124'),('sixteen_view_gaussian','eight_view_gaussian'),
    ('sixteen','eight'),('old_six','old_eight'),('common_six','common_seven'),
    ("[('body',430),('face',240),('left_hand',200),('right_hand',200)]","[('body',430),('face',240),('left_hand',200),('right_hand',200),('feet',220)]"),
    ('feet_quality_excluded=True','feet_quality_excluded=False,off_raster_and_occluded_feet_unknown=True'),
    ('new roster and new joint fit change together, not view-count-only ablation.','Roster, neutral-seed joint fit, uniform hand2D abstention, ROI2x, background and18k time schedule change together; no single-cause attribution.'),
    ('SIXTEEN_GAUSSIAN_REVIEW_READY','WINTERHAT_EIGHT_GAUSSIAN_REVIEW_READY'),
    ("    oldrows = {r['frame']:r for r in read(BASELINE/'data/training-inputs.json')['records']}", "    oldrows = {r['frame']:r for r in read(BASELINE/'data/training-inputs.json')['records']}\n    assert len(oldrows)==8\n    assert len(set(oldrows)&{r['frame'] for r in rows})==7"),
    ("for region in ('body','face','left_hand','right_hand'):", "for region in ('body','face','left_hand','right_hand','feet'):"),
    ('common_seven_foreground_mae=common_comparison,', 'common_seven_foreground_mae=common_comparison, baseline_frames=8, candidate_frames=8, common_source_frames=7,'),
    ("candidate_png_sha256=sha(snap/(frame+'_gaussian.png')),same_original_source_comparison=old is not None", "source_image_sha256=source['source_sha256'],source_mask_sha256=source['mask_sha256'],candidate_png_sha256=sha(snap/(frame+'_gaussian.png')),same_original_source_comparison=old is not None")]
for old,new in edits:
    assert old in code,old;code=code.replace(old,new)
exec(compile(code,'<same-native-pixel-winterhat-review>','exec'),scope)
if __name__=='__main__':scope['main'](int(sys.argv[1]))
