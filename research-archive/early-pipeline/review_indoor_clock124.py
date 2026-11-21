"""Reuse the same six-view, original-pixel ROI review with explicit baseline."""
import inspect
import sys
import review_indoor_person_zoom124 as original

def main(step):
    assert step in (1000,10000,18000)
    z=original.ROOT/'indoor-zoom-clock18k124'
    baseline=original.Z if step==1000 else original.B
    baseline_step=1000 if step==1000 else 18000
    source=inspect.getsource(original.main)
    old="    result=read(Z/'gaussian-run/result.json')\n    assert result['state']=='bounded_control_completed' and result['snapshot_step']==1000\n"
    new="    assert read(Z/'gaussian-run'/f'step-{STEP:06d}'/'result.json')['step']==STEP\n    assert read(Z/'progress.json')['state'] in ('review_pending','training','bounded_control_completed')\n"
    assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace("'gaussian-run/step-001000'","f'gaussian-run/step-{STEP:06d}'",1)
    source=source.replace("B/'gaussian-run/step-001000'","B/'gaussian-run'/f'step-{BASE_STEP:06d}'")
    old="        assert enlarged.size==tuple(row['wh']) and baseline.size==raw.size\n        roi=tuple(mapping['crop_xyxy']);x0,y0,x1,y1=roi\n"
    new="        assert enlarged.size==tuple(row['wh'])\n        roi=tuple(mapping['crop_xyxy']);x0,y0,x1,y1=roi\n        if baseline.size!=raw.size:\n            assert baseline.size==tuple(row['wh'])\n            reduced=baseline.resize((x1-x0,y1-y0),Image.Resampling.BOX)\n            baseline=Image.new('RGB',raw.size,'black');baseline.paste(reduced,(x0,y0))\n"
    assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace('高斯_室内人物放大2倍124_001000_', '高斯_室内放大时序18k124_{STEP:06d}_')
    source=source.replace("step=1000,records=records", "step=STEP,baseline=str(B),baseline_step=BASE_STEP,records=records")
    source=source.replace("['source','current_mesh','baseline1000','crop2x1000_downsampled']", "['source','current_mesh',f'baseline{BASE_STEP}',f'zoom_clock{STEP}']")
    scope=dict(original.main.__globals__,Z=z,B=baseline,STEP=step,BASE_STEP=baseline_step)
    exec(compile(source,'<unchanged-six-view-review-clock-candidate>','exec'),scope)
    scope['main']()

if __name__=='__main__':main(int(sys.argv[1]))
