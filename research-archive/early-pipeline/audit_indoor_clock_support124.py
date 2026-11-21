"""Reuse the frozen native opacity audit for only the new completed candidate."""
import inspect
from pathlib import Path
import audit_indoor_surface_support124 as original


def main():
    root=original.ROOT/'indoor-zoom-clock18k124'
    assert original.rt.read(root/'progress.json')['state']=='bounded_control_completed'
    assert original.rt.read(root/'process-exit.json')['exit_code']==0
    source=inspect.getsource(original.main)
    changes={
        "    candidates=[('base1000',B,1000),('base18000',B,18000),('zoom1000',Z,1000)]":
        "    candidates=[('zoom_clock18000',CANDIDATE,18000)]",
        "    paths=[Path(__file__),B/'plan.json',Z/'plan.json']":
        "    paths=[Path(__file__),WRAPPER,CANDIDATE/'plan.json',B/'plan.json']",
        "inputs='Same frozen indoor6 source views; baseline1000, baseline18000 and ROI2x1000.'":
        "inputs='Only completed ROI2x clock18000; original frozen indoor6 source cameras. Earlier 18 replays are not repeated.'",
        "purpose='Differentiate observed-view opacity gaps and point allocation from unobserved-view failure before new training.'":
        "purpose='Measure remaining observed-view opacity gaps and point allocation after the single registered clock candidate; no optimizer updates.'",
        "(pics[0].width*5,pics[0].height)":
        "(pics[0].width*len(pics),pics[0].height)",
        "分析_室内破碎支持124_{frame}_原图_mask_原1000alpha_原18000alpha_放大1000alpha.png":
        "分析_室内放大时序124_18000_{frame}_原图_mask_alpha.png",
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    scope=dict(original.main.__globals__,CANDIDATE=root,OUT=root/'surface-support',WRAPPER=Path(__file__))
    exec(compile(source,'<same-native-alpha-audit-new-candidate-only>','exec'),scope)
    scope['main']()


if __name__=='__main__':main()
