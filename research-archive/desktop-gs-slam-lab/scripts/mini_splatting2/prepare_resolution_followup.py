"""Freeze secondary same-video evaluation before viewing resolution results."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001';SCRIPTS=Path(__file__).parent


def main():
    original=SCRIPTS/'evaluate_additional_frames.py';text=original.read_text(encoding='utf-8')
    old='from local_refinement import ROOT,read_model,scores,ssim,save_image,SUBJECT_BOX'
    assert old in text;text=text.replace(old,old+'\nfrom evaluate_head_detail import metric as detail_metric')
    old="                if div==1:\n                    row['ssim']"
    assert old in text
    text=text.replace(old,"""                if div==1:
                    head_end=y0+round((y1-y0)/3)
                    row['head_box']=[x0,y0,x1,head_end]
                    row['head']=detail_metric(pred[y0:head_end,x0:x1].cpu().numpy(),gt[y0:head_end,x0:x1].cpu().numpy())
                    save_image(dest/f'{Path(c["name"]).stem}-head.png',pred[y0:head_end,x0:x1])
                    row['ssim']""")
    old="        models[key]=dict(aggregate=aggregate,rows=rows,gaussians=len(model['means']))"
    assert old in text;text=text.replace(old,"""        aggregate['1']['head']={key:float(np.mean([r['head'][key] for r in rows if r['divisor']==1]))
            for key in next(r['head'] for r in rows if r['divisor']==1)}
"""+old)
    old="    summary=dict(status='complete',frames=24,models=models,protocol=protocol,"
    assert old in text;text=text.replace(old,"""    summary=dict(status='complete',frames=24,models=models,protocol=protocol,
        renderer='gsplat antialiased with training3Dfilter baked once; quantitative secondary same-video comparison',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric_pixels='clamped floating RGB before8bit encoding; separate from primary native8bit evaluator',""")
    target=SCRIPTS/'evaluate_resolution_additional_frames.py';assert not target.exists();target.write_text(text,encoding='utf-8')
    out=ROOT/'native-resolution-followup-protocol-001.json';assert not out.exists()
    protocol=dict(status='frozen_before_resolution_results',created=time.time(),
        models={'low':'native-resolution-cache-low-18000-s0','full':'native-resolution-cache-full-18000-s0'},
        frames=24,prior_selection='additional-frames-001/selection.json',prior_cameras='additional-frames-001/cameras.json',
        observed_groups='native-spread-observed-groups-001/protocol.json',
        masks='same interpolated subject boxes; top third for head/shoulder; no result-dependent ROI selection',
        camera_scales=[1,2,4,8],training_updates=False,independent_test=False,
        scope='Same-video development extension; real observed distance/pitch groups, not extrapolation GT',
        script_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        input_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            'additional-frames-001/selection.json','additional-frames-001/cameras.json','native-spread-observed-groups-001/protocol.json']})
    out.write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(status='prepared',frames=24,independent_test=False)))


if __name__=='__main__':main()
