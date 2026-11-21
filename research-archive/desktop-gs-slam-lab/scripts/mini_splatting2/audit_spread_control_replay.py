"""CPU audit of the zero-variance control before interpreting candidate gains."""
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from evaluate_head_detail import metric

SCRIPTS=Path(__file__).parent
ROOT=SCRIPTS.parents[1]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')


def main():
    target=ROOT/'native-spread-zero-control-audit.json';assert not target.exists()
    source=RUNTIME/'repo-depthguide';replica=RUNTIME/'repo-depthspread'
    unchanged={}
    for path in source.rglob('*.py'):
        if any(s in path.parts for s in ['submodules','.git','__pycache__']):continue
        relative=path.relative_to(source);copy=replica/relative
        assert copy.exists() and path.read_bytes()==copy.read_bytes(),relative
        unchanged[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    guided=(source/'msv2/train_depth_guided.py').read_text()
    augmented=(replica/'msv2/train_depth_spread.py').read_text()
    lines=[line for line in augmented.splitlines(keepends=True)
        if 'from native_depth_spread import' not in line
        and 'spread_guidance = DepthSpread(' not in line
        and 'loss = loss + spread_guidance(' not in line
        and 'spread_guidance.finish()' not in line]
    assert ''.join(lines)==guided
    ref=ROOT/'native-depthguide003-18000-s0';control=ROOT/'native-depthspread000-18000-s0'
    assert (ref/'config').read_bytes()==(control/'config').read_bytes()
    usage=json.loads((control/'model/spread-guidance-usage.json').read_text())
    assert usage['coefficient']==0 and usage['active_calls']==0
    summaries=[json.loads((p/'evaluation-summary.json').read_text()) for p in [ref,control]]
    assert [r['frame'] for r in summaries[0]['native_rows']]==[r['frame'] for r in summaries[1]['native_rows']]
    assignment=next(n for n in ast.parse((SCRIPTS/'local_refinement.py').read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value);rows=[]
    for i,r in enumerate(summaries[0]['native_rows']):
        relative=Path('model/test/ours_18000');filename=f'{i:05d}.png'
        assert (ref/relative/'gt'/filename).read_bytes()==(control/relative/'gt'/filename).read_bytes()
        gt=np.asarray(Image.open(ref/relative/'gt'/filename).convert('RGB'),dtype='float32')/255
        h,w=gt.shape[:2];x0,y0,x1,y1=[round(a*b) for a,b in zip(boxes[int(r['frame'])],[w,h,w,h])]
        y1=y0+round((y1-y0)/3);data={}
        for key,path in [('reference',ref),('control',control)]:
            pred=np.asarray(Image.open(path/relative/'renders'/filename).convert('RGB'),dtype='float32')/255
            data[key]=metric(pred[y0:y1,x0:x1],gt[y0:y1,x0:x1])
        rows.append(dict(frame=r['frame'],**data))
    head_delta={k:float(np.mean([r['control'][k]-r['reference'][k] for r in rows])) for k in rows[0]['control']}
    result=dict(status='complete',created=time.time(),all_inherited_python_files_identical=unchanged,
        existing_training_code_preserved=True,configuration_identical=True,
        zero_variance_active_calls=0,native_delta={k:summaries[1]['native_mean'][k]-summaries[0]['native_mean'][k]
            for k in summaries[0]['native_mean']},head_delta=head_delta,head_rows=rows,
        seed=0,claim_bitwise_reproducible=False,
        interpretation='The zero-weight run differs from the earlier seed0 run despite identical inherited source and configuration. Compare new variance candidates to this fresh zero-weight control, and repeat any useful candidate; do not attribute small gains from different runs to the new loss.')
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_inherited_python_files_identical','head_rows']},indent=2))


if __name__=='__main__':
    main()
