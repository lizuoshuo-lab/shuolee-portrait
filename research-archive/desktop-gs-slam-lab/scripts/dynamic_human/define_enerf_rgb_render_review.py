from pathlib import Path
import json
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
assert not (R/'visual-review-plan.json').exists()
files=[f'paired-comparison/camera{c}-frame{f:06d}-full-person.png' for c in ['01','04','08','17'] for f in [745,751,755]]
files += [f'paired-comparison/camera{c}-frame000751-{part}.png' for c in ['01','04'] for part in ['dark_forearm','light_forearm','dark_shoe','dark_head']]
files += [f'heldout/paired-comparison/camera{c}-frame{f:06d}-full-person.png' for c in ['00','06','12'] for f in [745,751,755]]
files += [f'heldout/paired-comparison/camera{c}-frame000751-{part}.png' for c in ['00','06','12'] for part in ['dark_right_forearm','light_left_forearm']]
files += ['heldout/paired-comparison/camera06-frame000745-light_left_forearm.png','heldout/paired-comparison/camera06-frame000755-dark_right_forearm.png']
assert len(files)==37 and len(set(files))==37
out=dict(status='fixed_while15000update_training_running_before_metrics_and_render_review',files=files,
 additional='AlsoinspectworstforegroundPSNRregressionknownandnewviewifnotalreadyincluded.',
 targets='Fingers,forearmcontours,headtexture,airborneshoe,cheststreaks,handocclusionghosts. Allpairedregions reusedfrompreviouscontrols; originalsourcepixelsnotenhanced.',
 decision='Keep067pendingcompletednativecomparisons; noacceptancefrominputrepairorposefitlossalone.',goal_complete=False)
(R/'visual-review-plan.json').write_text(json.dumps(out,indent=2)+'\n');print('37native panels and worst-case additions fixed before viewing074renders.')
