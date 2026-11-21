from pathlib import Path
import json
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human';OUT=LAB/'outputs/dynamic-human-enerf-mask-uncertainty-053'
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
save(OUT/'input-review/visual-review.json',dict(status='12largest_native_components_and3whole_panels_reviewed',whole_cameras=['01','08','11'],
 findings='Disagreement includes both wall between arm/torso and real skin, shirt and trouser fragments. This supports treating it as unknown rather than forcing background. Original SAM falsely omits some forearm parts; currenthybrid fills visibleparts pluswall.',
 selection='Twelve largest connected disagreement components byarea; not selected using reconstruction results.',no_new_segmentation_ground_truth=True))
p=OUT/'protocol.json';r=json.loads(p.read_text());r['fixed']=r['fixed'].replace('Same500baseline','Same050baseline');save(p,r)
t=(S/'compare_enerf_absgrad.py').read_text().replace('dynamic-human-enerf-absgrad-052','dynamic-human-enerf-mask-uncertainty-053')
t=t.replace("'Standard gradient'", "'Full supervision'").replace("'Absolute gradient'", "'Uncertainty ignored'")
(S/'compare_enerf_mask_uncertainty.py').write_text(t)
print('Input reviewed; same fixed regions and comparison prepared')
