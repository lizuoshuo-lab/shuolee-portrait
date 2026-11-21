from pathlib import Path
import json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-support-correction-081';S=L/'outputs/dynamic-human-enerf-second-interval-079'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(O/'support-input-audit.json');assert len(audit['records'])==105
fallback=['08','09','10','11','14','16'];changed=[]
for r in audit['records']:
 f=f"camera{r['camera']}-frame{r['frame']:06d}.npz";z=np.load(S/'native-input'/f);q=np.load(O/'accepted-input'/f);assert sha(S/'native-input'/f)==r['input_sha256'];assert sha(O/'accepted-input'/f)==r['training_support_sha256'];assert np.array_equal(q['matte'],z['matte']*q['support']);assert np.array_equal(q['matte'][q['protected']],z['matte'][q['protected']])
 if r['camera'] in fallback:assert np.array_equal(q['matte'],z['matte'])
 else:changed.append(r)
 assert r['frame']%2==0 and r['camera'] not in ['00','06','12']
shutil.copy2(S/'graph.npz',O/'graph.npz')
save(O/'visual-review.json',dict(status='conservative_train_only_support_approved_for_matched_intervention',training_approved=True,images_reviewed=105,contact_sheets_reviewed=15,support_audit_sha256=sha(O/'support-input-audit.json'),changed_images=len(changed),fallback_original_images=42,fallback_cameras=fallback,removed_foreground_pixels=sum(r['removed_foreground_pixels'] for r in changed),actual_findings=['Initial SAM proposals deleted cow/green toys and partly occluded dark-clothed person; rejected as direct masks.','Original foreground restored in hand/object, lower-leg/foot and head/neck windows.','All seven input frames inspected for each of fifteen cameras. Six cameras with visible body deletion reverted entirely to original targets.','Other nine cameras show removal of wall-shadow regions; conservative protection retains some nearby false background. This is partial correction, not semantic ground truth.'],review_limits='400px-high per-frame sheets plus three native pilot panels, not individual pixel annotation. Final safety changes only restore original pixels, never remove additional pixels.',candidate_is_ground_truth=False,goal_complete=False))
print('APPROVED',len(changed),sum(r['removed_foreground_pixels'] for r in changed))
