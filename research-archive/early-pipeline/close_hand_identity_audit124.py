"""Record actual review of the24 preregistered native-pixel evidence images."""
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a

out=a.REC/'cross-subject-sh0-124/cross-hand-identity-audit'
result=a.read(out/'result.json')
assert not a.lp(out/'visual-review.json').exists()
notes={
 'grass':{
  '000031':'Both hands distinct; visible right hand at chest and visible left hand hanging. Wrist labels roughly plausible; not subpixel truth.',
  '000032':'Same distinct visible hands as31. Fine finger accuracy is not independently measured.',
  '000074':'Orange left fingers move to the visible right chest hand while red left body wrist remains near hip. One orange outlier stays low. Far left hand is occluded.',
  '000075':'Orange/cyan hand chains overlap on the visible right hand; red left body wrist shifts upward on clothing. Occluded left-hand coordinates lack visual support.',
  '000076':'Both hand chains and both wrist labels now cluster on one visible right hand. Adjacent74/75 history shows identity collapse, not evidence of physical hand contact.',
  '000077':'Collapse persists on the visible right hand. Hidden left-hand3Dlocation remains unknown.'},
 'winterhat':{
  '000012':'Visible left hand hangs. Right fingers are scattered between torso and left hand while actual right hand is partly visible at neck; right body wrist lies near waist. Different detector heads disagree.',
  '000013':'Right finger chain returns to the partly visible neck hand, but right body wrist remains near waist. Large wrist discrepancy does not by itself identify which source to discard.',
  '000070':'Right hand visible at chest, left hand largely hidden behind torso/skirt. Distinct2Dclusters do not prove hidden left fingers correct.',
  '000071':'Same as70: right visible, left largely occluded. A no-collision result is not a visibility certificate.',
  '000104':'Orange left fingers duplicate the visible right chest hand; a low orange outlier inflates hand diameter. Left body wrist remains low; far hand occluded.',
  '000105':'Both finger chains on visible right hand while left body wrist is on torso. Unsupported hidden-hand correspondence.'},
 'sofa':{
  '000021':'Real overlapping hands: right hand in front, left watch wrist visible behind. Finger overlap must not be treated as an identity error by proximity alone; hidden left fingers uncertain.',
  '000022':'Real hand contact/overlap continues; visible wrists remain distinct. Protect from automatic overlap deletion.',
  '000025':'Real overlapping hands, visible right fingers and partly hidden left fingers. Their exact hidden coordinates remain unknown.',
  '000026':'Real overlapping hands at frozen reference. Low centroid separation is expected, not a sufficient rejection.',
  '000048':'Side view of same real hand contact; far left hand largely hidden and wrist2Dlocalization less clear. No exact finger truth.',
  '000049':'Side contact continues with hand/root projection overlap. Preserve uncertainty; do not infer identity from confidence winner.'},
 'indoor':{
  '000077':'Visible left watch hand at hip; far right arm bends behind torso. Cyan right fingers appear on visible left hand and blue right wrist on near forearm: unsupported identity.',
  '000078':'Same duplicate hand problem; blue right wrist jumps higher on near arm, not to the hidden right wrist.',
  '000082':'Both finger chains and wrist labels cluster on visible left hip hand. Far right hand remains occluded.',
  '000083':'Collapse persists on one visible left hand. No evidence of real hand-to-hand contact.',
  '000104':'Both hip hands visible in frontal view. Distinct clusters and wrist assignments plausible; partial finger occlusion remains.',
  '000105':'Same frontal distinct hands as104; exact anatomical keypoint accuracy not independently verified.'}}
records=[]
for subject, data in result['subjects'].items():
    for item in data['images']:
        key=item['frame'].split('_')[-1]
        path=item['path']; win=path[5].upper()+':/'+path[7:]
        assert a.sha(win)==item['sha256']
        records.append(dict(subject=subject,frame=item['frame'],path=win,
            sha256=item['sha256'],actual_viewed=True,native_crop_not_resized=True,note=notes[subject][key]))
assert len(records)==24
a.save(out/'visual-review.json',dict(updated=datetime.now(timezone.utc).isoformat(),
    state='all24_selected_native_images_reviewed',selected_native_images_reviewed=24,
    all373_images_reviewed=False,selection_is_explanatory_not_unbiased_validation=True,
    observations_changed=False,training_started=False,quality_accepted=False,
    confirmed_mechanism='Cross-hand identity collapse under occlusion occurs in grass, winterhat and indoor examples; real hand contact in sofa is an explicit counterexample to overlap-only deletion.',
    limitations=['Wrist disagreement alone cannot identify the wrong source: winterhat13 fingers have visible neck support while body wrist is at waist.',
      'Hand diameter can be inflated by a single outlier; relative confidence is not calibrated visibility.',
      'These observations do not uniquely explain all3Dlimb-depth errors or camera/shape errors.'],records=records))
print('CROSS_HAND_IDENTITY24_ACTUALLY_REVIEWED')
