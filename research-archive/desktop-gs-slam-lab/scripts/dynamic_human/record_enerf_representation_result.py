from pathlib import Path
import json,hashlib,shutil
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-representation-048'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
paired=read(OUT/'paired-comparison/summary.json');comp=read(OUT/'first-frame/completion.json')
assert read(OUT/'native-evaluation/independent-image-audit.json')['full_pngs']==24
findings={
'00':'Shirt folds, hair and printed pattern sharper; moving left arm and toy are stretched/misplaced. Heldout foreground metric worsens.',
'06':'Cleaner silhouettes and shirt folds but long streaks at both arms and disocclusion; hidden actor shoulder has an extended narrow artifact.',
'12':'Sharper head and clothes, but left forearm/fingers and airborne shoe remain smeared.',
'01':'Training view improves clothes and hair substantially; hand/forearm and toy still have stretched appearance.',
'08':'Training view becomes much cleaner around torso and legs; hidden actor hand remains distorted.',
'17':'Training view improves broad surfaces; hand detail and partially visible held object remain unresolved.'}
items=[]
for r in paired['records']:
 p=OUT/'paired-comparison'/r['panel'];items.append(dict(file=str(p.relative_to(OUT)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),finding=findings[r['camera']]))
save(OUT/'visual-review.json',dict(status='six_native_paired_person_panels_reviewed',items=items,
 additional_review='Full native training-camera01 progress render and heldout06 two-column reference comparison also viewed.',
 decision='Useful direct3DGS alternative, but not accepted for clear moving limbs or across-view quality.',goal_complete=False))
save(OUT/'final-summary.json',dict(status='direct_anisotropic3dgs_sharper_broad_surfaces_moving_limbs_and_heldout_consistency_unresolved',
 updates=6000,gaussians=comp['points'],native_resolution=[1920,1080],training_cameras=15,appearance_heldout_cameras=3,
 foreground_psnr_delta_vs_044={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in paired['records']},
 conclusion='All three inspected training views improve, heldout00 worsens1.466dB,06 nearlyunchanged(-.066dB),12 improves.952dB. Actual renders show sharper hair/clothing but long arm/hand streaks. No broad clarity acceptance.',
 qualification='Two training pipelines, not an isolated representation ablation; initialization, loss, densification and budget differ. Same development instant, no temporal/fullscene result.',goal_complete=False))
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['evaluate_enerf_representation.py','compare_enerf_representation.py','record_enerf_representation_result.py','audit_4k4d_native_images.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
print(json.dumps(read(OUT/'final-summary.json')))
