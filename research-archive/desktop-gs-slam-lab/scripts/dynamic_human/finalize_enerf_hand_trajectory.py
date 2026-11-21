"""Record actual visual review after numerical audits and preserve native panels."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-trajectories-095'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'final-summary.json').exists()
s=read(O/'comparison/summary.json');m=read(O/'motion-audit.json');t=read(O/'temporal/summary.json')
assert s['PNG_checks']==1072 and s['max_recomputed_metric_difference']<1e-9
assert len(t['panels'])==26 and len(read(O/'temporal/completion.json')['records'])==104
for b in ['nodes','points']:
 assert sha(O/b/'resume.pt')==s['checkpoints'][b]==m['records'][b]['checkpoint_sha256']==read(O/b/'completion.json')['checkpoint_sha256']
 assert m['records'][b]['numpy_max_absolute_error']<1e-8 and m['records'][b]['contexts']==234
files=read(O/'visual-review-plan.json')['sheets']+read(O/'full-review-plan.json')['files']
assert len(files)==27
save(O/'visual-review.json',dict(status='27sheets_actually_viewed_no_clarity_promotion',files=[dict(file=Path(p).as_posix(),sha256=sha(O/p)) for p in files],coverage=dict(same_time_hand_views=15,target_hand_cases=6,hand_object_cases=12,consecutive_hand_frames_per_camera=13,temporal_cameras=['04','13'],whole_person_cases=[['10',729],['15',731],['12',729],['00',732],['06',732],['12',732]]),findings=['Both motion variants retain broken/merged fingers. Point model has conspicuous dark palm spots in01/03/04/09/10 at732.', '04 and13 consecutive image sequences retain grasp silhouette initially, then fingerfragmentation as palm opens near727..732; newmotion does not recover clear anatomy.', 'Worst known andheldout fullperson views retain torso/clothing/footoutline; hand/toy errors and preexisting backgroundfloaters remain. Fullbodyimages do not supersede native detail review.'],video_scope='All26exactframe panels viewed; videos verified by decode. No realtime playback QA claimed.',clarity_accepted=False,goal_complete=False))
# Correct a metadata-only hardcoded counter; numeric loop/status already checked104.
assert t['PNG_checks']==78;t['counter_correction']='PNG_checks was hardcoded78; 4branches x2cameras x13frames=104. Actual loop asserted104 and saved104records. No metric or image changed.';t['PNG_checks']=104;t['visual_review']='../visual-review.json';t['status']='104nativePNGs_26panels_24transitions_2decodedvideos_verified_frames_visually_reviewed';save(O/'temporal/summary.json',t)
p=L/'scripts/dynamic_human/audit_enerf_hand_trajectory_temporal.py';code=p.read_text();assert 'PNG_checks=78' in code;code=code.replace('PNG_checks=78','PNG_checks=104');p.write_text(code);shutil.copy2(p,O/'temporal'/p.name)
s['visual_review']='../visual-review.json';s['status']='134contexts_4models_1072PNGs_verified_targeted_visual_review_complete';save(O/'comparison/summary.json',s)
out=dict(status='matched_local_translation_controls_completed_both_not_clarity_accepted',training=dict(updates_per_branch=3000,train_images=105,training_cameras=15,training_even_frames=list(range(720,733,2)),surface_active=6330,surface_frozen=246654,density_change=False,motion_parameters={'nodes':672,'points':132930}),validation={k:{'images':v['images'],'foreground_psnr_db':v['foreground_psnr_db']} for k,v in s['splits'].items()},detail_summary=s['detail_summary'],checkpoint_sha256={b:s['checkpoints'][b] for b in ['nodes','points']},motion_audit='motion-audit.json',visual_review='visual-review.json',temporal='temporal/summary.json',conclusion='Freeing local translation capacity gives small average changes without resolving fingers; neither variant adopted. No inference that all motion models or longer budgets are ineffective. Translation-only residual does not rotate covariance/local SH. No camera/clock, topology or additional anatomical supervision change.',adopt_as_development_baseline=False,retain_baseline='079/local_bspline',scope='Repeated development scene. Original holdouts are development validation, not an independent new-scene test. Static background joint reconstruction unaccepted.',goal_complete=False)
save(O/'final-summary.json',out)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);im=Image.new('RGB',(640,448),'white');d=ImageDraw.Draw(im)
for j,c in enumerate(['04','13']):
 p=O/'comparison'/f'camera{c}-frame000731-target_right_hand.png';a=np.array(Image.open(p));assert a.shape==(192,800,3)
 d.text((4,j*224+4),f'Camera{c}, withheld frame731; native pixels',font=font,fill='black')
 for col,k in enumerate([0,1,3,4]):im.paste(Image.fromarray(a[:,k*160:(k+1)*160]),(col*160,j*224+24))
dest=O/'comparison/native-motion-hand-comparison.png';im.save(dest);actual=np.array(Image.open(dest))
for j,c in enumerate(['04','13']):
 a=np.array(Image.open(O/'comparison'/f'camera{c}-frame000731-target_right_hand.png'))
 for col,k in enumerate([0,1,3,4]):assert np.array_equal(actual[j*224+24:j*224+216,col*160:(col+1)*160],a[:,k*160:(k+1)*160])
save(O/'figure-audit.json',dict(file=str(dest.relative_to(O)),sha256=sha(dest),native_panel_pixels_exact=True,no_enhancement=True,columns=['Original','Parent','Node motion','Point motion']))
shutil.copy2(__file__,O/Path(__file__).name);print('095complete; no new baseline; native figure saved.')
