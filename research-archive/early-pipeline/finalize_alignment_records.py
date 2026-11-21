from pathlib import Path
from datetime import datetime
import hashlib
import json

base=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
build=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
run=base/'current/gaussianavatars-smplx-confidence-fresh-003'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
cleanup=read(run/'cleanup-reviewed-models.json')
assert len(cleanup['files'])==3
assert all(f['deleted'] and not Path(f['path']).exists() for f in cleanup['files'])
assert sha(Path(cleanup['protected_candidate']))==cleanup['protected_sha256']
cleanup['bytes_removed']=sum(f['bytes'] for f in cleanup['files'])
cleanup['verified']=now
write(run/'cleanup-reviewed-models.json',cleanup)
review=read(run/'gaussian-run/root-review.json')
review['cleanup']={'receipt':'../cleanup-reviewed-models.json','gaussian_model_files_removed':3,'bytes_removed':cleanup['bytes_removed'],'current_candidate_unchanged':True,'compact_fitted_geometry_diagnostic_retained':True}
write(run/'gaussian-run/root-review.json',review)
publication=read(build/'alignment-publication.json')
publication.update({'verified':now,'browser_verified':True,'web_sha256':sha(Path(publication['web_page'])),'screenshot':str(build/'alignment-browser-proof.png'),'screenshot_sha256':sha(build/'alignment-browser-proof.png'),'images_loaded':[{'name':'alignment-method-summary.png','width':2080,'height':820},{'name':'matched-hand-crops.jpg','width':1680,'height':930},{'name':'fixed-pose-comparison.jpg','width':2080,'height':2370}],'video_metadata':{'readyState':4,'duration_seconds':6},'viewer_link':'http://127.0.0.1:8870/?model=whole-body-smplx-aligned','cleanup_receipt':str(run/'cleanup-reviewed-models.json')})
assert sha(Path(publication['canonical_report']))==publication['report_sha256']
write(build/'alignment-publication.json',publication)
write(build/'alignment-visual-review.json',{'created':now,'slide_count':55,'changed_slide':55,'first54_pixel_identical':True,'slide55_read_and_visually_reviewed':True,'images_complete_and_proportions_preserved':True,'no_visible_text_clipping':True,'native_powerpoint_verified':False,'no_quality_acceptance':True,'evidence':str(run/'gaussian-run/root-review.json')})
state=read(base/'work-status.json')
state['updated']=now
state['smplx_alignment_review'].update({'browser_verified':True,'publication_receipt':str(build/'alignment-publication.json'),'cleanup':{'files':3,'bytes_removed':cleanup['bytes_removed'],'retained_candidate_unchanged':True}})
state['latest_report']['browser_evidence_verified']=True
state['latest_report']['publication_receipt']=str(build/'alignment-publication.json')
state['current_experiment']={'name':'SMPLX_fixed_pose_alignment','status':'shared_pose_candidate_retained_confidence_and_hip_trials_not_promoted','accepted':False,'current_candidate':'002','candidate_steps':18000,'candidate_points':191358,'latest_gaussian_trial':'003','latest_gaussian_trial_steps':18000,'latest_fitting_trial':'004','latest_fitting_steps':2400,'latest_fitting_gaussian_steps':0,'current_training_running':False,'remote_deployed':False,'eyewear_specialization':False,'all_new_experiments_from_scratch':True}
write(base/'work-status.json',state)
print(json.dumps({'publication_verified':True,'report_sha256':publication['report_sha256'],'cleanup_files':3,'cleanup_MiB':round(cleanup['bytes_removed']/1024**2,2),'goal_status':state['long_term_goal_status'],'current_phase':state['current_phase'],'next_state':state['next_local_experiment']['state']},ensure_ascii=False))
