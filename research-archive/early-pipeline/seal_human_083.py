"""Seal observed local delivery state; the larger research goal remains active."""
import finalize_human_083 as p
import shutil
from datetime import datetime
now=datetime.now().astimezone().isoformat(timespec='seconds')
c=p.read(p.R/'completion.json');pub=p.read(p.R/'publication.json')
assert c['workflow_complete'] and not c['quality_accepted']
assert p.sha(p.Path(c['standard_ply']))==c['ply_sha256']
assert p.sha(p.Path(c['deck']))==c['deck_sha256']=='27d77b0b93fa13811b83a1c1c6e5c5edeb2295a6fbccc3029d44c667628040c2'
latest=p.read(p.V/'latest-result.json');assert latest['round']=='083' and latest['points']==300220 and not latest['qualityAccepted']
assert p.sha(p.V/'model-whole-body-static-human-init.json')==latest['revision']
assert (p.R/'review/page-final.png').is_file()
ui=p.read(p.R/'review/viewer-interaction-review.json');ui.update(follow_latest_restored=True,preset_loaded=True,preset_camera_raw_frame=70);p.save(p.R/'review/viewer-interaction-review.json',ui)
proof=dict(updated=now,viewer_url='http://127.0.0.1:8870/?model=latest',round_shown='083',points_shown=300220,
    follow_latest_checked=True,quality_warning_visible=True,page_url=c['report_url'],seven_section_images_loaded=True,
    download_HEAD_status=200,standalone_modes=['gaussian','point_cloud'],pc00_deployed=False,
    screenshots=[dict(path=str(p.R/'review'/n),sha256=p.sha(p.R/'review'/n)) for n in ['viewer-preset.png','viewer-standalone-gaussians.png','viewer-standalone-points.png','page-final.png']])
p.save(p.R/'review/final-ui-verification.json',proof)
c.update(final_ui_verified=True,final_ui_receipt=str(p.R/'review/final-ui-verification.json'));p.save(p.R/'completion.json',c)
pub.update(browser_verified=True,final_ui_receipt=c['final_ui_receipt']);p.save(p.R/'publication.json',pub)
s=p.read(p.B/'work-status.json');s['latest_diagnostic']=dict(round='083',type='frozen post-training evaluation',status='completed_reviewed',path=str(p.R/'review'),candidate_adopted=False,gaussian_updates=0,report=c['report_url'])
s['updated']=now;s['next_bounded_work'].update(do_not_repeat='082 fitting and083 training are complete. Do not rerun either or scan initialization regions/weights.',previous_round_complete=True);p.save(p.B/'work-status.json',s)
for n in ['record_human_review_083.py','update_deck_083_result.mjs','commit_human_deck_083.py','seal_human_083.py']:shutil.copy2(p.W/n,p.R/'source-final'/n)
lab=p.B.parents[2]
for repo,names in [('GaussianAvatars_SMPLX-shuolee',['review_human_person_083.py']),('PGSR-shuolee',['review_human_transfer_083.py','review_human_depth_083.py'])]:
    for n in names:shutil.copy2(lab/'repos'/repo/n,p.R/'source-final'/n)
print('083 standard PLY, local UI, page, canonical PPT and bounded next diagnostic sealed; quality not accepted.')
