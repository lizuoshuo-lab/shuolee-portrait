"""Maintain the single research record and prepare an exact-file retention list."""
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont
from shuolee_geometry_diagnosis import ROOT, OUT, write_json, sha


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    home=ROOT/'shuolee';run=home/'current/da3-refinement';now=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    summary=read(run/'summary.json');review=read(run/'review/summary.json')
    assert summary['status']=='complete_pending_visual_review'
    picture=Image.open(run/'review/elevation-15.png').convert('RGB')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    panel=Image.new('RGB',(picture.width,picture.height+60),'white')
    panel.paste(picture,(0,60));draw=ImageDraw.Draw(panel)
    draw.text((12,7),'510-step pilot: A = DA3 guidance; B = RGB-only control',fill='black',font=font)
    draw.text((12,32),'No clear improvement in this reviewed sample. Not promoted.',fill='black',font=font)
    panel.save(run/'review/key-comparison.png')
    decision=dict(id='shuolee-da3-geometry-20261019',created=now,status='not_promoted',
        question='Does improving the training depth prior improve limited-view human rendering?',
        intervention='DA3-BASE known-pose51view depth, dense consistency and edge gating, paired native fixed-topology510step refinement',
        result='Depth-vs-SfM agreement improved; bounded 3D refinement did not demonstrate a clear visual improvement.',
        prior_head_high_confidence_median_relative_error={'da2':.2318329511674653,'da3':.013128550215722568},
        delta_vs_rgb_control=review['delta_vs_rgb_control'],
        parameter_change=read(run/'parameter-change.json'),
        review=dict(reviewer='assistant',actual_images=['observed-000016','elevation-15','elevation-45','distance-45'],
            finding='Known-view appearance almost unchanged; damaged facial detail and background surfaces persist in large-elevation and close-view samples.',
            human_ratings=0,full_video_perceptual_validation=False),
        interpretation='Reject promotion of this short fixed-topology candidate; not a rejection of DA3 or of full retraining with reliable depth.',
        next='Compare surface geometry representation against the retained baseline; retain validated DA3 priors as an optional controlled source, without increasing depth weight blindly.',
        source_model_preserved=True,accepted_as_best=False,
        evidence='current/da3-refinement/review/key-comparison.png')
    write_json(run/'decision.json',decision)
    summary.update(status='complete_not_promoted',decision_file='decision.json',accepted_as_best=False)
    write_json(run/'summary.json',summary)
    write_json(run/'status.json',dict(status='complete_not_promoted',training_running=False,accepted_as_best=False,cleanup_pending=True))
    ledger=read(home/'results-ledger.json')
    ledger['decisions']=[d for d in ledger['decisions'] if d.get('id')!=decision['id']]+[decision]
    ledger['updated']=now
    write_json(home/'results-ledger.json',ledger)
    state=read(home/'work-status.json');state.update(updated=now,long_term_authorized=True,long_term_goal_status='active',
        local_training_started=True,local_training_running=False,server_training_started=False,
        current_phase='geometry diagnosis complete; first DA3 refinement evaluated and not promoted',
        current_decision=decision,
        next_bounded_experiment='Surface geometry pilot using PGSR or2DGS, same known cameras; choose after checking native CUDA compatibility and rendering semantics.',
        latest_local_execution=dict(da3_inference_frames=51,da3_inference_seconds=10.417470932006836,da3_peak_allocated_mib=3549.44482421875,
            training_steps_per_arm=510,arms=['real_only','da3_guided'],gaussians=316044,
            training_peak_mib={k:v['peak_allocated_mib'] for k,v in summary['arms'].items()},
            continuous_paths_rendered=2,frames_per_path=60,independent_test=False),
        research_alternatives=['DA3','VGGT','DNGaussian','2DGS','PGSR','DP-GS','CoR-GS','DropGaussian','DDGS','Mip-Splatting','ViewExtrapolator','Difix3D'])
    state['completed'] += [x for x in ['native camera and depth audit','DA3 full training-view inference and conservative prior validation',
        'paired native3D refinement reviewed without clear gain','open-source alternatives integrated into long-term plan'] if x not in state['completed']]
    write_json(home/'work-status.json',state)
    # Only this completed, rejected pilot. Source model, inputs and DA3 priors
    # are deliberately outside the exact file set.
    candidates=[]
    for arm in ['real_only','da3_guided']:
        candidates.extend(run/arm/name for name in ['model.ply','filter_3d.pt','optimizer.pt'])
    for arm in ['initial','real_only','da3_guided']:
        candidates.extend((run/arm/'evaluation').glob('*.png'))
    candidates.extend(p for p in (run/'review').iterdir() if p.suffix in ['.png','.mp4'] and p.name!='key-comparison.png')
    rows=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in candidates]
    assert all(Path(r['path']).is_relative_to(run.resolve()) for r in rows)
    assert len({r['path'] for r in rows})==len(rows)
    cleanup=dict(status='prepared',created=now,root=str(run.resolve()),reason='Completed pilot reviewed; no clear gain, per user retention instruction',
        files=rows,total_bytes=sum(r['bytes'] for r in rows),preserve=[str(run/'review/key-comparison.png'),str(home/'long-term-plan.md')],
        protected_source=str(ROOT/'native-resolution-cache-full-18000-s0/model/point_cloud/iteration_18000/point_cloud.ply'),
        protected_source_sha256=summary['protocol']['source_sha256'])
    write_json(run/'cleanup.json',cleanup)
    print(json.dumps(dict(decision='not_promoted',cleanup_files=len(rows),cleanup_mib=cleanup['total_bytes']/2**20),indent=2))


if __name__=='__main__':main()
