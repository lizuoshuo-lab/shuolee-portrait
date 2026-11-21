"""Keep shared goal state tied to the actual current indoor clock process."""
import datetime
import os
import subprocess
import audit_gaussianavatars_params123 as a

ROOT=a.REC/'cross-subject-sh0-124';OUT=ROOT/'indoor-zoom-clock18k124'

def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');a.save(temp,data);os.replace(a.lp(temp),a.lp(path))

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    launch=a.read(OUT/'launch.json');p=a.read(OUT/'progress.json')
    query=subprocess.run(['wsl','-d','Ubuntu-20.04','--','ps','-p',str(launch['pid']),'-o','args='],capture_output=True)
    command=query.stdout.decode('utf-8',errors='replace').strip()
    alive='train_indoor_zoom_clock124.py train' in command
    completed=p['state']=='bounded_control_completed'
    ui_path=OUT/'delivery/step-018000/viewer-ui-review.json'
    delivered=completed and a.lp(ui_path).exists() and a.read(ui_path).get('actual_ui_verified',False)
    audit=a.read(ROOT/'indoor-surface-support-audit124/result.json')
    review=a.read(ROOT/'indoor-surface-support-audit124/visual-review.json')
    assert review['actual_visual_review']
    evidence=[]
    for r in audit['results']:
        den=sum(f['interior_known_pixels'] for f in r['frames'])
        evidence.append(dict(name=r['name'],points=r['points'],
            densify_events=r['original_clock_densify_events_before_saved_step'],
            clothing=r['regions']['clothing_triangles'],
            interior_alpha_below09=sum(f['alpha_under09'] for f in r['frames'])/den))
    stage='indoor_zoom_clock18k_completed_review_pending' if completed else ('indoor_zoom_clock18k_live' if alive else 'indoor_zoom_clock18k_stopped_inspection_required')
    if delivered:stage='indoor_zoom_clock18k_delivered_quality_open'
    entry=dict(updated=now,root=str(OUT),plan=str(OUT/'plan.json'),progress=p,
        actual_process=dict(pid=launch['pid'],command=command,alive=alive),
        generation='Same ROI2x source6; fresh20908 points and empty optimizer; six clock fields scaled600k-to18k.',
        recovery=a.read(OUT/'recovery-receipt.json'),density_audit=evidence,
        source_mask_pose_loss_unchanged=True,actual_1000_review=a.lp(OUT/'gaussian-run/step-001000/visual-review.json').exists(),
        quality_accepted=False,end_to_end_15min_accepted=False,target200k_accepted=False)
    if delivered:
        entry['delivery']=a.read(OUT/'delivery/publication.json')
        entry['actual_final_review']=a.read(OUT/'delivery/step-018000/visual-review.json')
    atomic(OUT/'state-summary.json',entry)
    next_action=('Review all6 final original-pixel comparisons and36 novel directions, audit density/geometry/native replay, then paired export with honest quality label; do not restart.' if completed else
        'Follow this exact PID and registered18k budget; actual all6 review at1000 before release, then inspect10000 and18000. Do not change running code, source hashes or optimizer; hidden indoor limbs remain unknown. No pose-repair loop or point-threshold sweep.')
    if delivered:next_action='Completed indoor ROI2x and scaled-clock candidate is in viewer with paired camera; no restart. Use frozen same-camera/opacity/novel-view evidence to isolate remaining fragmentation before another bounded change. Preserve unknown occluded limbs; overall quality and cold15min remain open.'
    for path in (a.R/'protocol.json',a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json'):
        d=a.read(path);d.update(updated=now,current_stage=stage,indoor_clock18k124=entry,
            training_running=alive,own_gpu_task_running=alive,current_worker_pid_wsl=launch['pid'] if alive else None,
            quality_accepted=False,goal_complete=False,goal_turn_classification='progress',goal_turn_blocked_streak=0,
            previous_goal_turn_audit=dict(classification='progress',blocked_streak=0,evidence='Authorized indoor person ROI2x1000 completed with all6 native-scale and36 camera-pair review;8percent color improvement, fragmentation remains.'),
            last_goal_turn_audit=dict(updated=now,classification='progress',blocked_streak=0,evidence='Read-only18 native RGB/alpha replays completed; clothing-point allocation and original3-event clock measured; new clock candidate running with preserved first bookkeeping interruption.'))
        if delivered:d['last_goal_turn_audit']['evidence']='Indoor ROI2x plus registered18k clock candidate completed; final6 source-pixel views, native readback,36 novel views and actual OpenFile Gaussian/points reviewed. Paired camera package delivered locally; quality remains open.'
        if 'updated_at' in d:d['updated_at']=now
        for k in ('worker_pid','active_pid','pid'):
            if k in d:d[k]=launch['pid'] if alive else None
        for k in ('running','gaussian_training_running','local_GPU_task_active','GPU_task_running'):
            if k in d:d[k]=alive
        if 'state' in d:d['state']=stage
        for k in ('next','next_action','nextAction','next_safe_action'):
            if k in d or k=='next_action':d[k]=next_action
        atomic(path,d)
    print(stage,launch['pid'] if alive else None,p['step'],flush=True)

if __name__=='__main__':main()
