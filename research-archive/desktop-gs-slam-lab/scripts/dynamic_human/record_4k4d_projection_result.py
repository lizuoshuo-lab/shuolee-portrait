from pathlib import Path
import json,hashlib,shutil,datetime
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-projection-control-040'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
rows=read(OUT/'paired-comparison/summary.json')['records'];assert len(rows)==6
assert read(OUT/'independent-state-audit.json')['matched_recorded_sampling']
assert all(r['psnr_delta_db']['foreground_matte_gt0_5']<0 for r in rows)
images=[f'paired-comparison/camera{c}-frame000750-full-person.png' for c in ['00','06','12','01','08','17']]
images+=[f'calibrated_projection/native-evaluation/camera{c}-frame000750-trained.png' for c in ['00','06','12']]
save(OUT/'visual-review.json',dict(status='empirical_projection_control_rejected_for_clarity',images=images,
    observations=['Hands, blue shirt motif, toy structure and crossing limbs remain blurred in both branches.',
    'Corrected branch retains or strengthens dotted shoulder and shoe edges in several views.',
    'Camera06 still blends the occluded toy/arm with the white shirt; no clear separation of fingers.',
    'Three complete calibrated held-out renders also inspected; no clarity acceptance.'],goal_complete=False))
summary=dict(status='matched_projection_control_completed_not_adopted',parent_updates=1000,new_updates_each=1000,history_updates_each=2000,
    heldout_foreground_psnr_delta_db={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in rows if r['split']=='heldout'},
    conclusion='All6 inspected views lose foreground PSNR and do not recover hand/clothing detail. Retain original projection for further diagnostics.',
    scope='Rejects this empirical static-background pixel-offset treatment from this parent and budget. Does not show that all physical camera calibration or time synchronization is ineffective.',
    calibration_access='All18camera separate static photographs700fit710/720check; no750heldoutpersonRGB used tofit.',
    recovery='Calibrated firstattempt failed before any optimizerupdate due CPUmetadata/GPUtensor device mismatch. Preserved failure and source; fixed per-alias device conversion and restarted from exact original parent. Mixed-device check passed.',goal_complete=False)
save(OUT/'final-summary.json',summary)
snap=OUT/'source-snapshot'
for n in ['retry_4k4d_projection_control.sh','check_projection_mixed_devices.py','record_4k4d_projection_result.py']:
    shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
save(OUT/'source-snapshot.json',dict(files=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(snap.iterdir()) if p.is_file()]))
manifest=LAB/'manifests/dynamic-human-work.json';m=read(manifest)
m.update(status='projection_pair_rejected_readonly_renderer_diagnostics_running',official4k4d_projection_control='outputs/dynamic-human-4k4d-projection-control-040/final-summary.json',
    current_runs=[dict(name='frozen_model_alpha_composition',path='outputs/dynamic-human-4k4d-alpha-composition-041',session_id=76920)],
    next_actions=['Complete041 actual alpha formula and native-image audit; inspect same-fragment before/after.',
                  'Run042 fixed-geometry source-mask support and premultiplication controls only after041GPU process ends.',
                  'Train only a supported candidate, then verify motion and joint scene continuity.'],
    pending=['Alpha and source blending diagnostic validation','Clear hand, clothing and occlusion reconstruction','Temporal and joint scene acceptance'],
    remaining_quality_issue='Input-mask repair only partially helps.040empirical static projection correction worsens all6view foreground metrics; current hands/clothes/occlusion remain blurred.',
    goal_turn_classification='Progress:040 paired1000newupdates completed and audited;12native renders48PNGs24panels verified,6paired panels+3fullimages reviewed; reject correction.041 mathematical difference confirmed, realmodel diagnostic active.',external_input_blocker=None)
m['completed'].append('040: exact039parent+1000newupdates each, recordedsampling matched;12nativeviews audited;9images viewed. Projection-offset candidate loses heldout foreground0.66to0.82dB; not adopted.')
m['updated']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['last_recorded_utc']=m['updated'];save(manifest,m)
print(json.dumps(summary))
