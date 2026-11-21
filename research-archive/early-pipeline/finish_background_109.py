from pathlib import Path
from datetime import datetime
import hashlib, json, shutil

T = Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
B = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O = B / 'current/background-architecture-109'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, d):
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(p)

r = read(T / 'human109-deck-receipt.json')
source, final = Path(r['source']), Path(r['finalPath'])
assert not r['canonical_replaced']
assert sha(source) == r['source_sha256'] and sha(final) == r['final_sha256']
valid = read(T / 'human109-validation.json')
assert valid['packageIntegrity']['status'] == 'pass'
assert valid['presentationLayout']['exitCode'] == 0 and valid['firstPartyImport']['passed']
assert valid['finalSha256'] == r['final_sha256']
assert r['changed_slides'] == [52] and len(r['renders']) == 52
assert all(not x['changed'] for x in r['renders'][:51]) and r['renders'][51]['changed']
assert (O / 'published-report.png').is_file()
assert not read(O / 'evaluation.json')['numerical_gate_pass']

staged = source.with_suffix('.staged.pptx')
assert not staged.exists()
shutil.copy2(final, staged)
assert sha(staged) == r['final_sha256']
staged.replace(source)
r.update(canonical_replaced=True, canonical_path=str(source), canonical_sha256=sha(source))
save(T / 'human109-deck-receipt.json', r)
shutil.copy2(T / 'human109-final-52.png', O / 'slide52.png')
shutil.copy2(T / 'human109-validation.json', O / 'presentation-validation.json')
now = datetime.now().astimezone().isoformat()
review = dict(created=now, canonical=str(source), sha256=sha(source), slides=52,
              changed_slides=[52], first51_render_hashes_unchanged=True,
              changed_slide_visually_inspected=True, actual_powerpoint_application_opened=False)
save(O / 'presentation-review.json', review)
p = read(O / 'publication.json')
p.update(updated=now, ui_verified=True, screenshot=str(O / 'published-report.png'),
         presentation=str(source), presentation_sha256=sha(source), presentation_pending=False,
         viewer_actual_loaded_points=310961, viewer_actual_SH=3,
         viewer_sha_check_passed=True, viewer_latest_diagnostic='109')
save(O / 'publication.json', p)
state = read(B / 'work-status.json')
assert state['latest_diagnostic']['round'] == '109'
state['latest_report'] = dict(path=str(source), sha256=sha(source), changed_slides=[52],
                            slides=52, updated=now, remote_deployed=False)
state['updated'] = now
save(B / 'work-status.json', state)
save(O / 'handoff.json', dict(updated=now, status='background_camera_candidate_rejected',
     source_cameras_retained='074', actual_gaussian='098', latest_diagnostic='109',
     new_training=False, new_point_cloud=False, goal_complete=False,
     report=p['report'], presentation=str(source), presentation_sha256=sha(source),
     next_bounded_work=state['next_bounded_work'], no_new_training_started=True))
save(O / 'finalization-sources.json', {str(q): sha(q) for q in [
    T / 'publish_background_109.py', T / 'update_deck_109.mjs', Path(__file__)]})
duplicates = [T / 'human109-candidate.pptx', final]
save(O / 'cleanup-candidates.json', dict(created=now, executed=False,
     scope='Only duplicate PPT files created in round109; canonical verified and retained.',
     canonical=str(source), canonical_sha256=sha(source),
     records=[dict(path=str(q.resolve()), bytes=q.stat().st_size, sha256=sha(q)) for q in duplicates]))
print(json.dumps(dict(canonical=str(source), sha256=sha(source), report=p['report'],
                     changed_slides=[52], actual_gaussian='098', quality_passed=False), ensure_ascii=False))
