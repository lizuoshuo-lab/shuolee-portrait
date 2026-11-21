from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

T = Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
LAB = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B = LAB / 'outputs/mini-splatting2-mip-depth-001/shuolee'
O = B / 'current/material-support-gradient-110'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)

r = read(T / 'human110-deck-receipt.json')
source, final = Path(r['source']), Path(r['finalPath'])
assert not r['canonical_replaced']
assert sha(source) == r['source_sha256']
assert sha(final) == r['final_sha256']
v = read(T / 'human110-validation.json')
assert v['packageIntegrity']['status'] == 'pass'
assert v['presentationLayout']['exitCode'] == 0
assert v['firstPartyImport']['passed'] and v['finalSha256'] == r['final_sha256']
assert r['changed_slides'] == [52] and len(r['renders']) == 52
assert all(not x['changed'] for x in r['renders'][:51]) and r['renders'][51]['changed']
assert (O / 'published-report.png').is_file()
assert read(O / 'geometry-only-check.json')['check_passed']
state = read(B / 'work-status.json')
assert state['latest_diagnostic']['round'] == '110'

staged = source.with_suffix('.staged.pptx')
assert not staged.exists()
shutil.copy2(final, staged)
assert sha(staged) == r['final_sha256']
staged.replace(source)
r.update(canonical_replaced=True, canonical_path=str(source), canonical_sha256=sha(source))
save(T / 'human110-deck-receipt.json', r)
shutil.copy2(T / 'human110-final-52.png', O / 'slide52.png')
shutil.copy2(T / 'human110-validation.json', O / 'presentation-validation.json')
now = datetime.now().astimezone().isoformat()
save(O / 'presentation-review.json', dict(
    created=now, canonical=str(source), sha256=sha(source), slides=52,
    changed_slides=[52], first51_render_hashes_unchanged=True,
    changed_slide_visually_inspected=True, actual_powerpoint_application_opened=False))
p = read(O / 'publication.json')
p.update(updated=now, ui_verified=True, screenshot=str(O / 'published-report.png'),
         presentation=str(source), presentation_sha256=sha(source), presentation_pending=False,
         viewer_actual_loaded_points=310961, viewer_actual_SH=3,
         viewer_sha_check_passed=True, viewer_latest_diagnostic='110')
save(O / 'publication.json', p)
state['latest_report'] = dict(path=str(source), sha256=sha(source), changed_slides=[52],
                            slides=52, updated=now, remote_deployed=False)
state['updated'] = now
save(B / 'work-status.json', state)
h = read(O / 'handoff.json')
h.update(updated=now, report=p['report'], presentation=str(source),
         presentation_sha256=sha(source), publication_complete=True,
         independent_followup_review='in_progress')
save(O / 'handoff.json', h)
files = [T / 'review_material_110.py', T / 'audit_material_regions_110.py',
         T / 'publish_material_110.py', T / 'update_deck_110.mjs', Path(__file__)]
files += [LAB / 'repos/PGSR-shuolee' / name for name in [
    'audit_material_support_gradients_110.py', 'inspect_material_moment_parity_110.py',
    'material_geometry_support.py', 'check_geometry_only_support_110.py']]
save(O / 'finalization-sources.json', dict(created=now,
     files=[dict(path=str(q), sha256=sha(q)) for q in files]))
print(json.dumps(dict(canonical=str(source), sha256=sha(source), report=p['report'],
                     changed_slides=[52], actual_gaussian='098', quality_accepted=False),
                 ensure_ascii=False))
