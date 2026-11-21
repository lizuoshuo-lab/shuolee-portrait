import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

BUILD = Path(__file__).resolve().parent
BASE = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
OUT = BASE / 'current/camera-local-inconsistency-audit-080'
VIEW = BASE.parent / 'viewer'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

now = datetime.now().astimezone().isoformat(timespec='seconds')
receipt = read(BUILD / 'camera080-deck-receipt.json')
candidate = Path(receipt['finalPath'])
canonical = Path(receipt['source'])
assert sha(candidate) == receipt['final_sha256']
assert sha(canonical) == receipt['source_sha256'], 'Canonical report changed since import'
validation = read(BUILD / 'camera080-validation.json')
assert validation['finalSha256'] == sha(candidate)
assert validation['packageIntegrity']['status'] == 'pass'
assert validation['presentationLayout']['finding_count'] == 0
assert receipt['changed_slides'] == [52]
assert len(receipt['renders']) == 52
review = dict(checked=now, presentation_sha256=sha(candidate), inspected_slides=[52],
              render_path=str(BUILD/'camera080-final-52.png'),
              render_sha256=sha(BUILD/'camera080-final-52.png'),
              visual_findings=[], text_clipping_observed=False,
              images_preserved=True, other_51_slides_render_hash_unchanged=True,
              native_powerpoint_verified=False, native_wps_verified=False)
save(BUILD/'camera080-deck-visual-review.json', review)
shutil.copyfile(candidate, canonical)
assert sha(canonical) == receipt['final_sha256']
receipt['canonical_replaced'] = True
receipt['canonical_replaced_at'] = now
save(BUILD/'camera080-deck-receipt.json', receipt)

manifest_path = BASE/'reports/report-manifest.json'
manifest = read(manifest_path)
report = next(r for r in manifest['reports'] if r['key'] == 'reconstruction-optimization')
report.update(sha256=sha(canonical), bytes=canonical.stat().st_size, slides=52,
              local_validation=str(BUILD/'camera080-validation.json'),
              local_visual_review=str(BUILD/'camera080-deck-visual-review.json'),
              latest_scope='080整图外参、局部观测一致性与跟踪漂移的独立诊断',
              current_local_report_deployed_to_00=False)
manifest['updated'] = now
manifest['last_local_revision'] = dict(path=str(canonical), sha256=sha(canonical),
    bytes=canonical.stat().st_size, slides=52, remote_delivered=False,
    transfer_performed=False, updated=now)
save(manifest_path, manifest)

browser = read(OUT/'browser-check.json')
assert len(browser['page']['images']) == 3
assert all(x['loaded'] for x in browser['page']['images'])
assert browser['model']['report'] == 'body-smplx.html#camera-residual'
assert browser['model']['model'].startswith('079 ')
assert '质量尚未通过验收' in browser['model']['status']
ply = VIEW/'models/whole-body-static-surface/shuolee_079_s18000_f0100.ply'
assert sha(ply) == '9463d848c81c688732b14e1d87199e37cefb5532bc82a580c8e227c26a14d6bd'
cameras = BASE/'current/full-scene-localization-074/cameras.json'
assert sha(cameras) == '5fe1cf0afff5dc856958873895fbb7d6c626b656557226dcd1909cb14ba2eefe'
latest = read(VIEW/'latest-result.json')
assert latest['round'] == '079' and latest['diagnosticRound'] == '080'
screens = [OUT/'page-verified.png', OUT/'page-result-verified.png']
ui = dict(verified=now, page='http://127.0.0.1:8870/body-smplx.html#camera-residual',
          actual_browser_check=browser, current_model_unchanged=True,
          current_model_sha256=sha(ply),
          screenshots=[dict(path=str(p), sha256=sha(p)) for p in screens],
          pc00_tested_this_round=False)
save(OUT/'final-ui-verification.json', ui)
publication = read(OUT/'publication.json')
publication.update(browser_verified=True, browser_verification=str(OUT/'final-ui-verification.json'),
                   ppt_updated=True, ppt_path=str(canonical), ppt_sha256=sha(canonical), verified=now)
save(OUT/'publication.json', publication)
completion = dict(completed=now, diagnostic_round='080', workflow_completed=True,
                  quality_accepted=False, root_cause_established=False,
                  physical_motion_proven=False, source_camera_updates=0,
                  gaussian_optimizer_updates=0, latest_gaussian_round='079',
                  latest_gaussian_sha256=sha(ply), source_camera_sha256=sha(cameras),
                  browser_verified=True, ppt_updated=True, pc00_deployed=False,
                  goal_complete=False, goal_turn_classification='progress',
                  consecutive_blocked_turns=0,
                  evidence=['protocol.json','result.json','readout-result.json',
                            'direct-tracking-check.json','confirmation.json','decision.json',
                            'final-ui-verification.json','publication.json'],
                  next='Fixed-camera static observation consistency check before any fresh training')
save(OUT/'completion.json', completion)
status_path = BASE/'work-status.json'
status = read(status_path)
assert status['latest_diagnostic']['round'] == '080'
status['updated'] = now
status['latest_diagnostic'].update(completion=str(OUT/'completion.json'), browser_verified=True, ppt_updated=True)
status['local_training_running'] = False
save(status_path, status)
print(json.dumps(dict(completed=True, gaussian_round='079', diagnostic_round='080',
                      ppt_sha256=sha(canonical), browser_verified=True, quality_accepted=False), ensure_ascii=False))
