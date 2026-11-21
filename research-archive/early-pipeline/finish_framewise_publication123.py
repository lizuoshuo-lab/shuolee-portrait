from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
from urllib.parse import unquote

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
REF = T / 'shuolee/current/local-clothed-surface-123/smplifyx-reference-control'
OUT = REF / 'framewise-canonical-route'
SYNC = Path('D:/00/工作/查看器')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

stamp = datetime.now(timezone.utc).isoformat()
receipt_path = H / 'framewise-route123-deck-receipt.json'
receipt = read(receipt_path)
assert not receipt['canonical_replaced']
assert receipt['changed_slides'] == [52]
assert len(receipt['renders']) == 52
assert all(not item['changed'] for item in receipt['renders'][:51])
assert read(H / 'framewise-route123-exact-package.json')['other_members_byte_identical']
assert sha(receipt['source']) == receipt['source_sha256']
assert sha(receipt['finalPath']) == receipt['final_sha256']
# The rendered slide52 was inspected in the tool result before this publication.
shutil.copy2(receipt['finalPath'], receipt['source'])
mirror = T / 'shuolee/reports' / Path(receipt['source']).name
shutil.copy2(receipt['finalPath'], mirror)
assert sha(mirror) == sha(receipt['source']) == receipt['final_sha256']
receipt.update(canonical_replaced=True, updated=stamp,
               actual_before_after_slide52_reviewed=True,
               figure_aspect_ratio_preserved=True)
save(receipt_path, receipt)

ui = dict(updated=stamp, tab_id='32',
          url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#framewise-canonical',
          actual_existing_tab_reloaded=True, actual_section_DOM_text_verified=True,
          new_route_and69_images_notice_verified=True, image_elements_present=9,
          image_elements_loaded_in_browser=0, browser_image_visual_acceptance=False,
          source_composite_images_actually_reviewed=9, screenshot_verified=False,
          screenshot_limitation='Current IAB screenshot returned Cannot take screenshot with 0 width. The9lazy image elements report naturalWidth0 and complete=false. No alternate capture path was used.',
          marked_deliverable=True, tab15_untouched=True, new_PLY=False,
          company00_remote_verified=False, WPS_cloud_sync_verified=False)
save(OUT / 'viewer-ui-review.json', ui)
publication = read(OUT / 'publication.json')
publication.update(updated=stamp, actual_viewer_UI_pending=True,
                   viewer_text_verified=True, viewer_images_visual_verified=False,
                   viewer_UI_limitation=ui['screenshot_limitation'],
                   viewer_ui_review=str(OUT / 'viewer-ui-review.json'),
                   deck_pending=False, deck=receipt['source'],
                   deck_sha256=receipt['final_sha256'], only_slide52_changed=True,
                   first51_byte_and_render_checks_passed=True,
                   source_composite_images_actually_reviewed=9)
save(OUT / 'publication.json', publication)
destinations = [R / 'video10-global-initializer',
                T / 'viewer/diagnostics/video10-global-initializer',
                SYNC / 'diagnostics/video10-global-initializer']
for destination in destinations:
    target = destination / 'smplifyx-reference-control/framewise-canonical-route'
    for name in ['viewer-ui-review.json', 'publication.json']:
        shutil.copy2(OUT / name, target / name)
    page = (destination / 'index.html').read_text(encoding='utf-8')
    section = re.search(r'<section id="framewise-canonical">.*?</section>', page, re.S).group()
    images = re.findall(r'<img[^>]+src="([^"]+)"', section)
    assert len(images) == 9
    assert all((destination / unquote(src)).is_file() for src in images)

coord_path = H.parent / 'shuolee-coordination/local-research-status.json'
coord = read(coord_path)
coord.update(updated_at=stamp, latest_publication=str(OUT / 'publication.json'),
             latest_viewer_ui_review=str(OUT / 'viewer-ui-review.json'),
             viewer_UI_image_acceptance=False, current_report=receipt['source'],
             current_report_sha256=receipt['final_sha256'])
save(coord_path, coord)
print(json.dumps(dict(published=True, record_images=69, deck=receipt['source'],
                      only_slide52_changed=True, first51_preserved=True,
                      viewer_text_verified=True, viewer_images_UI_unverified=True,
                      gaussian_steps=0, goal_active=True), ensure_ascii=False))
