from pathlib import Path
from datetime import datetime
import json
import hashlib

BASE = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
build = Path(__file__).resolve().parent
root = BASE / 'current/smplx-gvhmr-prior-fresh-019'
read = lambda p: json.loads(p.read_text('utf-8-sig'))
def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')

receipt = read(build / 'gvhmr019-deck-receipt.json')
cleanup = read(root / 'publication-cleanup.json')
assert receipt['canonical_replaced'] and cleanup['state'] == 'completed'
assert hashlib.sha256(Path(receipt['source']).read_bytes()).hexdigest() == receipt['final_sha256']
assert (build / 'gvhmr019-page-proof.png').is_file()
assert (build / 'gvhmr019-final-renders/55.png').is_file()
publication = {
    'created': datetime.now().astimezone().isoformat(timespec='seconds'),
    'page': 'http://127.0.0.1:8870/body-smplx.html#gvhmr-prior-019',
    'page_file': str(BASE.parent / 'viewer/body-smplx.html'),
    'browser_screenshot': str(build / 'gvhmr019-page-proof.png'),
    'canonical_presentation': receipt['source'],
    'presentation_sha256': receipt['final_sha256'],
    'changed_slides': [55],
    'preserved_slides': list(range(1, 55)),
    'validation_receipt': str(build / 'gvhmr019-final-validation.json'),
    'rendering_boundary': 'All55 slides exported with artifact-tool; first54 pixel-identical and final55 visually inspected. Native PowerPoint was not executed.',
    'cleanup': {'manifest': str(root / 'publication-cleanup.json'), 'files': cleanup['total_files'], 'bytes': cleanup['total_bytes']},
    'verified_http_200': ['body-smplx.html'] + ['body-fitting-method/gvhmr-prior/' + name for name in [
        'geometry-left90.jpg', 'geometry-front.jpg', 'source-000000.jpg', 'source-000023.jpg',
        'source-000045.jpg', 'source-000058.jpg', 'review-manifest.json', 'phase-conclusion.json']],
    'body_geometry_accepted': False,
    'new_personalized_fit_steps': 0,
    'new_gaussian_updates': 0,
    'remote_deployed': False,
}
save(root / 'publication.json', publication)
status = read(BASE / 'work-status.json')
status['gvhmr_prior_019']['publication'] = publication
status['gvhmr_prior_019']['visual_review_scope'] = read(root / 'phase-conclusion.json')['visual_reviewed']
status['gvhmr_prior_019']['full_temporal_review_completed'] = False
status['updated'] = publication['created']
save(BASE / 'work-status.json', status)
print(json.dumps({'published': True, 'canonical_ppt_sha256': receipt['final_sha256'], 'body_accepted': False, 'cleanup_files': cleanup['total_files']}))
