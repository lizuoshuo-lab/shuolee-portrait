from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(H/'video10-global-preflight123-deck-receipt.json');source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert not receipt['canonical_replaced'] and sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert read(H/'video10-global-preflight123-exact-package.json')['other_members_byte_identical']
assert (H/'video10-global-preflight123-final-validation.json').exists()
# Final slide rendered and actually inspected before this copy.
mirror=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx')
shutil.copy2(final,source);shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,canonical_path=str(source),mirror_path=str(mirror),actual_last_page_visual_review=True,updated=datetime.now(timezone.utc).isoformat())
(H/'video10-global-preflight123-deck-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['canonical_path','final_sha256','canonical_replaced','actual_last_page_visual_review']},ensure_ascii=False))
