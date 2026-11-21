from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(H/'indoor-semantic-all123-deck-receipt.json');source=Path(receipt['source']);final=Path(receipt['finalPath'])
prior=read(H/'field-background123-deck-receipt.json')
assert not receipt['canonical_replaced'] and sha(source)==receipt['source_sha256'] and sha(final)==receipt['final_sha256']
assert sha(Path(prior['finalPath']))==receipt['source_sha256'],'Prior canonical source must remain preserved'
assert receipt['changed_slides']==[52] and len(receipt['renders'])==52
assert read(H/'indoor-semantic-all123-exact-package.json')['other_members_byte_identical']
assert (H/'indoor-semantic-all123-final-validation.json').is_file()
# Last-page rendering actually opened and reviewed before running this helper.
mirror=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx')
shutil.copy2(final,source);shutil.copy2(final,mirror)
assert sha(source)==sha(mirror)==receipt['final_sha256']
receipt.update(canonical_replaced=True,canonical_path=str(source),mirror_path=str(mirror),actual_last_page_visual_review=True,updated=datetime.now(timezone.utc).isoformat())
(H/'indoor-semantic-all123-deck-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(canonical_replaced=True,changed_slides=[52],original_other51_unchanged=True,sha256=receipt['final_sha256']),ensure_ascii=False))
