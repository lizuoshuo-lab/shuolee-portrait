from pathlib import Path
import json,hashlib,copy
from datetime import datetime,timezone
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry/material-quarantine')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (H/'support122-deck-receipt.json').exists()
new=read(H/'quarantine122-deck-receipt.json');oldmeta=read(O/'ppt-update.json');oldbuild=read(H/'quarantine122-deck-build.json')
assert new['finalPath'].endswith('122背景材料核验.pptx') and new['source_sha256']==oldmeta['final_sha256']
assert sha(Path(new['source']))==oldmeta['final_sha256']
assert sha(Path(oldbuild['finalPath']))==oldmeta['final_sha256']
assert sha(Path(new['finalPath']))==new['final_sha256']
save(H/'support122-initial-misnamed-receipt.json',new);save(H/'support122-deck-receipt.json',new)
old=copy.deepcopy(new);old.update(oldmeta);old.update(source=oldbuild['source'],source_sha256=oldbuild['source_sha256'],finalPath=oldbuild['finalPath'],canonical_replaced=True)
old['renders'][51]['render_sha256']=sha(H/'quarantine122-final-52.png')
old['receipt_recovered_from']=dict(updated=datetime.now(timezone.utc).isoformat(),sources=['quarantine122-deck-build.json','material-quarantine/ppt-update.json','quarantine122-final-52.png'],reason='N11 builder cloned receipt filename and wrote new intermediate receipt over prior intermediate. Source and final PPTX files unchanged; recover earlier receipt from immutable build, acceptance andrender artifacts. All51unchanged renderhashes independently verified bynewbuilder.')
save(H/'quarantine122-deck-receipt.json',old)
save(H/'support122-receipt-repair.json',dict(repaired=True,canonical_PPTX_untouched=True,old_sha256=oldmeta['final_sha256'],new_sha256=new['final_sha256'],source_mistake='Receipt output filename accidentally retained from priorbuilder; fixed outputfilename without rerunning deckauthoring',reran_research=False))
print('Receipt routing repaired; canonical presentation was not changed')
