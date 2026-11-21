from pathlib import Path
from datetime import datetime
import json, hashlib
BASE=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
build=Path(__file__).resolve().parent
root=BASE/'current/smplx-convergence-fresh-018'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
receipt=read(build/'convergence018-deck-receipt.json')
cleanup=read(root/'cleanup.json')
assert receipt['canonical_replaced'] and cleanup['state']=='completed'
assert hashlib.sha256(Path(receipt['source']).read_bytes()).hexdigest()==receipt['final_sha256']
publication={'created':datetime.now().astimezone().isoformat(timespec='seconds'),
 'page':'http://127.0.0.1:8870/body-smplx.html#motion-convergence-018',
 'page_file':str(BASE.parent/'viewer/body-smplx.html'),
 'browser_screenshot':str(build/'convergence018-page-proof.png'),
 'canonical_presentation':receipt['source'],'presentation_sha256':receipt['final_sha256'],
 'changed_slides':[55],'preserved_slides':list(range(1,55)),
 'validation_receipt':str(build/'convergence018-final-validation.json'),
 'cleanup':{'manifest':str(root/'cleanup.json'),'files':cleanup['total_files'],'bytes':cleanup['total_bytes']},
 'verified_http_200':['body-smplx.html','body-fitting-method/motion-convergence/fixed-left90.jpg',
 'body-fitting-method/motion-convergence/source-000045.jpg','body-fitting-method/motion-convergence/source-000023.jpg',
 'body-fitting-method/motion-convergence/phase-conclusion.json','body-fitting-method/motion-convergence/knee-objective-audit.json'],
 'body_geometry_accepted':False,'new_gaussian_updates':0,'remote_deployed':False}
save(root/'publication.json',publication)
status=read(BASE/'work-status.json')
status['motion_convergence_closeout']['publication']=publication
status['updated']=publication['created']
save(BASE/'work-status.json',status)
for folder in ['smplx-floor-contact-fresh-016','smplx-motion-consistency-fresh-017']:
    file=BASE/'current'/folder/'run-state.json';state=read(file)
    state['superseded_arrays_cleanup']={'manifest':str(root/'cleanup.json'),
     'scope':'Only the exact files in this audit were deleted. Necessary017 short control, original observations and current018 models remain.'}
    save(file,state)
print(json.dumps({'published':True,'body_accepted':False,'cleanup_files':cleanup['total_files']}))
