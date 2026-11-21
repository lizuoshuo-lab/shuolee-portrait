"""Fresh static-world diagnostic; raw observations only, never old learned state."""
from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/person-clothing-pipeline-fresh-060';R=B/'current/person-static-world-fresh-062'
assert not R.exists(),'Do not overwrite a round'
R.mkdir();sources=[]
for rel in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
    src=S/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    sources.append(dict(file=rel,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source=str(src),role='Raw observation / official pretrained output, not personalized fit'))
protocol=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),inputs=sources,
    old_subject_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
    model='shared_static_world',real_frames=175,initial_points=200000,steps=18000,
    body_refit_usage='Fresh shared observation fit only defines a bounded proposal box and units; no body deformation or vertex attraction in Gaussian training.',
    hypothesis='Does removing uncertain body/cloth time deformation improve the clothed surface and unseen views?',
    limitation='A diagnostic static assumption, not a validated dynamic-human pipeline.',
    protected_raw_input_dependency=str(S/'data'))
(R/'source-inputs.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(R)
