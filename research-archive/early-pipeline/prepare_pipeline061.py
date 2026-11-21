from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
S=B/'current/person-clothing-pipeline-fresh-060';R=B/'current/person-clothing-motion-fresh-061'
assert not R.exists(),'A new experiment needs a new directory'
R.mkdir();records=[]
for relative in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
    source=S/relative;target=R/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    records.append(dict(file=relative,source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        role='Immutable source-image observation or official pretrained prediction; not a personalized fit.'))
(R/'source-inputs.json').write_text(json.dumps(dict(created=datetime.datetime.now().astimezone().isoformat(),
    inputs=records,old_personalized_fit_copied=False,old_gaussian_copied=False,old_optimizer_copied=False,
    source_images_and_masks_remain_read_only=True,protected_dependency=str(S/'data'),
    design='Fresh all175-frame body fit; fresh clothing seeds; new observed-motion support; fresh20w Gaussians and18000 updates.'),ensure_ascii=False,indent=2),encoding='utf-8')
repo=B.parents[2]/'repos/GaussianAvatars_SMPLX-shuolee'
# Freeze the code that actually produced 060 before adding optional 061 code.
archive=S/'source';archive.mkdir(exist_ok=True)
for name in ['train_clothing_cloud.py','initialize_clothing_cloud.py','scene/clothing_gaussian_model.py']:
    target=archive/name;target.parent.mkdir(exist_ok=True,parents=True)
    if not target.exists():shutil.copy2(repo/name,target)
print(R)
