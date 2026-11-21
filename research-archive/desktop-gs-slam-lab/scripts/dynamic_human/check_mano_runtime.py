from pathlib import Path
import json,torch
from hamer.models.mano_wrapper import MANO
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-hand-prior-096')
m=MANO(model_path='/mnt/d/GS-SLAM-Data/models/mano_v1_2/numpy',is_rhand=True,use_pca=False,flat_hand_mean=True)
rot=torch.eye(3)[None,None];out=m(global_orient=rot,hand_pose=rot.expand(1,15,3,3),betas=torch.zeros(1,10),pose2rot=False)
assert out.vertices.shape==(1,778,3) and out.joints.shape==(1,21,3) and torch.isfinite(out.vertices).all()
(O/'mano-runtime-check.json').write_text(json.dumps(dict(status='official_HaMeR_MANO_layer_forward_passed',vertices=778,joints=21,hand='right',no_network_prediction_yet=True,goal_complete=False),indent=2)+'\n');print('HaMeR MANO wrapper:778vertices21joints valid')
