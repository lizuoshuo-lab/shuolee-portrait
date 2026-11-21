from pathlib import Path
import json,sys,torch,hashlib,subprocess
from hamer.models import HAMER
from hamer.datasets.vitdet_dataset import ViTDetDataset
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-hand-prior-096';O.mkdir(exist_ok=True)
out=dict(status='official_network_and_preprocessing_imported_weights_and_MANO_not_loaded',python=sys.executable,torch=torch.__version__,CUDA_available=torch.cuda.is_available(),GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,repo_commit=subprocess.check_output(['git','-C',str(L/'repos/hamer'),'rev-parse','HEAD'],text=True).strip(),MANO_template_required='MANO_RIGHT.pkl',network_weights_required='hamer.ckpt;officialauthorarchive downloading',no_hand_predictions_generated=True,goal_complete=False)
(O/'environment-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
