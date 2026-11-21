from pathlib import Path
import json,argparse,numpy as np,torch
from enerf_graph_renderer import GraphMotion
from enerf_actor_renderer import ActorSupervisedRenderer,ownership_loss
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-actor-supervision-070';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
ap=argparse.ArgumentParser();ap.add_argument('--candidate',action='store_true');a=ap.parse_args();root=R if a.candidate else L/'outputs/dynamic-human-enerf-residual-graph-067';cp=torch.load(root/'residual_graph/resume.pt',map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};graph=GraphMotion(root/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);model=ActorSupervisedRenderer(L/'outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz',graph);torch.set_num_threads(8);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda');rows=[]
cams=json.loads((R/'mask-visual-review.json').read_text())['accepted_cameras']
with torch.no_grad():
 for c in cams:
  z=np.load(I/'native-input'/f'camera{c}-frame000750.npz');labels=torch.tensor(np.load(R/'masks'/f'camera{c}-frame000750.npz')['labels'],device='cuda',dtype=torch.long);rgb,alpha,info=model.render(p,c,750,tensor(z['w2c']),tensor(z['K']),1920,1080,2);w=info['actor_contributions'];prob=(w[0]+1e-6)/(alpha[0]+2e-6);known=labels>=0
  assigned=prob[known].gather(1,labels[known][:,None]).squeeze(1);rows.append(dict(camera=c,known_pixels=int(known.sum()),loss=float(ownership_loss(w,alpha,labels)),wrong_owner_pixels=int((assigned<.5).sum()),uncertain_owner_pixels=int((assigned<.9).sum()),mean_assigned_probability=float(assigned.mean())))
save=R/('candidate-training-ownership.json' if a.candidate else 'baseline-training-ownership.json');save.write_text(json.dumps(dict(status='13training_frame750_actor_contributions_measured',records=rows,goal_complete=False),indent=2)+'\n');print(json.dumps(rows))
