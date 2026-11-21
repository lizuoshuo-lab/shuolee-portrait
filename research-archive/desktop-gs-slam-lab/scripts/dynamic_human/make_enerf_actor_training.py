"""Create a separately versioned ablation trainer without editing067 source."""
from pathlib import Path
import shutil
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-actor-supervision-070'
text=(S/'train_enerf_graph_control.py').read_text(encoding='utf8')
def replace(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
replace('from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer,bind_graph','from enerf_graph_renderer import GraphMotion,bind_graph\nfrom enerf_actor_renderer import ActorSupervisedRenderer,ownership_loss')
replace("model=ResidualGraphRenderer(motion_file,graph)","model=ActorSupervisedRenderer(motion_file,graph)")
replace("root.mkdir(parents=True,exist_ok=True);assert not", "root.mkdir(parents=True,exist_ok=True);assert read(a.result_root/'mask-visual-review.json')['accepted_for_weak_training'];assert read(a.result_root/'actor-runtime-smoke.json')['passed'];assert not")
replace(" p=initial_parameters(OUT/'initial.npz');", " for o in obs:\n  if o['frame']==750:\n   label_file=a.result_root/'masks'/f\"camera{o['camera']}-frame000750.npz\";record=next(r for r in read(a.result_root/'mask-preparation.json')['records'] if r['camera']==o['camera']);assert sha(label_file)==record['labels_sha256'];labels=np.load(label_file)['labels'];source=np.load(OUT/'native-input'/f\"camera{o['camera']}-frame000750.npz\");x0,y0,x1,y1=source['bbox'];o['actor_labels']=torch.tensor(labels[y0:y1,x0:x1],device='cuda',dtype=torch.long)\n p=initial_parameters(OUT/'initial.npz');")
replace(" save(root/'graph-protocol.json',graph_protocol)"," graph_protocol['actor_supervision']=dict(weight=.05,frame=750,cameras=15,mask_sha256=sha(a.result_root/'mask-preparation.json'),loss='Class-balanced negative log normalized jointlysorted actor contribution on conservative SAM2trainingcores; unknown ignored; RGBmatte unchanged')\n save(root/'graph-protocol.json',graph_protocol)")
replace("  photometric=loss;regularization=", "  actor_loss=ownership_loss(info['actor_contributions'],alpha,o['actor_labels']) if 'actor_labels' in o else loss*0\n  photometric=loss;regularization=")
replace("loss=photometric+regularization", "loss=photometric+regularization+.05*actor_loss")
replace("photometric_loss=float(photometric),rigidity=", "photometric_loss=float(photometric),actor_loss=float(actor_loss),rigidity=")
replace("'train_enerf_graph_control.py','enerf_articulated_model.py'", "'train_enerf_actor_supervision.py','enerf_actor_renderer.py','enerf_articulated_model.py'")
replace("15000update_residual_graph_training_complete_native_evaluation_pending", "15000update_actor_supervision_ablation_complete_native_evaluation_pending")
# The graph protocol includes the additional fixed weak-supervision specification before resume checks.
text=text.replace(" if (root/'graph-protocol.json').exists():assert read(root/'graph-protocol.json')==graph_protocol\n", "")
text=text.replace("if o['frame']==750:","if o['frame']==750 and o['camera'] in read(a.result_root/'mask-visual-review.json')['accepted_cameras']:")
text=text.replace("cameras=15,mask_sha256=sha(a.result_root/'mask-preparation.json')","cameras=read(a.result_root/'mask-visual-review.json')['accepted_cameras'],mask_sha256=sha(a.result_root/'mask-preparation.json'),review_sha256=sha(a.result_root/'mask-visual-review.json')")
text=text.replace(" save(root/'graph-protocol.json',graph_protocol)"," if (root/'graph-protocol.json').exists():assert read(root/'graph-protocol.json')==graph_protocol\n save(root/'graph-protocol.json',graph_protocol)")
(S/'train_enerf_actor_supervision.py').write_text(text,encoding='utf8')
shutil.copy2(L/'outputs/dynamic-human-enerf-residual-graph-067/graph.npz',R/'graph.npz')
shutil.copy2(__file__,R/Path(__file__).name)
print('Separate070trainer and immutable067graph copy prepared')
