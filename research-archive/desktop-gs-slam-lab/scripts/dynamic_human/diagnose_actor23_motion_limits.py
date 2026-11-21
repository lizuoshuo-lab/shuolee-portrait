"""Read-only CPU parameter diagnostic, no model changes or heldout fit."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from actor23_graph_model import ActorGraph
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-actor23-independent-106';D=O/'motion-diagnostic';D.mkdir(exist_ok=True);torch.set_num_threads(3)
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();graph=np.load(O/'model/graph.npz');initial=np.load(O/'model/initial.npz');cal=read(O/'cameras.json')['04'];W=np.array(cal['w2c']);K=np.array(cal['K']);nodes=graph['node_points'];components=graph['point_component'][graph['source_point_indices']]
def project(x):
    q=(x@W[:3,:3].T+W[:3,3])@K.T;return q[:,:2]/q[:,2:]
uv=project(initial['points']);largest=np.argsort(-np.bincount(initial['component']))[:2];standing=max(largest,key=lambda c:np.ptp(uv[initial['component']==c,1]));y0,y1=np.min(uv[initial['component']==standing,1]),np.max(uv[initial['component']==standing,1]);group=(components==standing)&(project(nodes)[:,1]>y0+.6*(y1-y0));assert group.sum()>5
records={}
for b in ['dynamic_nominal','dynamic_clock']:
    p=O/b/'resume.pt';before=sha(p);cp=torch.load(p,map_location='cpu',weights_only=False);assert cp['step']==15000;g=ActorGraph(O/'model/graph.npz',cp['scene_scale'],device='cpu');g.load_state_dict(cp['graph']);g.requires_grad_(False);stats={}
    for name,value in g.named_parameters():
        t=value.tanh().numpy();stats[name]={}
        for label,mask in [('all_nodes',np.ones(len(nodes),bool)),('standing_lower_projection_nodes',group)]:
            v=abs(t[mask]);stats[name][label]=dict(nodes=int(mask.sum()),coefficients=int(v.size),fraction_abs_tanh_above_095=float((v>.95).mean()),max_abs_tanh=float(v.max()),median_tanh_derivative=float(np.median(1-v*v)))
    samples=[]
    for f in [600,601,604,612,620,623,624]:
        with torch.no_grad():R,d=g(torch.tensor((f-612)/12,dtype=torch.float32));angle=torch.acos(((R.diagonal(dim1=1,dim2=2).sum(1)-1)/2).clamp(-1,1)).numpy();tr=d.numpy();shift=np.linalg.norm(project(nodes+tr)-project(nodes),axis=1)
        samples.append(dict(common_index=f,lower_nodes_max_rotation_degrees=float(np.rad2deg(angle[group]).max()),lower_nodes_max_translation_scene_units=float(np.linalg.norm(tr[group],axis=1).max()),lower_nodes_max_projected_center_displacement_px=float(shift[group].max())))
    assert sha(p)==before;records[b]=dict(checkpoint_sha256=before,coefficient_statistics=stats,reference_projection_group_nodes=int(group.sum()),samples=samples)
    del cp,g
out=dict(status='read_only_parameter_saturation_and_control_node_motion_diagnostic',records=records,group_definition='Two largest train-only initial components; choose taller camera04referenceprojection,thenbottom40percent ofthatcomponentbyprojectedy. Heuristiccontext,notsemanticleglabels.',interpretation='Near-bound tanh coefficients lose gradient sensitivity; this is a diagnostic, not proof that bounds caused a particular artifact. Node-center displacement is not surface-point displacement, and rotations may also move the surface. No bound or parameter was changed.',goal_complete=False)
(D/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
