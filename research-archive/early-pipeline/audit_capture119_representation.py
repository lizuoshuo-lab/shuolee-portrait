"""CPU comparison of centers and coordinate-unit constraints, not training."""
import argparse
import numpy as np
from plyfile import PlyData
from capture119_clothed import ROOT,BASE,REC,PROJECT,rt
from shuolee_portrait.clothed_surface import center_statistics
p=argparse.ArgumentParser();p.add_argument('--subject',choices=['portrait','sofa'],default='portrait');a=p.parse_args()
root=ROOT/a.subject;old=BASE/a.subject;fit=dict(np.load(root/'fit/fitted.npz'))
rows=rt.read(root/'data/training-inputs.json')['records'];scale=float(fit['world_scale'])
assert rt.sha(root/'data/training-inputs.json')==rt.sha(old/'data/training-inputs.json')
assert np.array_equal(fit['vertices_world'],np.load(old/'fit/fitted.npz')['vertices_world'])
tri=fit['vertices_world'][0][fit['faces']].astype(float)
u=tri[:,1]-tri[:,0];s0=np.linalg.norm(u,axis=1);u/=s0[:,None]
v=tri[:,2]-tri[:,0];normal=np.cross(u,v);normal/=np.linalg.norm(normal,axis=1)[:,None]
tangent=-np.cross(normal,u);s1=np.abs((tangent*v).sum(1));size=(s0+s1)*.5/scale
stats={}
for name,path in [('old_mesh_bound1000',old/'gaussian-run/step-001000/world.ply'),
                  ('new_clothed1000',root/'gaussian-run/step-001000/world.ply')]:
    data=PlyData.read(path)['vertex'].data;xyz=np.column_stack([data[k] for k in ['x','y','z']])
    stats[name]=dict(sha256=rt.sha(path),points=len(xyz),**center_statistics(xyz,fit,rows))
output=REC/('representation-audit.json' if a.subject=='portrait' else 'sofa-representation-audit.json')
rt.save(output,dict(created=rt.now(),subject=a.subject,same_camera_hash=rt.sha(root/'data/training-inputs.json'),
    same_2D_observations=rt.sha(root/'data/observations.npz')==rt.sha(old/'data/observations.npz'),
    independently_recomputed_fits_bitwise_identical=True,
    fresh_fit_logs=[str(root/'logs/prior.log'),str(root/'logs/fit.log')],
    old_position='center=triangle_rotation*local_xyz*triangle_scale+triangle_center',
    old_position_penalty='0.01*relu(norm(local_xyz)-1), penalty starts at one triangle scale from face centroid, not hard clamping',
    triangle_scale_model_m_quantiles={str(q):float(np.quantile(size,q)) for q in [.1,.5,.9,.99]},
    old_fixed_triangle_quota=True,new_position_parameter_is_world=True,new_binding_is_none=True,new_body_attraction=False,
    comparison=stats,physical_metric_scale_independently_verified=False,
    causal_limits='Representation, initialization and coordinate-unit learning rates change together. This establishes a direct restriction and measured center distribution difference, not a unique cause of all ghosting.',
    source_hashes={str(PROJECT/'src/shuolee_portrait'/f):rt.sha(PROJECT/'src/shuolee_portrait'/f) for f in ['training.py','clothed_training.py','clothed_surface.py']}))
print({k:dict(away3cm=v['centers_over_3cm_from_bare_vertices'],coverage=[round(x['fraction_within_2px_of_center'],4) for x in v['projected_center_coverage']]) for k,v in stats.items()})
