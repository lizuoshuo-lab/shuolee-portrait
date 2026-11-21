"""Read-only thin-triangle attribution and native pixel scale."""
import numpy as np
from run_official_gaussians123 import read,save,now,ASSETS
import train_smplify_eight123 as e


def main():
    cap=e.OUT.parent/'body-gaussian-smplify-eight-failure-capture'
    loc=read(cap/'gaussian-run/failure-location.json')
    with np.load(e.OUT/'fit/fitted.npz',allow_pickle=False) as z:f=dict(z)
    with np.load(e.OUT/'fit/surface-cache.npz',allow_pickle=False) as z:c=dict(z)
    with np.load(cap/'gaussian-run/step-000900/current-meshes.npz',allow_pickle=False) as z:m=dict(z)
    with np.load(ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:weights=z['weights']
    from smplx.joint_names import JOINT_NAMES
    records=[]
    for item in loc['triangles']:
        i=list(f['frame_ids']).index(item['frame']);vids=item['vertices'];K=f['K_integer_centers'][i]
        initial=c['base_vertices'][i,vids];current=m['vertices_camera'][i,vids]
        def project(v):
            p=v@K.T;return p[:,:2]/p[:,2:]
        uv0=project(initial);uv1=project(current)
        joint=int(weights[vids].mean(0).argmax())
        records.append(dict(**item,dominant_joint=joint,dominant_joint_name=JOINT_NAMES[joint],
            initial_altitude_mm=item['initial_area_m2']*2/max(item['initial_edges_m'])*1000,
            maximum_projected_vertex_move_px=float(np.linalg.norm(uv1-uv0,axis=-1).max()),
            projected_initial_triangle=uv0.tolist(),projected_current_triangle=uv1.tolist(),
            initial_projected_area_px2=float(abs(np.cross(uv0[1]-uv0[0],uv0[2]-uv0[0]))/2),
            current_projected_area_px2=float(abs(np.cross(uv1[1]-uv1[0],uv1[2]-uv1[0]))/2)))
    result=dict(updated=now(),triangles=records,interpretation='Both instabilities are in excluded poses. The Gaussian training sees8 frames but the shared clothing offset is transported to all59 poses. Normal reversal is a local orientation warning, not proof of visible large-scale body distortion or self-intersection.',
        next='Keep the8 training frames fixed. Any subsequent fix should act on the shared clothing mesh regularity across poses rather than silently adding the validation images into RGB training.',
        no_training_changes=True,quality_accepted=False)
    save(e.OUT/'failure-geometry-analysis.json',result)
    print(records)


if __name__=='__main__':main()
