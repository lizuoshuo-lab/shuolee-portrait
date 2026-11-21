"""Bounded CPU geometry counts for unchanged official near/AA policies."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
import evaluate_official_hierarchy as e
from read_hierarchy import read_hierarchy
OUTPUT=Path(__file__).resolve().parent
run=e.OUTROOT/'truck-official-continuous-heldout-002'
audit=e.read_json(run/'baseline-bridge/model-audit.json');h=read_hierarchy(audit['source_path'])
e.require(e.digest(h.path)==audit['source_sha256'],'Frozen model changed')
selection=run/'baseline-bridge/leaves/selections/fixed-author-leaves.npz'
leaves=np.load(selection)['indices'];xyz=np.asarray(h.positions[leaves],np.float32)
homogeneous=np.column_stack([xyz,np.ones(len(xyz),np.float32)])
geometry=np.load(e.TRACK/'training/truck-baseline-001/parser_geometry.npz')
camera_rows=e.read_json(run/'official-camera-audit.json')['per_image'];rows=[]
for row in camera_rows:
    index=row['test_index'];K=geometry['K'][index];camera=e.make_official_camera_CPU(K,geometry['camtoworlds'][index],245,136,row['name'])
    view=homogeneous@camera.world_view_transform.numpy();z=view[:,2]
    rejected=(z>.01)&(z<=.2)
    with np.errstate(divide='ignore',invalid='ignore'):
        x=view[:,0]/z*K[0,0]+K[0,2]-.5;y=view[:,1]/z*K[1,1]+K[1,2]-.5
    center_in_image=(x>=0)&(x<245)&(y>=0)&(y<136)
    rows.append(dict(name=row['name'],leaves=len(leaves),center_depth_gt01=int((z>.01).sum()),center_depth_gt02=int((z>.2).sum()),
        source_native_cutoff_extra_rejected_center_count=int(rejected.sum()),extra_rejected_center_inside_image_count=int((rejected&center_in_image).sum()),
        all_positive_center_inside_image_count=int(((z>.01)&center_in_image).sum()),center_depth_min=float(z.min()),center_depth_max=float(z.max()),
        boundary_near_02_within1e6=int((np.abs(z-.2)<1e-6).sum())))
native=e.ROOT/'repos/hierarchy-rasterizer-large-gs/cuda_rasterizer'
header=native/'auxiliary.h';forward=native/'forward.cu'
e.require('#define DGR_FIX_AA' in header.read_text() and 'p_view.z <= 0.2f' in header.read_text(),'Pinned source policy differs')
e.save(OUTPUT/'outcome.json',dict(status='passed_CPU_source_policy_and_all32_leaf_center_depth_diagnostics',cuda_called=False,
    source_model_sha256=audit['source_sha256'],leaf_order_npz_sha256=e.digest(selection),source_header_path=str(header),source_header_sha256=e.digest(header),
    source_forward_path=str(forward),source_forward_sha256=e.digest(forward),per_image=rows,
    anti_aliasing='Source DGR_FIX_AA enabled: projected covariance plus0.3 and opacity sqrt(max(.000025,det_raw/det_filtered)) compensation; gsplat baseline usedclassic. No author source modification.',
    native_near='in_frustum rejects center z<=.2 even with official Camera projection znear=.01; gsplatbaseline near=.01.',
    interpretation='Counts use exact saved-source CPUfloat32 view matrices and leaf XYZ; potential near-cut difference only, not pixel visibility/quality loss attribution. A Gaussian footprint can reach image with center outside; covariance, occlusion and alpha are not evaluated here. Boundary floating rounding may differ from native FMA.',
    max_extra_rejected_center_inside_image=max(x['extra_rejected_center_inside_image_count'] for x in rows)))
print('CPU source/center-depth audit32 complete; maxcenter_inside_near_difference',max(x['extra_rejected_center_inside_image_count'] for x in rows))
