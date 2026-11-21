"""Read-only 022 sensitivity of the original shared SMPL-X beta basis."""
import ast
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from official_smplifyx_entry123 import sha
from prepare_leafwoman_ear139 import OUT as EAR
from probe_leafwoman_ear_pose139 import jacobian
from replay_leafwoman_arm138 import ARCHIVE, ASSET, FIT, OFFICIAL, SMPLX_ROOT, atomic, read


OUT = EAR.parent/'shared-shape140'


def posed_affine(model, archive, i):
    """Linear SMPL-X skinning at fixed joint rotations, including moving pivots."""
    tpl, dirs, expr_dirs, posedirs, reg, weights, parents, hands = model
    hand_l, hand_r, mean_l, mean_r = hands
    L = archive['left_hand_pose'][i]@hand_l + mean_l
    R = archive['right_hand_pose'][i]@hand_r + mean_r
    pose = np.concatenate([archive['global_orient'][i], archive['body_pose_axis_angle'][0],
        *(archive[k][i] for k in ('jaw_pose','leye_pose','reye_pose')), L, R]).reshape(55,3)
    local = Rotation.from_rotvec(pose).as_matrix()
    feature = (local[1:]-np.eye(3)).reshape(-1)
    shape = tpl + dirs@archive['betas'][0] + expr_dirs@archive['expression'][i]
    # Last axis 0 carries the actual saved beta geometry, 1:11 are d/dbeta.
    shaped = np.concatenate([shape[...,None], dirs],axis=2)
    joints = (reg@shaped.reshape(len(tpl),-1)).reshape(55,3,11)
    posed = shaped.copy()
    posed[...,0] += (posedirs@feature).reshape(-1,3)
    global_R = np.empty_like(local)
    centers = np.empty_like(joints)
    global_R[0], centers[0] = local[0], joints[0]
    for j,p in enumerate(parents[1:],1):
        global_R[j] = global_R[p]@local[j]
        centers[j] = centers[p]+global_R[p]@(joints[j]-joints[p])
    trans = centers-global_R@joints
    v = (weights@global_R.reshape(55,9)).reshape(-1,3,3)@posed
    v += (weights@trans.reshape(55,33)).reshape(-1,3,11)
    t = archive['camera_translation'][i]
    v[...,0] += t
    centers[...,0] += t
    return v, centers


def pinned_function(path, container, name, scope):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = tree.body if container is None else next(n for n in tree.body if getattr(n,'name',None)==container).body
    fn = next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


def affine_inputs():
    path = OFFICIAL/'smplifyx/utils.py'
    mapper = pinned_function(path,None,'smpl_to_openpose',{'np':np})(use_face_contour=False)
    package = SMPLX_ROOT/'smplx'
    vdef = ast.parse((package/'vertex_ids.py').read_text(encoding='utf-8'))
    ids = ast.literal_eval(next(n.value for n in vdef.body if isinstance(n,ast.Assign)))['smplx']
    selector = package/'vertex_joint_selector.py'
    # Read order from the original selector, retaining data and avoiding Torch.
    fn = pinned_function(selector,'VertexJointSelector','__init__',{
        'np':np, 'super':lambda *_:type('NoParent',(),{})(),
        'torch':type('TorchType',(),{'long':np.int64}), 'to_tensor':lambda a,dtype: a.astype(dtype)})
    class Collector:
        def register_buffer(self,name,value):
            setattr(self,name,value)
    obj = Collector()
    fn.__globals__['VertexJointSelector'] = Collector
    fn(obj,vertex_ids=ids)
    return mapper, obj.extra_joints_idxs


def project(xyz,K):
    q = xyz@K.T
    return q[:,:2]/q[:,2:]


def optimization(samples, with_rigid):
    n = 10+6*len(samples)*int(with_rigid)+1
    rows,bounds = [],[]
    for i,sample in enumerate(samples):
        uv,betas,rigid,polygon_lows,polygon_highs = sample
        design = np.zeros((52,2,n))
        design[:,:,:10] = betas
        if with_rigid:
            design[:,:,10+6*i:16+6*i] = rigid
        face = design[:51].reshape(-1,n)
        rows.extend([face,-face]); bounds.extend([np.ones(len(face))]*2)
        # Admit the outer enclosing rectangle (more permissive than a polygon).
        for low,high in zip(polygon_lows,polygon_highs,strict=True):
            block = np.vstack([design[-1],-design[-1]])
            block[:,-1] = -1
            rows.append(block);bounds.append(np.r_[high-uv[-1],uv[-1]-low])
    cost = np.zeros(n);cost[-1] = 1
    A,b = np.vstack(rows),np.concatenate(bounds)
    fit = linprog(cost,A_ub=A,b_ub=b,bounds=[(None,None)]*(n-1)+[(0,None)],method='highs')
    assert fit.success and np.max(A@fit.x-b) < 1e-6
    return fit.x


def main():
    result_path = OUT/'result.json'
    assert not result_path.exists()
    parent_plan = read(EAR/'plan.json')
    assert parent_plan['archive_sha256']==sha(ARCHIVE)
    checked = read(FIT/'plan.json')['source_hashes']
    paths = [ASSET, OFFICIAL/'smplifyx/utils.py',*(SMPLX_ROOT/'smplx'/p for p in ('body_models.py','lbs.py'))]
    pins = {str(p):sha(p) for p in paths}
    for path,digest in pins.items():
        short = '/'.join(Path(path).parts[-2:]) if path.endswith('.py') else Path(path).name
        assert {v for k,v in checked.items() if k.endswith(short)}=={digest}
    annotation_paths = [EAR/('source-only-annotation-'+name+'.json') for name in 'ab']
    marks = [read(p) for p in annotation_paths]
    for m in marks: assert m['plan_sha256'] == sha(EAR/'plan.json')
    plan_path = OUT/'plan.json'
    plan = dict(archive_sha256=sha(ARCHIVE),parent_plan_sha256=sha(EAR/'plan.json'),pins=pins,
        annotations=[sha(p) for p in annotation_paths], runner_sha256=sha(__file__),
        variant='Only add shared beta10 to fixed-geometry local head transform control. Also inspect beta-only.',
        guard='For each of six views, protect all51 old inner-face projections at ≤1px per coordinate. '
              'Fit both source ear region bounding rectangles padded by the declared read uncertainty. '
              'Minimal worst remaining coordinate excess is a relaxed first-order bound, not a true image-space guarantee.',
        side_effect='Measure all24 existing body projections with saved root/pose/translation and shape-only effect. '
                    'No betas, mesh or camera published as a candidate.')
    if plan_path.exists(): assert read(plan_path) == plan
    else: atomic(plan_path,plan)
    mapper, extras = affine_inputs()
    assert mapper.shape==(118,) and extras.shape==(21,) and extras[3:5].tolist()==[616,6]
    with np.load(ASSET,allow_pickle=False) as m, np.load(ARCHIVE,allow_pickle=False) as fit:
        archive = dict(fit)
        face, f = archive['faces'], m['f']
        assert np.array_equal(f,face) and len(fit['frame_ids'])==24
        landmarks = f[m['lmk_faces_idx']]
        bary = m['lmk_bary_coords'].astype(np.float64)
        sd = m['shapedirs']; assert sd.shape[2]>=400
        parent=m['kintree_table'][0].astype(int);parent[0]=-1
        model = (m['v_template'].astype(float),sd[:,:,:10].astype(float),sd[:,:,300:310].astype(float),
                 m['posedirs'].reshape(-1,486).astype(float),m['J_regressor'].astype(float),m['weights'].astype(float),
                 parent, (m['hands_componentsl'][:12],m['hands_componentsr'][:12],m['hands_meanl'],m['hands_meanr']))
    ids = archive['frame_ids'].tolist()
    K = archive['K_integer_centers']
    v_err,j_err,uv_err = 0.,0.,0.
    eval_set, samples, bodies = {}, [], []
    for i,name in enumerate(ids):
        v,J = posed_affine(model,archive,i)
        LM = np.einsum('lijc,li->ljc', v[landmarks],bary)
        full = np.concatenate([J,v[extras],LM],axis=0)[mapper]
        uv = project(full[...,0],K)
        v_err=max(v_err,float(np.max(np.abs(v[...,0]-archive['vertices_camera'][i]))))
        j_err=max(j_err,float(np.max(np.abs(full[...,0]-archive['joints_camera'][i]))))
        uv_err=max(uv_err,float(np.max(np.abs(uv-archive['joints_projected_native'][i]))))
        # Use body points that were active in the baseline and observed.
        support=(archive['observed_native_keypoints'][i,:25,2]>0);support[[1,9,12,17,18]]=False
        bodies.append((full[:25][support],name))
        if name in [r['frame'] for r in parent_plan['frames']]:
            eval_set[name] = full
    assert v_err < 2e-5 and j_err < 2e-5 and uv_err < .01,(v_err,j_err,uv_err)
    verify_archive = dict(archive,betas=archive['betas'].astype(float)+.15*np.cos(np.arange(10)))
    i = ids.index(parent_plan['frames'][0]['frame'])
    old_v,_ = posed_affine(model,archive,i)
    check_v,_ = posed_affine(model,verify_archive,i)
    affine_error=float(np.max(np.abs(old_v[...,0]+old_v[...,1:]@(.15*np.cos(np.arange(10)))-check_v[...,0])))
    assert affine_error<1e-9
    lows,highs,xyz_sets = [],[],[]
    for i,item in enumerate(parent_plan['frames']):
        full = eval_set[item['frame']]
        j=18 if item['side']=='left' else 17
        v = full[[*range(67,118),j]]
        xyz = v[...,0]; center = xyz[:-1].mean(0)
        uv,rigid = jacobian(xyz,center,K)
        betas = rigid[...,3:]@v[...,1:]
        low,high = [],[]
        for labels in marks:
            row = labels['rows'][i]
            assert row['frame']==item['frame'] and row['side']==item['side']
            poly = np.asarray(row['polygon_xy'])+np.asarray(item['crop_xyxy'][:2])
            tol = row['uncertainty_px']; low.append(poly.min(0)-tol);high.append(poly.max(0)+tol)
        samples.append((uv,betas,rigid,low,high));xyz_sets.append(v);lows.append(low);highs.append(high)
    variants = {}
    for with_rigid in (False,True):
        x = optimization(samples,with_rigid)
        details = []
        for i,(xyz,(ref,_,_,low,high)) in enumerate(zip(xyz_sets,samples,strict=True)):
            candidate = xyz[...,0]+xyz[...,1:]@x[:10]
            angle = 0.
            if with_rigid:
                pose = x[10+6*i:16+6*i]
                center = xyz[:-1,:,0].mean(0)
                angle=float(np.rad2deg(np.linalg.norm(pose[:3])))
                candidate=(candidate-center)@Rotation.from_rotvec(pose[:3]).as_matrix().T+center+pose[3:]
            cur = project(candidate,K)
            outside = max(0.,float(np.max(low-cur[-1])),float(np.max(cur[-1]-high)))
            details.append(dict(frame=parent_plan['frames'][i]['frame'],exact_ear_rectangle_excess_px=outside,
                exact_face_max_per_coordinate_change_px=float(np.max(np.abs(cur[:-1]-ref[:-1]))),
                head_rotation_deg=angle))
        drift=[np.linalg.norm(project(b[...,0]+b[...,1:]@x[:10],K)-project(b[...,0],K),axis=1) for b,_ in bodies]
        drift=np.concatenate(drift)
        variants['shared_beta_and_head_rigid' if with_rigid else 'shared_beta_only']=dict(
            linear_min_max_rectangle_excess_px=float(x[-1]),beta_delta_norm=float(np.linalg.norm(x[:10])),
            body_projection_shift_median_px=float(np.median(drift)),body_projection_shift_max_px=float(np.max(drift)),
            checks=details)
    atomic(result_path,dict(plan_sha256=sha(plan_path), replay=dict(max_xyz_mesh=v_err,max_xyz_joints=j_err,
        max_projection_px=uv_err,affine_perturbation_max_model_m=affine_error,frames=24),variants=variants,
        no_model_written=True,quality_accepted=False))
    for name,value in variants.items():
        print(name,value,flush=True)


if __name__=='__main__':
    main()
