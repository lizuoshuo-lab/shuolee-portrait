"""Read exact R21 10k and 18k snapshots through the normal bound export contract."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

import review_smpl_delivery_round20 as prior


pin,read,sha,require,descriptor,rebound = prior.pin,prior.read,prior.sha,prior.require,prior.descriptor,prior.rebound
review,round1,r13 = prior.review,prior.round1,prior.r13


def planned_model(run, project, expected, steps):
    """Validate the same frozen native roster at the declared actual checkpoint."""
    terminal_path = pin(run/'result.json',expected)
    terminal = read(terminal_path)
    require(terminal['state']=='completed' and terminal['format_passed'] and terminal['steps']==steps
            and terminal['final_snapshot']==f'step-{steps:06d}',f'Unexpected R21 checkpoint: {run}')
    rows_path = run/'data/training-inputs.json'
    rows = read(rows_path)['records']
    by_frame = {row['frame']:row for row in rows}
    holds = read(run/'fit/selection.json')['retained_holdouts']
    require(len(rows)==len(by_frame)==24 and set(holds).isdisjoint(by_frame),'Invalid protected native pose roster.')
    snapshot,replay_file,bound = round1.delivery_paths(run,terminal)
    replay = read(replay_file)
    require(replay['retained_holdouts']==holds and snapshot.name==terminal['final_snapshot'],
            'The terminal bound replay belongs to another checkpoint or evaluation roster.')
    pin(rows_path,bound['inputs_sha256']['training_rows'])
    for name,digest in bound['source_files_sha256'].items():
        pin(project/name,digest)
    path = pin(run/'fit/fitted.npz',bound['inputs_sha256']['fitted'])
    with np.load(path,allow_pickle=False) as archive:
        fit = {name:archive[name] for name in (*review.POSE_KEYS,'frame_ids','faces','shared_betas')}
    slots = {name:index for index,name in enumerate(fit['frame_ids'].tolist())}
    require(list(slots)==list(by_frame),'Frozen pose slots differ from the selected frame order.')
    provenance = dict(run=str(run),project=str(project),terminal_sha256=expected,
                      training_inputs_sha256=bound['inputs_sha256']['training_rows'],fitted_sha256=bound['inputs_sha256']['fitted'],
                      protected_holdouts=holds,reference_frame_id=terminal['reference_frame_id'],points=terminal['points'],
                      gaussian_profile=terminal['gaussian_profile'],from_scratch_receipt=terminal['from_scratch'],steps=steps)
    return SimpleNamespace(provenance=provenance,rows=by_frame,fit=fit,slots=slots)


def portable(provenance,native,parts):
    """Read the archived 3DGS camera and exact shipped model bytes for either checkpoint."""
    run = Path(provenance['run'])
    terminal = read(run/'result.json')
    if terminal['steps']==10000:
        return r13.portable(provenance,native,parts)
    path,camera_path = (run/terminal[key] for key in ('ply','camera'))
    pin(path,terminal['sha256'])
    provenance_path = camera_path.with_name('camera_provenance.json')
    receipt = read(provenance_path)
    camera_sha = receipt['files_sha256']['cameras.json']
    pin(camera_path,camera_sha)
    with ZipFile(pin(run/terminal['scene'],terminal['scene_sha256'])) as bundle:
        items = [item for item in bundle.namelist() if Path(item).name==path.name]
        require(len(items)==1 and items[0].startswith('point_cloud/iteration_')
                and hashlib.sha256(bundle.read(items[0])).hexdigest()==terminal['sha256']
                and hashlib.sha256(bundle.read('cameras.json')).hexdigest()==camera_sha,
                'The portable model or camera differs inside the scene archive.')
    camera_doc, = read(camera_path)
    row, = receipt['records']
    require(receipt['cameras_share_model_world'] and row['frame']==native['frame']
            and camera_doc['pixelConvention']=='pixel_corners' and camera_doc['projectionModel']=='pinhole',
            'The portable projection or fixed reference frame differs.')
    R = np.asarray(camera_doc['rotation']).T
    E = np.eye(4)
    E[:3,:3],E[:3,3] = R,-R@np.asarray(camera_doc['position'])
    K = [[camera_doc['fx'],0.,camera_doc['cx']],[0.,camera_doc['fy'],camera_doc['cy']],[0.,0.,1.]]
    view = dict(frame=native['frame'],wh=[camera_doc['width'],camera_doc['height']],K=K,w2c=E.tolist())
    transform = receipt['world_to_viewer']
    expected = r13.exported_view(native,transform)
    require(view['K']==row['K']==expected['K'] and view['wh']==row['wh']==expected['wh']
            and np.allclose(E,row['w2c'],rtol=0,atol=1e-12) and np.allclose(E,expected['w2c'],rtol=0,atol=1e-12)
            and transform['scale']==1. and parts[0].is_dir(),'Portable camera is not the exact native projection.')
    return dict(ply=str(path),ply_sha256=terminal['sha256'],points=terminal['points'],camera_path=str(camera_path),
                camera_sha256=camera_sha,front=view,world_to_viewer=transform)


def common_plans(old,new,locked,parts):
    aliases = SimpleNamespace(portable=portable,mesh_parts=r13.mesh_parts,common_centers=r13.common_centers,
                              exported_view=r13.exported_view)
    return rebound(prior.r14.common_plans,r13=aliases)(old,new,locked,parts)


def active_pose_proofs(run,wrapper,terminal):
    snapshot = terminal['final_snapshot']
    heldout = Path('gaussian-run')/snapshot/'heldout-replay'
    return rebound(prior.pose.proof_files,HELDOUT=heldout)(run,wrapper,terminal)


def pose_plans(run,plan,terminal):
    snapshot = terminal['final_snapshot']
    old = run/'gaussian-run/step-010000/bound-replay.json'
    correct = run/'gaussian-run'/snapshot/'bound-replay.json'

    def active_read(path):
        return read(correct if path==old else path)

    supplement = rebound(prior.pose.supplement_plan,read=active_read)(run,plan)
    heldout_dir = Path('gaussian-run')/snapshot/'heldout-replay'
    holdout = rebound(prior.pose.holdout_plan,read=active_read,HELDOUT=heldout_dir)(run,dict(plan,supplement=supplement))
    return supplement,holdout
