"""Cross-person execution of the frozen selected-view SMPLify-X interface."""
from pathlib import Path
import inspect,sys
import numpy as np
import run_grass_selected_views124 as original
from run_official_gaussians123 import read,save,sha,now

SRC=original.ROOT/'winterhat-input'
OUT=original.ROOT/'winterhat-selected-view-pose124'
H=Path(__file__).resolve().parent
scope=original.__dict__.copy();scope.update(SRC=SRC,OUT=OUT)
for name in ('data','root_initialization','make_models','preflight','run'):
    exec(compile(inspect.getsource(getattr(original,name)),'<frozen-eight-view-'+name+'-winterhat>','exec'),scope)


def prepare():
    assert not OUT.exists()
    selection=read(SRC/'selection.json');rows,target=scope['data']()
    assert len(rows)==8 and [r['frame'] for r in rows]==selection['training_frames']
    assert not set(selection['training_frames'])&set(selection['review_only_frames'])
    ref=selection['reference_frame'];assert ref in selection['training_frames']
    for row in rows:
        assert row['wh']==[1520,2704]
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        assert row['K_integer_centers']==rows[0]['K_integer_centers']
        assert np.array_equal(row['w2c'],np.eye(4))
    initial=scope['root_initialization'](rows,ref)
    assert np.allclose(initial,[[0,np.pi,0]])
    sources=[Path(__file__),Path(original.__file__),Path(original.adapter.__file__),Path(original.entry.__file__),
        H/'smplify_score_contract123.py',H/'smplify_observation_validity124.py',
        SRC/'data/inputs.json',SRC/'data/observations.npz',SRC/'selection.json',
        SRC/'reference-fit/winner-camera-space.npz']
    sources += [original.OFFICIAL/f for f in original.entry.PINNED]
    sources += [original.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered',subject='winterhat',frames=selection['training_frames'],
        rows=rows,reference_for_root_seed=ref,review_only_excluded=selection['review_only_frames'],
        hypothesis='Test the same selected-view static local-pose assumption on another person with a raised hand holding the bag strap. Different viewing directions may disambiguate depth.',
        actual_input_difference='Winterhat full-K native1520x2704 pixels, masks, landmarks, own reference71 shared shape. No grass geometry or optimized pose reused.',
        unchanged='Same upstream five stages,30iterations,LBFGS,VPoser/L2/bending,one shared local body latent,per-view root placement and face/finger parameters,mean objective normalization.',
        identity_camera_warning='All source extrinsics are an explicit identity placeholder. Applying the unchanged initialization rule yields the standard front rotation in every view, not a recovered trajectory. Root rotations remain free; this initialization may fail for side/back observations.',
        static_gesture_evidence='Original13 and71 images inspected: right hand near hair/bag strap, other arm lowered, feet standing. Other selected views require actual result review; constant local pose remains a hypothesis.',
        fixed_K=True,K_physically_calibrated=False,shared_betas_frozen=True,old_optimized_pose_used=False,
        baseline_algorithm='grass-selected-view-pose124; code hashes frozen, source subjects remain independent',
        risks=['Identity root seeds can converge to wrong orientation.','Shared local pose can erase actual motion.','Long skirt and hat hide anatomy; close reprojection is not independent3D truth.'],
        budget=dict(candidates=1,selected_views=8,GPU_seconds=600,new_gaussian_steps=0),
        admission='Actual all8-candidate body/face/feet and novel-side review, then about8 quality-first selection; unchanged gates. Never train on reserved review-only frames or fill with bad poses.',
        source_hashes={str(p):sha(p) for p in sources},quality_accepted=False,gaussian_admitted=False))
    code,delta=original.adapter.source(original.OFFICIAL);(OUT/'upstream-interface.diff').write_text(delta)
    _,delta=original.adapter.make_entry(8,initial);(OUT/'entry-interface.diff').write_text(delta)
    print('WINTERHAT_EIGHT_VIEW_REGISTERED',flush=True)


def run():
    scope['run']()
    result=read(OUT/'result.json')
    result.update(conditional_rotation_used_only_as_seed=False,source_camera_identity_placeholder=True,
        actual_root_seed='Every view starts at standard front rotation; optimized per-view roots are body-view estimates, not calibrated scene cameras.',
        old_optimized_pose_used=False,other_subject_geometry_used=False,
        upstream_loss_monitor_optimizer_unchanged=False,
        actual_original_components='Original loss/monitor/LBFGS implementations reused with shared-body broadcast,mean view objective,full K and per-view placement interface.')
    save(OUT/'result.json',result)


if __name__=='__main__':{'prepare':prepare,'preflight':scope['preflight'],'run':run}[sys.argv[1]]()
