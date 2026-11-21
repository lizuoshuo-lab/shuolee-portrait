"""Transfer the reviewed ROI2x/SH0/18k profile to eight winterhat views."""
import ast,copy,inspect,os,sys,shutil,signal,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import train_sofa_selected_sh0124 as native
import train_indoor_person_zoom124 as zoom
from train_indoor_circle16_sh0124 import reused,replace
import cross_subject_sh0124 as gs

SUBJECT='winterhat-half8-paired18k124'
OUT=gs.ROOT/SUBJECT
FIT=gs.ROOT/'winterhat-half8-pose124'
HANDOFF=FIT/'gaussian-native-handoff'
BASELINE=gs.ROOT/'indoor-paired-background18k124'

def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists() and not HANDOFF.exists()
    review=gs.read(FIT/'conclusion.json');assert review['actual_visual_review'] and review['exploratory_gaussian_admitted']
    # Refuse stale per-view winterhat betas or an unreviewed input. This new
    # eight-view fit uses one freshly fitted10D shape, identity E and reference71.
    expected=['frame_000016','frame_000028','frame_000041','frame_000056',
        'frame_000057','frame_000071','frame_000096','frame_000109']
    source_plan=gs.read(FIT/'plan.json');fit_result=gs.read(FIT/'result.json')
    assert [r['frame'] for r in source_plan['rows']]==expected
    assert source_plan['old_individual_shape_used'] is False and source_plan['old_optimized_pose_used'] is False
    assert fit_result['shared_beta_parameter_count']==10 and fit_result['old_individual_shape_used'] is False
    assert fit_result['hand_data_control']['candidate']==[0.,0.,0.,0.,0.]
    assert fit_result['hand_data_control']['same_for_all_frames']
    for row in source_plan['rows']:
        assert row['wh']==[1520,2704] and np.array_equal(row['w2c'],np.eye(4))
    with np.load(FIT/'selected-fit.npz',allow_pickle=False) as archive:
        assert archive['frame_ids'].astype(str).tolist()==expected
        assert archive['betas'].shape==(1,10),archive['betas'].shape
    # The reused native handoff recomputes SMPL-X roots, vertices and the
    # clothing-offset bridge from this fit, then checks the original-forward
    # oracle for all eight frames; no old winterhat Gaussian/fit cache is used.
    edits=[('assert n == 7','assert n == 8'),('沙发人物7帧','冬帽人物8帧'),
        ("'assert len(training_slots)==7'","'assert len(training_slots)==8'"),
        ("'training_frames=7'","'training_frames=8'"),('seven-frame-interface.diff','eight-frame-interface.diff'),
        ('Seven preregistered sofa inputs','Eight preregistered winterhat inputs'),('All seven preregistered','All eight preregistered'),
        ("record_id='006'","record_id='001'")]
    fn,diff=reused(native.prepare,edits,dict(OUT=HANDOFF,SOURCE=FIT,SUBJECT=SUBJECT,REFERENCE='frame_000071',__file__=__file__),'eight-native-handoff')
    fn();(HANDOFF/'eight-handoff.diff').write_text(diff)
    # The winterhat sequence has visible boots and a long skirt; anatomy under the skirt is unknown: do not transfer the
    # indoor furniture/hip-cut abstention rule to this different observation.
    # Full raster support is the ordinary original masked RGB loss. Off-raster
    # feet and unseen rear remain unknown; no extrapolated pixels are created.
    doc=gs.read(HANDOFF/'data/training-inputs.json');(HANDOFF/'data/validity').mkdir()
    for row in doc['records']:
        p=HANDOFF/'data/validity'/(row['frame']+'.png')
        Image.new('L',tuple(row['wh']),255).save(p)
        row.update(validity=str(p),validity_sha256=gs.sha(p))
    gs.save(HANDOFF/'data/training-inputs.json',doc)
    rec=gs.read(HANDOFF/'smplify-gaussian-interface-receipt.json');rec['training_inputs_sha256']=gs.sha(HANDOFF/'data/training-inputs.json');gs.save(HANDOFF/'smplify-gaussian-interface-receipt.json',rec)
    shutil.copyfile(BASELINE/'loss_utils_validity.py',HANDOFF/'loss_utils_validity.py')
    p=gs.read(HANDOFF/'plan.json')
    for path in p['source_hashes']:
        if Path(path).is_relative_to(HANDOFF):p['source_hashes'][path]=gs.sha(path)
    gs.save(HANDOFF/'plan.json',p)
    before=inspect.getsource(zoom.prepare)
    section=before[before.index("    original=(BASE/'adapter.py')"):before.index('    plan=copy.deepcopy(baseline)')]
    edits=[("assert gs.read(BASE/'gaussian-run/result.json')['state']=='bounded_control_completed'","assert gs.read(BASE/'smplify-gaussian-interface-receipt.json')['frames']==8"),
        ('assert len(source_rows)==6','assert len(source_rows)==8'),
        ("sheet=Image.new('RGB',(1800,1380),(35,35,35))","sheet=Image.new('RGB',(1800,460*((len(panels)+1)//2)),(35,35,35))"),
        ('all6_projection_checks=True','all8_projection_checks=True'),
        (section,"    original=(BASE/'adapter.py').read_text()\n    (OUT/'adapter.py').write_text(original)\n")]
    fn,diff=reused(zoom.prepare,edits,dict(BASE=HANDOFF,OUT=OUT,SUBJECT=SUBJECT,LIMIT=18000,__file__=__file__),'eight-roi2x')
    fn();(OUT/'eight-roi-interface.diff').write_text(diff)
    before=(BASELINE/'adapter.py').read_text();code=before
    edits=[("receipt['frames'] == 6","receipt['frames'] == 8"),('self.num_timesteps = 6','self.num_timesteps = 8'),
        ('assert len(training_slots)==6','assert len(training_slots)==8'),('counts = np.zeros(6, dtype=int)','counts = np.zeros(8, dtype=int)'),
        ('frames_rendered=6','frames_rendered=8'),('training_frames=6, review_frames=6','training_frames=8, review_frames=8'),
        ("    from known_pixel_loss124 import KnownPixelLoss\n    validity_loss = module(OUT/'loss_utils_validity.py', 'indoor_known_pixel_loss')\n    photometric = KnownPixelLoss(validity_loss)\n    l1_loss, ssim = photometric.l1, photometric.ssim","    from utils.loss_utils import l1_loss, ssim"),
        ('        photometric.use_camera(cam)\n',''),('assert set(np.unique(valid)) == {0, 255}','assert set(np.unique(valid)) == {255}'),
        ('Same indoor ROI2x and18k clock; paired stochastic background; reviewed1000 and hard stop18000','Winterhat eight views with ROI2x and18k clock; paired background; reviewed1000 and hard stop18000'),
        ('Only paired stochastic training background relative to indoor zoom clock; original RGB loss and all options unchanged','Winterhat eight-view input; same ROI2x/paired background/SH0/18k profile; original full-raster masked RGB loss')]
    for a,b in edits:code=replace(code,a,b)
    assert code.count("'frame_000105'")==2;code=code.replace("'frame_000105'","'frame_000071'")
    ast.parse(code);(OUT/'adapter.py').write_text(code)
    import difflib
    (OUT/'profile-transfer.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='indoor-completed-profile',tofile='winterhat-eight-profile')))
    old=gs.read(BASELINE/'plan.json');p=gs.read(OUT/'plan.json')
    for key in ('options','upstream_options','schedule_overrides','shared_laplacian_weight','goal_points_approx','goal_point_band','max_points_safety_stop'):p[key]=copy.deepcopy(old[key])
    p.update(updated=gs.now(),state='prepared_input_review_pending',subject=SUBJECT,label='冬帽人物 · 半圈8帧 SH0',record_id='001',
        frames=8,training_frames=8,evaluation_frames=8,source_fit=str(FIT/'selected-fit.npz'),reference_frame='frame_000071',
        user_authorization='半圈8，一圈16；继续可复现全流程训练。',
        hypothesis='Transfer the ROI2x, paired-background and18k clock profile to eight reviewed winterhat inputs.',
        only_parameter_change='Subject-specific eight-frame fresh fit and source masks. The indoor occlusion abstention is not applicable: winterhat keeps ordinary full-raster masked loss.',
        causal_scope='Cross-person pipeline comparison. Versus old winterhat8, roster/neutral-seed refit/hand2D abstention/ROI/background/time schedule change together; not isolated view-count evidence.',
        fixed='SH0;2x person ROI;original loss and LR amplitudes;18k time schedule;shared clothing offset1e-6;protected face/head/hands/feet;reference71.',
        visibility_rule='Original person mask, all recorded raster pixels supervised; pixels outside the source image and unseen rear do not exist as observations. No indoor hip cutoff.',
        new_loss_validity_enabled=False,new_pose_fitting=True,source_fitted_pose_reused_for_control=False,
        source_hand2d_data_uniformly_disabled=True,source_shared_beta_shape=[1,10],old_winterhat_fit_cache_reused=False,
        from_scratch=True,optimizer_state_reused=False,gaussian_parameters_reused=False,
        comparison_budget=18000,configured_schedule_steps=18000,snapshots=[1000,10000,18000],
        criteria='Actual all8native and novel-view review at1000; continue same optimizer only without new gross anatomy failure. Final all8/continuous views and exact camera/PLY review.',
        budget_note='One candidate, max18000steps/2400seconds including review; no automatic retries.',
        known_limitation='Half-circle or less, no rear observations. Approximately static upright gesture and conditional K remain assumptions; far hand fingers occluded in several views are unknown and unused by the uniform original hand2D fit data term.',
        quality_accepted=False,model_promoted=False,end_to_end_15min_accepted=False)
    for path in list(p['source_hashes']):
        if Path(path).is_relative_to(OUT):p['source_hashes'][path]=gs.sha(path)
    for path in [Path(__file__),Path(native.__file__),Path(zoom.__file__),BASELINE/'adapter.py',BASELINE/'plan.json',FIT/'conclusion.json',gs.HERE/'paired_training_background124.py',OUT/'profile-transfer.diff']:
        p['source_hashes'][str(path)]=gs.sha(path)
    gs.save(OUT/'plan.json',p)
    gs.save(OUT/'validity-review.json',dict(updated=gs.now(),actual_visual_review=False,training_admitted=False,
        rule=p['visibility_rule'],source_mask_unchanged=True,quality_accepted=False))
    gs.save(OUT/'progress.json',dict(updated=gs.now(),state=p['state'],step=0,pid=None))
    print('WINTERHAT_EIGHT_PROFILE_PREPARED',flush=True)

def preflight():
    edits=[("dataset=ctx['Images']();results=[];torch.cuda.reset_peak_memory_stats()",
        "dataset=ctx['Images']();results=[];torch.cuda.reset_peak_memory_stats()\n    from paired_training_background124 import PairedTrainingBackground\n    paired=PairedTrainingBackground(ctx['render'],ctx['masks'],[r['frame'] for r in ctx['rows']],ctx['target_cache'])"),
        ("output=ctx['render'](cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image","output=paired(cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image"),
        ('assert not model.optimizer.state','assert paired.calls==8\n    assert not model.optimizer.state')]
    fn,diff=reused(zoom.preflight,edits,dict(OUT=OUT,LIMIT=18000),'winterhat-eight-paired-preflight')
    (OUT/'preflight-interface.diff').write_text(diff);fn('eight-preflight')

def train():
    assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
    assert gs.read(OUT/'validity-cuda-preflight.json')['passed'] and gs.read(OUT/'validity-review.json')['training_admitted']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Registered2400seconds')));signal.alarm(2400)
    try:gs.train(SUBJECT)
    finally:signal.alarm(0)

if __name__=='__main__':
    try:
        mode=sys.argv[1];{'prepare':prepare,'preflight':preflight,'train':train}[mode]()
        if mode=='train':gs.save(OUT/'process-exit.json',dict(updated=gs.now(),exit_code=0))
    except BaseException as e:
        if OUT.exists():gs.save(OUT/(sys.argv[1]+'-failure.json'),dict(updated=gs.now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
