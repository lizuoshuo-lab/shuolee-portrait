"""One cross-person use of the existing shared-shape selected-view interface."""
from pathlib import Path
import inspect,sys,os,subprocess
import numpy as np
from PIL import Image
import run_grass_selected_views124 as original
import run_shared_shape_views124 as shared
from run_official_gaussians123 import read,save,sha,now

SRC=original.ROOT/'sofa-input'
OUT=original.ROOT/'sofa-selected-view-pose124'
scope=shared.scope.copy();scope.update(SRC=SRC,OUT=OUT)
exec(compile(inspect.getsource(original.data),'<same-selected-input-sofa>','exec'),scope)
all_data=scope['data']


def data():
    rows,target=all_data()
    review=read(OUT/'source-review.json')
    allowed=review['retained_input_frames']
    ix=[i for i,r in enumerate(rows) if r['frame'] in allowed]
    assert [rows[i]['frame'] for i in ix]==allowed
    return [rows[i] for i in ix],target[ix]


scope['data']=data
runner=shared.run_source
for old,new in [('make_models(8)','make_models(len(p["frames"]))'),
    ('adapter.make_entry(8,','adapter.make_entry(len(p["frames"]),'),
    ('shape==(8,10475,3)','shape==(len(p["frames"]),10475,3)'),
    ("state='fitted_eight_views_review_pending'","state='fitted_selected_views_review_pending'")]:
    assert runner.count(old)==1,old
    runner=runner.replace(old,new)
exec(compile(runner,'<same-shared-shape-run-sofa>','exec'),scope)


def source_review():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    import cv2
    from review_full_sequence123 import crop_box,joined
    rows,target=all_data();assert len(rows)==8
    OUT.mkdir();records=[];tiles=[]
    for i,row in enumerate(rows):
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        box=crop_box(mask.getbbox(),raw.size,35);crop=raw.crop(box);points=np.asarray(crop).copy()
        for ids,color in [(original.entry.COCO,(0,255,0)),(range(25,46),(255,60,60)),(range(46,67),(60,100,255))]:
            for j in ids:
                p=target[i,j]
                if p[2]>0:cv2.circle(points,tuple(np.round(p[:2]-box[:2]).astype(int)),3,color,1,cv2.LINE_AA)
        panel=joined([crop,Image.fromarray(points)])
        path=Path('/mnt/d/00/记录/006')/f'分析_沙发八帧124_{row["frame"]}_原图与身体手部2D观测_原生像素.png'
        assert not path.exists();panel.save(path)
        records.append(dict(frame=row['frame'],image=str(path),sha256=sha(path),native_crop=box,
            input_sha256=row['source_sha256'],actual_visual_review=False))
        thumb=panel.copy();thumb.thumbnail((800,600));tiles.append(thumb)
    pages=[]
    for start in (0,4):
        page=Image.new('RGB',(1600,1200))
        for j,tile in enumerate(tiles[start:start+4]):page.paste(tile,((j%2)*800,(j//2)*600))
        path=OUT/f'source-overview-{start//4+1}.png';page.save(path);pages.append(str(path))
    save(OUT/'source-review.json',dict(updated=now(),state='generated_pending_review',records=records,pages=pages,
        exact_native_pixels_and_unchanged_targets=True,no_new_detector=True,quality_accepted=False))
    print('SOFA_EIGHT_SOURCE_REVIEW_READY',pages,flush=True)


def prepare():
    assert not (OUT/'plan.json').exists()
    review=read(OUT/'source-review.json');assert review['actual_visual_review'] and review['static_gesture_plausible']
    selection=read(SRC/'selection.json');rows,target=scope['data']()
    ids=[r['frame'] for r in rows]
    assert len(rows)==7 and ids==review['retained_input_frames']
    assert set(ids)<set(selection['training_frames'])
    assert review['rejected_input_frames']==['frame_000053']
    assert not set(selection['training_frames'])&set(selection['review_only_frames'])
    ref=selection['reference_frame'];assert ref in selection['training_frames']
    wrist=read(original.ROOT/'selected-observation-audit124/wrist-preselection/sofa.json')
    guards={r['frame']:r['screen_flags'] for r in wrist['records']}
    assert all(not guards[r['frame']] for r in rows)
    for row in rows:
        assert row['wh']==[2704,1520] and row['K_integer_centers']==rows[0]['K_integer_centers']
        assert np.array_equal(row['w2c'],np.eye(4))
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
    sources=[Path(__file__),Path(original.__file__),Path(shared.__file__),Path(original.adapter.__file__),
        Path(original.entry.__file__),original.H/'smplify_score_contract123.py',original.H/'smplify_observation_validity124.py',
        SRC/'data/inputs.json',SRC/'data/observations.npz',SRC/'selection.json',SRC/'reference-fit/winner-camera-space.npz',
        OUT/'source-review.json',original.ROOT/'sofa-body-placement124/conclusion.json',
        original.ROOT/'selected-observation-audit124/wrist-preselection/sofa.json',
        original.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    sources += [original.OFFICIAL/f for f in original.entry.PINNED]
    import smplx
    sources += [Path(smplx.__file__).parent/'body_models.py',Path(smplx.__file__).parent/'lbs.py']
    save(OUT/'plan.json',dict(updated=now(),state='registered',subject='sofa',frames=ids,
        rows=rows,reference_for_root_seed=ref,review_only_excluded=selection['review_only_frames'],
        source_rejected_frames=review['rejected_input_frames'],source_exclusion_before_any_fit=True,
        hypothesis='Apply the existing shared-shape/local-pose interface to the approximately constant seated gesture. Joint views and freely optimized body placement may resolve the old per-frame seated-pose mismatch.',
        user_basis='About8good dispersed frames, not every video frame. Shared body shape may update; original-repo components and minimal interfaces preferred.',
        algorithm='Exact existing shared.run_source and selected-view adapter; no new loss, thresholds, optimizer or stage schedule. Shared10Dbeta and32DVPoser; per-view root/translation/face/fingers.',
        initialization='Own original reference26beta seed, original zero VPoser latent and standard front root. No optimized pose or shape from any rejected control reused.',
        identity_camera_warning='All input extrinsics are identity placeholders. Per-view roots and translations are body-view estimates, not physically calibrated scene cameras.',
        scope='Cross-person pipeline test, not another single-variable placement claim: sharedpose/sharedshape/joint objective differ from the old independent temporal run.',
        original_inputs_K_and_priors_preserved=True,shared_shape_can_update=True,static_gesture_assumption=True,
        risks=['Shared local pose can suppress real small motions.','Overlapping hands and clothing hide anatomical centers.','Identity front root seeds can fail for side views.'],
        budget=dict(candidates=1,selected_views=7,GPU_seconds=600,new_gaussian_steps=0),
        admission='Use unchanged original-supported body/feet3percent gate, source ownership and actual native/side review. Do not require all56frames. Reject bad candidate frames; never promote by averaged loss or fill with heldout frames.',
        stop='One fit and review; no parameter/threshold/roster sweep or automatic GS. No resuming old47frame solver.',
        source_hashes={str(p):sha(p) for p in sources},quality_accepted=False,gaussian_admitted=False))
    _,delta=original.adapter.source(original.OFFICIAL);(OUT/'upstream-interface.diff').write_text(delta)
    _,delta=original.adapter.make_entry(len(rows),original.root_initialization(rows,ref));(OUT/'entry-interface.diff').write_text(delta)
    (OUT/'executed-runner.py.txt').write_text(runner)
    print('SOFA_SHARED_SELECTED_REGISTERED',len(rows),flush=True)


def preflight():
    from types import SimpleNamespace
    proxy=SimpleNamespace(**dict(original.__dict__,SRC=SRC))
    ns=shared.__dict__.copy();ns.update(OUT=OUT,original=proxy)
    code=inspect.getsource(shared.preflight).replace('for n in (1,8):','for n in (1,len(plan["frames"])):')
    code=code.replace('if n==8:','if n==len(plan["frames"]):').replace('original_shape_prior_not_divided_by8','original_shape_prior_not_divided_byN')
    exec(compile(code,'<same-shared-beta-preflight-sofa>','exec'),ns)
    ns['preflight']()
    r=read(OUT/'preflight.json');r.update(original_shared_interface_reused=True,new_fit_calls=0,
        heldout_excluded=read(OUT/'plan.json')['review_only_excluded']);save(OUT/'preflight.json',r)


def run():
    scope['run']()
    r=read(OUT/'result.json')
    with np.load(OUT/'selected-fit.npz') as z:beta=z['betas'].copy()
    with np.load(SRC/'reference-fit/winner-camera-space.npz') as z:seed=z['betas'].copy()
    r.update(beta_before=seed.tolist(),beta_after=beta.tolist(),beta_delta_l2=float(np.linalg.norm(beta-seed)),
        shared_beta_parameter_count=10,shared_shape_updated=True,freeze_betas=False,
        conditional_rotation_used_only_as_seed=False,source_camera_identity_placeholder=True,
        upstream_loss_monitor_optimizer_unchanged=False,actual_reuse='Existing shared-shape and mean-view interfaces, with original loss/monitor/LBFGS implementations and stages/weights.',
        actual_root_seed='All front rotations. Roots/placements are body-view estimates, not scene camera recovery.',
        quality_accepted=False,gaussian_admitted=False)
    save(OUT/'result.json',r)


def launch():
    assert not (OUT/'progress.json').exists()
    with (OUT/'fit.log').open('x') as f:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'run']
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        save(OUT/'parent-launch.json',dict(updated=now(),pid=child.pid,command=command,
            proc_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]))
        print('SOFA_SHARED_EIGHT_STARTED',child.pid,flush=True)
        rc=child.wait()
    save(OUT/'process-exit.json',dict(updated=now(),exit_code=rc,pid=child.pid))
    if rc:raise subprocess.CalledProcessError(rc,command)
    print('SOFA_SHARED_EIGHT_FINISHED',flush=True)


if __name__=='__main__':{'source-review':source_review,'prepare':prepare,'preflight':preflight,'run':run,'launch':launch}[sys.argv[1]]()
