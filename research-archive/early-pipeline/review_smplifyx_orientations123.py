"""CPU review of both saved upstream orientations, retaining every observation."""
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import cv2
from review_full_sequence123 import mesh_renderer, joined
import control_smplifyx_orientations123 as run


def load(path):
    with np.load(path, allow_pickle=False) as value:
        return {k:value[k].copy() for k in value.files}


def metrics(data, targets):
    joints = data['joints_camera'][0].astype(np.float64)
    assert np.isfinite(joints).all() and np.all(joints[:,2]>0)
    q = joints @ data['K_integer_centers'].T
    uv = q[:,:2]/q[:,2:]
    error = np.linalg.norm(uv-targets[:,:2],axis=1)
    active = targets[:,2]>0
    active[[1,9,12]] = False
    groups = dict(body17=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]),
        feet6=np.arange(19,25),left_hand21=np.arange(25,46),right_hand21=np.arange(46,67),face51=np.arange(67,118))
    result = {}
    for name, ids in groups.items():
        selected = error[ids[active[ids]]]
        result[name] = dict(count=len(selected),median=float(np.median(selected)),
            p90=float(np.quantile(selected,.9)),maximum=float(selected.max()))
    return dict(groups=result,left_elbow_px=float(error[6]),right_elbow_px=float(error[3]),
        per_joint=[dict(index=i,active=bool(active[i]),confidence=float(targets[i,2]),error_px=float(error[i])) for i in range(118)])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    completion = run.read(run.OUT/'result.json')
    assert completion['state'] == 'complete_pending_review'
    run.verify_plan()
    out = run.OUT/'review'
    out.mkdir(exist_ok=True)
    assert not (out/'comparison.json').exists()
    cv2.setNumThreads(2)
    baseline = load(run.BASE/'fit/fitted.npz')
    ids = list(baseline['frame_ids'].astype(str))
    rows = {r['frame']:r for r in run.read(run.BASE/'data/training-inputs.json')['records']}
    mesh, renderer_source = mesh_renderer()
    record = Path('/mnt/d/00/记录/demo')
    record.mkdir(exist_ok=True)
    compared, images = [], []
    for name in run.FRAMES:
        row = rows[name]
        assert run.sha(row['image']) == row['source_sha256']
        raw = Image.open(row['image']).convert('RGB')
        target = baseline['observed_native_keypoints'][ids.index(name)]
        folder = run.OUT/'frames'/name
        trace = run.read(folder/'candidate-selection.json')
        old = load(run.BASE/'frames'/name/'winner-camera-space.npz')
        candidates = [load(folder/f'candidate-{i:02d}.npz') for i in (0,1)]
        winner = load(folder/'winner-camera-space.npz')
        assert np.array_equal(winner['vertices_camera'],candidates[trace['selected_original']]['vertices_camera'])
        for candidate in candidates:
            assert np.array_equal(candidate['betas'],old['betas'])
            assert np.array_equal(candidate['K_integer_centers'],old['K_integer_centers'])
        before = metrics(old,target)
        candidate_metrics = [metrics(c,target) for c in candidates]
        after = candidate_metrics[trace['selected_original']]
        protections = {k:dict(delta_px=after['groups'][k]['median']-before['groups'][k]['median'],
            tolerance_px=tolerance,passed=after['groups'][k]['median']-before['groups'][k]['median']<=tolerance)
            for k,tolerance in [('body17',1.),('feet6',1.),('face51',.5)]}
        overlays = [mesh(raw,np.array([0,0,*raw.size]),c) for c in candidates]
        reference = run.read(run.BASE/'review'/(name+'_saved-review.json'))
        per_frame_images = []
        for region in ('body','face','left_hand','right_hand','feet'):
            original = next(x for x in reference['files'] if x['region']==region)
            crop = original['crop_integer_xyxy']
            panels = [raw.crop(crop),overlays[0].crop(crop),overlays[1].crop(crop),overlays[trace['selected_original']].crop(crop)]
            path = out/f'{name}_{region}_raw_original_alternate_selected.png'
            assert not path.exists()
            joined(panels).save(path)
            check = Image.open(path).convert('RGB')
            assert np.array_equal(np.array(check)[:,:panels[0].width],np.array(raw.crop(crop)))
            destination = record/f'SMPL-X_原版双朝向_{name}_{region}_原图_原朝向_另一朝向_原库选择.png'
            assert not destination.exists()
            destination.write_bytes(path.read_bytes())
            item = dict(frame=name,region=region,path=str(path),record_path=str(destination),sha256=run.sha(path),
                crop_integer_xyxy=crop,panels=['raw','original_orientation0','alternate_orientation1','official_selected'],
                selected=trace['selected_original'],no_resampling=True,no_text_added=True)
            images.append(item); per_frame_images.append(item)
        compared.append(dict(frame=name,baseline=before,candidates=candidate_metrics,selected=trace['selected_original'],
            original_selection=trace,protections=protections,left_elbow_remaining_ratio=after['left_elbow_px']/before['left_elbow_px'],images=per_frame_images))
    parity = completion['baseline_parity_passed']
    reduction = all(r['left_elbow_remaining_ratio']<=.5 for r in compared if r['frame'] in ('frame_000016','frame_000018'))
    protected = all(p['passed'] for r in compared for p in r['protections'].values())
    ranking = all(r['original_selection']['objective_selection_agrees'] for r in compared)
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    result = dict(state='comparison_complete_visual_review_pending',updated=run.now(),frames=compared,images=images,
        elapsed_seconds=completion['elapsed_seconds'],baseline_parity_passed=parity,elbow_reduction_passed=reduction,
        protections_passed=protected,saved_loss_ranking_agrees=ranking,numerical_criteria_passed=parity and reduction and protected and ranking,
        actual_visual_review=False,quality_accepted=False,full59_admitted=False,Gaussian_training_started=False,
        source_hashes={str(p):run.sha(p) for p in (Path(__file__),renderer_source,run.PLAN,run.OUT/'result.json')},
        scope='Fitted detector residuals only; not independent physical truth. CPU mesh overlay uses existing depth-sorted triangle display, not exact z-buffer.')
    run.save(out/'comparison.json',result)
    print(__import__('json').dumps({k:result[k] for k in ('baseline_parity_passed','elbow_reduction_passed','protections_passed','saved_loss_ranking_agrees','numerical_criteria_passed')}))


if __name__ == '__main__': main()
