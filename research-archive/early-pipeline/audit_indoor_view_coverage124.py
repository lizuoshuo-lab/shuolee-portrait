"""Read-only view coverage and a deterministic larger input shortlist.

Fitted root directions are a diagnostic proxy, not calibrated cameras.
No fitting, image warping, renderer changes or Gaussian updates are performed.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from PIL import Image

H = Path(__file__).resolve().parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
ROOT = T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'
OUT = ROOT/'indoor-view-coverage124'
BASE = ROOT/'indoor-selected-sh0-pipeline124'
SOURCE = ROOT/'indoor-input'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def host(p):
    p = str(p)
    if p.startswith('/mnt/c/'): return Path('C:/'+p[7:])
    if p.startswith('/mnt/d/'): return Path('D:/'+p[7:])
    if p.startswith('/opt/'): return Path('//wsl.localhost/Ubuntu-20.04'+p)
    return Path(p)


def main():
    assert not OUT.exists(), 'Keep one immutable audit; inspect an existing result instead.'
    OUT.mkdir()
    paths = [Path(__file__), BASE/'fit/fitted.npz', BASE/'selection.json',
             SOURCE/'selection.json', SOURCE/'data/inputs.json',
             ROOT/'selected-observation-audit124/wrist-preselection/indoor.json',
             ROOT/'indoor-directional-support124/result.json',
             ROOT/'indoor-selected-view-pose124/conclusion.json']
    hashes = {str(p):sha(p) for p in paths}
    save(OUT/'plan.json', dict(updated=datetime.now(timezone.utc).isoformat(),
        purpose='Audit whether six training views leave angular gaps, and prepare one larger reproducible input shortlist.',
        scope='CPU only; existing six fitted roots and all111 frozen input records. No new pose, cameras or training.',
        root_formula='world=(root_zero_vertices-pivot)@R.T+pelvis_world; body-frame ray=R.T@(camera_world-pelvis_world).',
        angle_warning='Conditional fitted-body directions, not independent scene trajectory or physical camera truth.',
        shortlist_rule='Keep existing six. Split chronological111 slots into24 equal-population bins. In each empty bin, select minimum existing prefit quality score among non-heldout, wrist-screen-clear inputs; tie by frame ID. Never refill using heldout or contradictory inputs.',
        shortlist_not_admission=True, requested_count=24, source_hashes=hashes,
        optimizer_updates=0, source_mutations=False))
    fit = dict(np.load(BASE/'fit/fitted.npz', allow_pickle=False))
    selection = read(SOURCE/'selection.json')
    rows = read(SOURCE/'data/inputs.json')['records']
    index = {r['frame']:r for r in rows}
    candidates = {r['frame']:r for r in selection['candidates']}
    ids = fit['frame_ids'].astype(str).tolist()
    assert len(ids) == 6 and set(ids) == set(read(BASE/'selection.json')['selected_frames'])
    R = fit['root_rotation']; E = fit['w2c']; pelvis = fit['world_translation']
    assert np.max(abs(R.transpose(0,2,1)@R-np.eye(3))) < 1e-6
    assert np.allclose(np.linalg.det(R), 1, atol=1e-6)
    assert np.allclose(fit['body_pose_axis_angle'], fit['body_pose_axis_angle'][0], atol=0)
    camera = -np.einsum('nji,nj->ni', E[:,:3,:3], E[:,:3,3])
    delta = camera-pelvis
    ray = np.einsum('nji,nj->ni', R, delta)
    reconstruction_error = float(abs(np.einsum('nij,nj->ni',R,ray)-delta).max())
    assert reconstruction_error < 1e-10
    yaw = np.degrees(np.arctan2(ray[:,0], ray[:,2])) % 360
    elevation = np.degrees(np.arctan2(ray[:,1],np.linalg.norm(ray[:,[0,2]],axis=1)))
    angles = [dict(frame=f, time_seconds=index[f]['time_seconds'],
        yaw_proxy_deg=float(yaw[i]), elevation_proxy_deg=float(elevation[i])) for i,f in enumerate(ids)]
    order = np.argsort(yaw)
    gaps = [dict(start_frame=ids[int(order[i])], end_frame=ids[int(order[(i+1)%6])],
        degrees=float((yaw[order[(i+1)%6]]-yaw[order[i]])%360)) for i in range(6)]
    gaps.sort(key=lambda r:r['degrees'], reverse=True)
    held = set(selection['review_only_frames'])
    guards = {r['frame']:r['screen_flags'] for r in read(paths[5])['records']}
    chronological = sorted(rows, key=lambda r:(r['time_seconds'], r['frame']))
    chosen = set(ids); bins = []
    for bi, group in enumerate(np.array_split(np.arange(len(rows)),24)):
        names = [chronological[int(i)]['frame'] for i in group]
        retained = sorted(chosen.intersection(names))
        eligible = [f for f in names if f not in held and not guards[f]]
        if not retained and eligible:
            f = min(eligible, key=lambda f:(candidates[f]['score'],f))
            chosen.add(f); retained = [f]
        bins.append(dict(bin=bi,source_frames=names,selected_frames=retained,
            eligible_frames=eligible,empty=not bool(retained)))
    chosen = [r['frame'] for r in chronological if r['frame'] in chosen]
    assert set(ids) <= set(chosen) and not set(chosen)&held
    assert all(not guards[f] for f in chosen)
    samples = []
    for f in chosen:
        row = index[f]
        assert sha(host(row['image'])) == row['source_sha256']
        assert sha(host(row['mask'])) == row['mask_sha256']
        raw = Image.open(host(row['image'])).convert('RGB')
        mask = Image.open(host(row['mask'])).convert('L')
        box = mask.getbbox(); assert box
        tile = raw.crop(box); tile.thumbnail((320,430))
        samples.append((f,tile))
    pages = []
    for first in range(0,len(samples),8):
        chunk=samples[first:first+8]
        sheet=Image.new('RGB',(1280,860),(24,24,24))
        for i,(_,im) in enumerate(chunk): sheet.paste(im,((i%4)*320,(i//4)*430))
        name=f'candidate-{first//8+1:02d}.png'; sheet.save(OUT/name)
        record=Path('D:/00/记录/008')/f'分析_室内视角覆盖124_候选{first//8+1:02d}_原图.png'
        assert not record.exists();sheet.save(record)
        pages.append(dict(path=str(OUT/name),record=str(record),frames=[f for f,_ in chunk],sha256=sha(OUT/name)))
    current = Image.new('RGB',(1920,430),(24,24,24))
    selected_order = [f for f in chosen if f in ids]
    for i,f in enumerate(selected_order):
        im=next(t for n,t in samples if n==f);current.paste(im,(320*i,0))
    current.save(OUT/'current-six.png')
    record=Path('D:/00/记录/008/分析_室内视角覆盖124_当前六帧_原图.png')
    assert not record.exists(); current.save(record)
    historical = read(ROOT/'indoor-directional-support124/result.json')
    support = [dict(model=r['name'], contribution_weighted_effective_view_median=
        r['regions']['all']['metrics']['effective_views']['contribution_weighted_median']) for r in historical['reports']]
    result=dict(updated=datetime.now(timezone.utc).isoformat(), state='readonly_audit_complete_visual_review_pending',
        input_count=len(rows),current_training_count=6, current_training_order=selected_order,
        current_angles=angles,circular_gaps=gaps,elevation_proxy_range_deg=[float(elevation.min()),float(elevation.max())],
        ray_transform_roundtrip_max_abs=reconstruction_error, camera_calibration_claimed=False,
        source_selector='Chronological8bins then wrist exclusions; not geometric equal-angle selection; no gap refill after rejecting31and89.',
        original_rejected_frames=['frame_000031','frame_000089'],heldout_preserved=sorted(held),
        candidate_count=len(chosen),candidate_frames=chosen,additional_frames=[f for f in chosen if f not in ids],
        candidate_bins=bins,candidate_pages=pages,current_six=str(OUT/'current-six.png'),
        historical_effective_support=support,historical_support_is_not_latest_random_background_measurement=True,
        candidate_pose_quality_verified=False,candidate_gaussian_admitted=False,
        quality_accepted=False,optimizer_updates=0,source_hashes=hashes,
        caveats=['Directions depend on an approximate fitted body; identity scene extrinsics are an explicit gauge, not repeated identical subject views.',
                 'Interpolation enlargement cannot add observed directions or new physical detail.',
                 'Existing wrist screen does not establish correct limb depth, face correspondence or camera pose.',
                 'Fitting a larger roster changes the initializer. A pure view-count Gaussian experiment needs both arms to consume the same fitted archive.',
                 'Six-view baseline already has visible fitting and masking errors; more bad views could amplify conflicts.'])
    assert all(sha(p)==h for p,h in hashes.items())
    save(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['state','input_count','current_training_count','circular_gaps','candidate_count','candidate_frames']},ensure_ascii=False))


if __name__ == '__main__': main()
