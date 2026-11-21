"""Archive exact R22 scene/camera pairs and native image-only records."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round22 as qa
from archive_smpl_delivery_round1 import common, verify_scene
from archive_smpl_delivery_round2 import copy_proof
from archive_smpl_delivery_round18 import bundle
from archive_smpl_delivery_round21 import five_panel, SELECTED_COLUMNS


checked, pin, read, sha, require, descriptor = qa.checked, qa.pin, qa.read, qa.sha, qa.require, qa.descriptor
DELIVERY = qa.launch.DELIVERY
TAG = f'交付22_{qa.COMMIT[:7]}'
PITCH_KEYS = ('close_yaw_p00_pitch_p30', 'close_yaw_p00_pitch_p00', 'close_yaw_p00_pitch_m30',
              'close_yaw_m90_pitch_p00', 'close_yaw_p90_pitch_p00', 'normal_yaw_p00_pitch_p00',
              'normal_yaw_m90_pitch_p00', 'normal_yaw_p90_pitch_p00')


def write_receipt(path, receipt):
    if path.exists():
        require(read(path) == json.loads(json.dumps(receipt, allow_nan=False)), f'Preserve archived receipt {path}.')
    else:
        qa.review.save(path, receipt)
    return descriptor(path)


def branch(out, plan, name):
    steps = qa.NEW[name]
    pose = plan['poses'][name]
    run = Path(pose['new']['run'])
    terminal_path = pin(run/'result.json', pose['new']['terminal_sha256'])
    terminal = read(terminal_path)
    require(terminal['steps'] == steps and terminal['format_passed'], 'R22 export branch is incomplete.')
    snapshot, replay_path, _ = qa.prior.round1.delivery_paths(run, terminal)
    mesh = verify_scene(run, terminal)
    packages = bundle(run, snapshot, terminal, out, pose['proof'])
    static = plan['static_grid']['arms'][name]
    ply = pin(static['ply'], static['ply_sha256'])
    camera = pin(static['camera_path'], static['camera_sha256'])
    provenance = camera.with_name('camera_provenance.json')
    files = copy_proof(out, dict(portable_ply=(ply,sha(ply)),portable_camera=(camera,sha(camera)),
                                 camera_provenance=(provenance,sha(provenance))))
    panel_path = pin(run/terminal['bound_panels'],terminal['bound_panels_sha256'])
    panels,replay = read(panel_path),read(replay_path)
    roster = pose['matched_frames']
    require(panels['same_frame_bound_gaussians'] and panels['native_crops']
            and [row['frame'] for row in panels['panels']] == [row['frame'] for row in replay['records']] == roster,
            'The R22 selected native replay does not cover every selected frame.')
    receipt = dict(steps=steps,points=terminal['points'],reference_frame=terminal['reference_frame_id'],
                   scene=packages,**files,bound_replay=descriptor(replay_path),terminal=descriptor(terminal_path),
                   selected_native_panels=descriptor(panel_path),reference_mesh=descriptor(mesh))
    return receipt, panels, replay


def evaluation(out, result, review, plan):
    """Keep every matched diagnostic image and fixed camera descriptor together."""
    target = out/'paired-evaluation.zip'
    files = {'review.json':descriptor(result),'plan.json':review['plan'],
             'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest'],
             'fixed-head-proxies.json':plan['static_grid']['proof']['proxy_manifest'],
             'fixed-head-proxies.npz':plan['static_grid']['proof']['proxy_masks'],
             'training-curve.csv':review['training_curve']['csv'],
             'main-training-loss.jsonl':review['training_curve']['raw']}
    for name,item in review['panels'].items():
        files[f'{item["scope"]}/{name}.png'] = item
    if not target.exists():
        with TemporaryDirectory(prefix='.r22-evaluation-',dir=out) as temp:
            stage = Path(temp)/target.name
            with ZipFile(stage,'x') as package:
                for name,item in files.items():
                    package.write(checked(item),name)
            require(not target.exists(),'An R22 evaluation archive already exists.')
            stage.rename(target)
    with ZipFile(target) as package:
        require(len(package.namelist()) == len(files) and set(package.namelist()) == set(files)
                and package.testzip() is None,'The R22 evaluation archive has invalid members.')
        for name,item in files.items():
            require(sha256(package.read(name)).hexdigest() == item['sha256'],f'Changed evaluation image: {name}')
    return dict(descriptor(target),panel_count=len(review['panels']),static_views=50)


def collect(subject):
    root = qa.launch.root_for(subject)
    results = root/'view-comparison/result.json'
    review = read(results)
    plan = read(checked(review['plan']))
    require(review['state'] == 'paired_native_and_static_views_rendered' and review['commit'] == plan['commit'] == qa.COMMIT
            and review['subject'] == plan['subject'] == subject and review['static_camera_count'] == 50
            and review['frozen_fit_reused'] and not review['quality_accepted'],
            'Archive requires both independently reviewed exact R22 endpoints.')
    out = DELIVERY/subject
    out.mkdir(parents=True, exist_ok=True)
    early, early_panels, early_replay = branch(out/'duration-10000',plan,'r22_10')
    late, late_panels, late_replay = branch(out,plan,'r22')
    paired_evaluation = evaluation(out,results,review,plan)
    pose = plan['poses']['r22']
    run = Path(pose['new']['run'])
    rows = read(pin(run/'data/training-inputs.json',pose['new']['training_inputs_sha256']))['records']
    selection = read(run/'fit/selection.json')
    require(early_panels['mesh_sha256'] == late_panels['mesh_sha256']
            and len(rows) == len(selection['selected_frames']) == len(set(selection['selected_frames'])) == 24
            and set(pose['matched_frames']) == set(selection['selected_frames'])
            and set(selection['selected_frames']).isdisjoint(pose['new']['protected_holdouts']),
            'The R22 durations do not share one frozen clothing/pose input.')
    record_dir = common.RECORDS/subject
    require(record_dir.is_dir() and record_dir.resolve(strict=True) == record_dir, 'Use the existing subject record folder.')
    images, panels = {}, {}
    selected = out/'selected-views'
    selected.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='shuolee-r22-record-') as scratch:
        pending, pins = [], {}

        def image_only(item, name):
            source = checked(item)
            pins[source] = item['sha256']
            tmp, target = Path(scratch)/name, record_dir/name
            with Image.open(source) as image:
                image.convert('RGB').save(tmp,quality=95,subsampling=0)
            digest = sha(tmp)
            if target.exists():
                pin(target,digest)
            else:
                pending.append((tmp,target))
            return dict(file=str(target),sha256=digest)

        for row, a, b, ar, br in zip(rows, early_panels['panels'],late_panels['panels'],
                                    early_replay['records'],late_replay['records'],strict=True):
            frame = row['frame']
            require(frame == a['frame'] == b['frame'] and a['panel'] == b['panel'] == frame+'.png'
                    and row['source_sha256'] == a['source_sha256'] == b['source_sha256']
                    and row['source_frame_index'] == a['source_frame_index'] == b['source_frame_index']
                    and a['crop_xyxy'] == b['crop_xyxy'] and a['pose_index'] == b['pose_index']
                    and ar['camera'] == br['camera'] and ar['image_sha256'] == a['gaussian_sha256']
                    and br['image_sha256'] == b['gaussian_sha256'], 'A selected native paired frame changed.')
            paths = [pin(Path(arm['selected_native_panels']['file']).parent/item['panel'],item['panel_sha256'])
                     for arm,item in ((early,a),(late,b))]
            merged = selected/(frame+'.png')
            five_panel(merged,*paths,b['crop_xyxy'])
            proof = descriptor(merged)
            jpg = image_only(proof,f'3DGS_{subject}_{TAG}_原图-高斯10k-高斯18k-Mesh-特征点_{frame}.jpg')
            panels[frame] = dict(**proof,record=jpg,columns=SELECTED_COLUMNS,source_frame_index=row['source_frame_index'],
                                 crop_xyxy=b['crop_xyxy'],source_sha256=row['source_sha256'])
        for name, panel in review['panels'].items():
            if name in PITCH_KEYS or panel['scope'] in ('original_pose','heldout_native_face','local_split','training_curve'):
                images[name] = image_only(panel,f'分析_{subject}_{TAG}_{name}.jpg')
        common.publish_images(pending,pins)
    proof = dict(review=descriptor(results),plan=review['plan'],wrapper=plan['proof']['wrapper'])
    surface = DELIVERY/'input-audit'/f'skin-distribution-{subject}.json'
    if surface.is_file():
        proof['skin_distribution'] = descriptor(surface)
    receipt = dict(schema='shuolee-round22-paired-delivery/1',subject=subject,commit=qa.COMMIT,proof=proof,
                   selected_frames=selection['selected_frames'],protected_holdouts=pose['new']['protected_holdouts'],
                   selected_panels={frame:panels[frame] for frame in selection['selected_frames']},
                   analysis_images=images,primary=late,secondary=early,paired_evaluation=paired_evaluation,
                   quality_accepted=False)
    record = write_receipt(out/'receipt.json',receipt)
    print(json.dumps(dict(receipt=record,selected_images=len(panels),analysis_images=len(images))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    collect(parser.parse_args().subject)
