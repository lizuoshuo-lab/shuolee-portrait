"""Archive both checked R23 endpoints and original-pixel image-only records."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_smpl_delivery_round23 as review
import archive_smpl_delivery_round22 as prior
from archive_smpl_delivery_round1 import common, verify_scene
from archive_smpl_delivery_round2 import copy_proof


qa = review.qa
checked,pin,read,sha,require,descriptor = qa.checked,qa.pin,qa.read,qa.sha,qa.require,qa.descriptor
DELIVERY = prior.DELIVERY.with_name('smpl-delivery-round23')
TAG = f'交付23_{review.COMMIT[:7]}'
TITLE = ('Original native RGB','R22 18,000','R23 10,000','R23 18,000','Frozen SMPL-X guide','2D input / fit')
COLS = ('原图','R22 18k','R23 10k','R23 18k','冻结 Mesh 投影','2D 特征与投影')
SOFA_COMPARISON_SHA = '95d1c6b549c38af50d9bd823e5641886d67ffe7722538cf39e07b178bff2e0c7'


def paste_panel(target, parts):
    w,h = parts[0].size
    require(all(image.size==(w,h) for image in parts),'Same-pose record crops must match at native size.')
    image = Image.new('RGB',(6*w,h+26),'#181818')
    draw = ImageDraw.Draw(image)
    for i,(label,part) in enumerate(zip(TITLE,parts,strict=True)):
        draw.text((i*w+6,6),label,fill='white')
        image.paste(part,(i*w,26))
    if target.exists():
        with Image.open(target) as prior_image:
            require(prior_image.size==image.size and prior_image.convert('RGB').tobytes()==image.tobytes(),
                    'Preserve the existing R23 selected frame image.')
    else:
        with target.open('xb') as stream:
            image.save(stream,format='PNG',compress_level=1)
    return descriptor(target)


def crop_overlap(*boxes):
    out = [max(box[0] for box in boxes),max(box[1] for box in boxes),
           min(box[2] for box in boxes),min(box[3] for box in boxes)]
    require(out[2]>out[0] and out[3]>out[1],'Matched native crops have no overlap.')
    return out


def local_box(crop, source_box):
    return tuple(value-offset for value,offset in zip(crop,source_box[:2]*2,strict=True))


def crop_file(path, crop, source_box=None, col=0, header=0):
    with Image.open(path) as image:
        if source_box is None:
            return image.convert('RGB').crop(crop)
        x0,y0,x1,y1 = local_box(crop,source_box)
        width,height = source_box[2]-source_box[0],source_box[3]-source_box[1]
        require(image.width>=(col+1)*width and image.height>=height+header
                and 0<=x0<x1<=width and 0<=y0<y1<=height,'A frozen native record crop changed.')
        dx = col*width
        return image.convert('RGB').crop((dx+x0,header+y0,dx+x1,header+y1))


def drive_image(plan, result, name, slot, crop):
    view = plan['drives'][name]['records'][slot]
    row = result['native'][name][view['frame']]
    require(row['camera']['pose_index']==slot and row['role']==('selected_fit' if slot<24 else 'holdout'),
            'The matched native drive image uses a different slot.')
    return crop_file(checked(row['delivered_media']['black']),crop,view['box_xyxy'])


def current_endpoints(subject, plan):
    run = review.root_for(subject)/'run'
    result = read(checked(plan['result']))
    files = {}
    for name,record in zip(review.NEW,result['endpoints'],strict=True):
        snapshot = run/'gaussian-run'/record['path']
        proof = plan['drives'][name]
        portable = read(checked(proof['portable']))
        arm = plan['static_grid']['arms'][name]
        mesh = verify_scene(snapshot,dict(portable,reference_frame_id=arm['ref']))
        ply = pin(arm['ply'],arm['ply_sha256'])
        camera = pin(arm['camera_path'],arm['camera_sha256'])
        provenance = camera.with_name('camera_provenance.json')
        scene = checked(proof['scene'])
        drive = checked(proof['package'])
        target = DELIVERY/subject/('duration-10000' if name=='r23_10' else 'duration-18000')
        target.mkdir(parents=True,exist_ok=True)
        saved = copy_proof(target,dict(portable_ply=(ply,sha(ply)),camera=(camera,sha(camera)),
            camera_provenance=(provenance,sha(provenance)),scene=(scene,sha(scene)),drive=(drive,sha(drive))))
        def peak_motion(rows):
            keys = ('root_removed_center_motion_median_m','root_removed_center_motion_p95_m')
            peaks = {key:max(rows,key=lambda row:row[key]) for key in keys}
            return {key:dict(value=row[key],frame=row['frame']) for key,row in peaks.items()}

        files[name] = dict(saved,step=record['step'],points=record['points'],free_clothing=record['free_garment_points'],
            reference_frame=arm['ref'],reference_mesh=descriptor(mesh),native_replay=proof['replay'],
            max_root_removed_motion=peak_motion(proof['records']),
            max_selected_root_removed_motion=peak_motion([row for row in proof['records'] if row['role']=='training']))
    return files


def training_curve(out, source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    receipt = out/'training-curve.json'
    if receipt.exists():
        old = read(receipt)
        require(old['raw']==source,'The saved training curve has a different source.')
        checked(old['csv'])
        checked(old['image'])
        return old
    rows = [json.loads(line) for line in checked(source).read_text(encoding='utf-8').splitlines()]
    steps = [row['step'] for row in rows]
    fields = ('step','frame','loss','l1','ssim','mask_loss','protected_face_loss','points','elapsed_seconds')
    require(steps==[1,*range(100,18001,100)]
            and all(np.isfinite([row[k] for k in fields if k!='frame']).all() for row in rows),
            'The R23 training curve is incomplete.')
    values = out/'training-curve.csv'
    with values.open('x',encoding='utf-8',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained',dpi=120)
    for axis,field,label,color in zip(axes,('l1','mask_loss','protected_face_loss'),
            ('Foreground RGB L1','Mask loss','Face protection loss'),('#265eaf','#12745c','#9c5d10'),strict=True):
        numbers = np.asarray([row[field] for row in rows])
        median = [np.median(numbers[max(0,i-2):i+3]) for i in range(len(numbers))]
        axis.plot(steps,numbers,color=color,alpha=.30,linewidth=.7)
        axis.plot(steps,median,color=color,linewidth=1.6)
        for step in (10000,18000):
            axis.axvline(step,linestyle='--',color='#657085',linewidth=.8)
        axis.set_ylabel(label)
        axis.grid(axis='y',alpha=.20)
    axes[-1].set_xlabel('Main garment optimizer step')
    axes[-1].set_xlim(0,18100)
    figure_path = out/'training-curve.png'
    with figure_path.open('xb') as stream:
        figure.savefig(stream,format='png')
    plt.close(figure)
    saved = dict(raw=source,csv=descriptor(values),image=descriptor(figure_path),main_optimizer_steps=18000)
    prior.write_receipt(receipt,saved)
    return saved


def full_native_records(subject, out, plan, result):
    r22 = qa.launch.root_for(subject)
    old_plan = read(checked(read(checked(plan['r22_review']))['plan']))
    old_pose = old_plan['poses']['r22']
    run = r22/'run'
    old_result = read(pin(run/'result.json',old_pose['new']['terminal_sha256']))
    panel_file = pin(run/old_result['bound_panels'],old_result['bound_panels_sha256'])
    old_selected = read(panel_file)['panels']
    require([r['frame'] for r in old_selected]==old_pose['matched_frames'],'Frozen selected record order changed.')
    out.mkdir(exist_ok=True)
    records = {}
    for slot,row in enumerate(plan['frozen']['rows']):
        frame = row['frame']
        b10,b18 = (plan['drives'][name]['records'][slot]['box_xyxy'] for name in review.NEW)
        source = pin(row['image'],row['source_sha256'])
        if slot<24:
            original = old_selected[slot]
            require(original['frame']==frame and original['source_sha256']==row['source_sha256']
                    and original['pose_index']==slot,'The original R22 mesh or feature overlay used another frame.')
            same = original['crop_xyxy']
            crop = crop_overlap(same,b10,b18)
            canvas = pin(panel_file.parent/original['panel'],original['panel_sha256'])
            original_tile,old_gaussian,mesh,features = (
                crop_file(canvas,crop,same,col=i,header=26) for i in range(4))
            require(np.array_equal(np.asarray(original_tile),np.asarray(crop_file(source,crop))),
                    'The original panel pixels differ from the pinned RGB image.')
        else:
            held = old_pose['holdout']['frames'][slot-24]
            require(held['frame']==frame and held['source_sha256']==row['source_sha256'],
                    'The R22 heldout mesh or feature overlay used another source.')
            same = held['crop_xyxy']
            crop = crop_overlap(same,b10,b18)
            original_tile,old_gaussian = crop_file(source,crop),crop_file(checked(held['files']['black']),crop)
            mesh = crop_file(checked(held['files']['mesh']),crop,same)
            features = crop_file(checked(held['files']['feature']),crop,same)
        early,late = (drive_image(plan,result,name,slot,crop) for name in review.NEW)
        combined = paste_panel(out/(frame+'.png'),(original_tile,old_gaussian,early,late,mesh,features))
        records[frame] = dict(combined,frame=frame,role='selected_fit' if slot<24 else 'holdout',
                              source_sha256=row['source_sha256'],crop_xyxy=crop,columns=COLS)
    return records


def bundle_evaluation(out, result_path, result, plan, curve, native):
    files = {'plan.json':result['plan'],'review.json':descriptor(result_path),
        'fixed-cameras.json':plan['static_grid']['proof']['camera_manifest'],
        'training/curve.png':curve['image'],'training/curve.csv':curve['csv'],'training/loss.jsonl':curve['raw']}
    if plan['subject']=='006':
        files['119-006-source-comparability.json'] = descriptor(pin(
            DELIVERY/'input-audit/119-006-source-comparability.json',SOFA_COMPARISON_SHA))
    for label,item in result['panels'].items():
        name = 'native-torso' if label.startswith('native_torso_') else 'fixed-views'
        files[f'{name}/{label}.png'] = item
    for name,item in native.items():
        files[f'native-full/{name}.png'] = item
    target = out/'paired-evaluation.zip'
    if not target.exists():
        with TemporaryDirectory(prefix='.r23-eval-',dir=out) as tmp:
            stage = Path(tmp)/target.name
            with ZipFile(stage,'x') as archive:
                for name,item in files.items():
                    archive.write(checked(item),name)
            require(not target.exists(),'An R23 evidence archive already exists.')
            stage.rename(target)
    with ZipFile(target) as archive:
        require(len(archive.namelist())==len(files) and set(archive.namelist())==set(files)
                and archive.testzip() is None,'The R23 evaluation archive has invalid members.')
        for name,item in files.items():
            require(sha256(archive.read(name)).hexdigest()==item['sha256'],f'Changed R23 evidence: {name}')
    return dict(descriptor(target),count=len(files))


def numerical_proof(subject, plan):
    path = DELIVERY/'input-audit'/f'{subject}-numerical-paired.json'
    audit = read(path)
    run = review.root_for(subject)/'run'
    require(audit['schema']=='shuolee-r23-independent-numerical/1' and audit['subject']==subject
            and audit['kind']=='formal' and Path(audit['run'])==run
            and audit['baseline']['fit']==plan['frozen']['fit']
            and [item['step'] for item in audit['endpoints']]==list(review.NEW.values()),
            'The independent numerical audit refers to a different R23 run or baseline.')
    for item in audit['endpoints']:
        snapshot = run/'gaussian-run'/f"raw-step-{item['step']:06d}"
        pin(snapshot/'main-state.pt',item['checkpoint_sha256'])
        require(item['exact_frozen_noncloth_parameters'] and item['exact_frozen_pose_cameras']==28,
                'Independent audit detected a protected row, camera or pose change.')
    return descriptor(path)


def archive(subject):
    root = review.root_for(subject)/'view-comparison'
    result_path = root/'result.json'
    result = read(result_path)
    plan = read(checked(result['plan']))
    require(result['state']=='paired_driven_and_static_views_rendered' and result['static_camera_count']==50
            and result['native_poses']==28 and result['commit']==plan['commit']==review.COMMIT
            and result['subject']==plan['subject']==subject and result['quality_accepted'] is False,
            'Archive requires the independent review of both exact R23 durations.')
    out = DELIVERY/subject
    out.mkdir(parents=True,exist_ok=True)
    endpoints = current_endpoints(subject,plan)
    numerical = numerical_proof(subject,plan)
    curve = training_curve(out,plan['training_log'])
    native = full_native_records(subject,out/'native-full',plan,result)
    evaluation = bundle_evaluation(out,result_path,result,plan,curve,native)
    record_dir = common.RECORDS/subject
    require(record_dir.is_dir() and record_dir.resolve(strict=True)==record_dir,'Use the existing subject record folder.')
    saved = {}
    with TemporaryDirectory(prefix='shuolee-r23-record-') as tmp:
        todo,pins = [],{}

        def add(key,item,name):
            src = checked(item)
            pins[src] = item['sha256']
            stage,dest = Path(tmp)/name,record_dir/name
            with Image.open(src) as image:
                image.convert('RGB').save(stage,quality=95,subsampling=0)
            digest = sha(stage)
            if dest.exists():
                pin(dest,digest)
            else:
                todo.append((stage,dest))
            saved[key] = dict(file=str(dest),sha256=digest)

        for frame,item in native.items():
            if item['role']=='selected_fit':
                add(frame,item,f'3DGS_{subject}_{TAG}_原图-高斯对照-Mesh-特征点_{frame}.jpg')
            else:
                add(frame,item,f'分析_{subject}_{TAG}_留出原图-高斯对照-Mesh-特征点_{frame}.jpg')
        for key in prior.PITCH_KEYS:
            add(key,result['panels'][key],f'分析_{subject}_{TAG}_{key}.jpg')
        reference = 'native_torso_'+plan['frozen']['ref']
        if reference in result['panels']:
            add(reference,result['panels'][reference],f'分析_{subject}_{TAG}_{reference}.jpg')
        add('training_curve',curve['image'],f'分析_{subject}_{TAG}_10000-18000_主训练曲线.jpg')
        common.publish_images(todo,pins)
    receipt = dict(schema='shuolee-r23-paired-delivery/1',subject=subject,commit=review.COMMIT,quality_accepted=False,
                   review=descriptor(result_path),plan=result['plan'],endpoints=endpoints,curve=curve,records=saved,
                   native_full=native,evaluation=evaluation,numerical=numerical,selected_count=24,heldout_count=4)
    item = prior.write_receipt(out/'receipt.json',receipt)
    print(json.dumps(dict(receipt=item,record_images=len(saved),selected_images=24,heldouts=4)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=qa.prior.SUBJECTS)
    archive(parser.parse_args().subject)
