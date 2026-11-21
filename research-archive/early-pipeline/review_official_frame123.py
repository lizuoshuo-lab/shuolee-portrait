"""Read one completed official fit; CPU native-pixel evidence, no refitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from review_full_sequence123 import relocate, read, sha, mesh_renderer, crop_box, joined


def require(ok, message):
    if not ok:
        raise ValueError(message)


def statistics(values):
    values = np.asarray(values)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values,.9)), maximum=float(values.max())) if len(values) else dict(count=0)


def sequence_observations(root, directory, inputs, data, name):
    """Read the exact frozen batch observations without constructing a model."""
    from smplify_score_contract123 import relative_confidence
    from official_smplifyx_entry123 import COCO

    require(directory.resolve() == (root/'frames'/name).resolve(), 'Wrong sequence/frame directory.')
    provenance_path = root/'input-provenance.json'
    provenance = read(provenance_path)
    bindings = {relocate(p): h for p,h in provenance['input_files'].items()}
    require(inputs in bindings and sha(inputs) == bindings[inputs], 'Unbound batch inputs.')
    for path,digest in bindings.items():
        require(sha(path) == digest, 'Batch source changed: '+str(path))
    obs_paths = [p for p in bindings if p.name == 'observations.npz']
    require(len(obs_paths) == 1, 'Expected one bound observation file.')
    rows = read(inputs)['records']; ids = [r['frame'] for r in rows]
    require(name in provenance['frame_ids'] and len(set(ids)) == len(ids), 'Unknown batch frame.')
    with np.load(obs_paths[0],allow_pickle=False) as f: obs = dict(f)
    require(obs['frame_ids'].astype(str).tolist() == ids, 'Batch observation identities mismatch.')
    confidence,q90 = relative_confidence(obs['scores'],obs['boxes'],obs['flip_disagreement'])
    require(np.array_equal(q90,provenance['per_joint_score_q90']), 'Batch confidence mapping changed.')
    dest = np.r_[COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
    source = np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
    target = np.zeros((118,3),np.float32); index = ids.index(name)
    target[dest,:2] = obs['keypoints'][index,source]
    target[dest,2] = confidence[index,source]
    beta_path = relocate(provenance['shared_betas_path'])
    require(provenance['freeze_betas'] and sha(beta_path) == provenance['shared_betas_sha256'], 'Shared shape source changed.')
    with np.load(beta_path,allow_pickle=False) as f:
        require(np.array_equal(data['betas'],f[provenance['shared_betas_key']].reshape(1,10)), 'Different shared body shape.')
    return target, list(bindings)+[provenance_path]


def main(args):
    directory = relocate(args.fit_dir); output = relocate(args.records)
    receipt = directory/'saved-mesh-review.json'
    require(not receipt.exists(), 'Review already exists; preserve it.')
    result = read(directory/'result.json')
    require(result.get('state') == 'complete', 'Read completed new fit only.')
    if not args.sequence_root:
        require(read(directory/'progress.json').get('state') == 'complete', 'Single-frame fit is not complete.')
    require(result.get('gvhmr_used') is False, 'Expected new official zero/mean initialization.')
    require(result.get('coordinate_system') == 'opencv_camera_model_meters', 'Wrong saved mesh domain.')
    with np.load(directory/'winner-camera-space.npz',allow_pickle=False) as f: data = dict(f)
    name = str(data['frame_id'].item())
    require(name == result['frame_id'], 'Fit frame identity mismatch.')
    inputs = relocate(args.inputs); crops_path = relocate(args.crops)
    if args.sequence_root:
        target,bound_paths = sequence_observations(relocate(args.sequence_root),directory,inputs,data,name)
        image_label = '官方SMPLifyX完整94帧共享体型'
    else:
        target = data['observed_native_keypoints'][0]
        bound_paths = [directory/'input-provenance.json']
        image_label = '官方SMPLifyX零初值五阶段'
    records = read(inputs)['records']; found = [r for r in records if r['frame'] == name]
    require(len(found) == 1, 'Frame not uniquely bound to native inputs.')
    row = found[0]; image_path = relocate(row['image'])
    require(sha(image_path) == row['source_sha256'] == str(data['image_sha256'].item()), 'Image hash mismatch.')
    require(int(data['source_frame'].item()) == row['source_frame_index'], 'Source video frame mismatch.')
    require(np.array_equal(data['K_integer_centers'],row['K_integer_centers']), 'Native complete K mismatch.')
    rules = output.parent/'规则.md'
    require('只要图片不要文本' in rules.read_text(encoding='utf-8-sig'), 'Records rule missing.')
    with Image.open(image_path) as im:
        im.load(); raw=im.convert('RGB')
    require(list(raw.size) == row['wh'], 'Native dimensions mismatch.')
    crops = [x for x in read(crops_path) if int(x['source_frame']) == row['source_frame_index']]
    require(len(crops) <= 1, 'Duplicate source-frame-bound crops.')
    if crops:
        crops = crops[0]['regions']; crop_origin = 'Existing frozen source-frame integer crops'
    else:
        require(args.sequence_root, 'Expected existing source-frame-bound crops.')
        crops = {}
        for key,ids in [('face',np.arange(67,118)),('left_hand',np.arange(25,46)),('right_hand',np.arange(46,67))]:
            points = target[ids[target[ids,2]>0],:2]
            require(len(points)>0, 'No observed crop support: '+key)
            crops[key] = dict(crop_integer=crop_box(np.r_[points.min(0),points.max(0)],raw.size,padding=24))
        crop_origin = 'Integer crops derived from unchanged positive-confidence input observations, padding24; no resampling'
    mesh, renderer_source = mesh_renderer()
    hashes={str(p):sha(p) for p in [inputs,crops_path,image_path,rules,renderer_source,
        directory/'result.json',directory/'winner-camera-space.npz',directory/'official-result.pkl',*bound_paths]}
    vertices=data['vertices_camera']; faces=data['faces']; joints=data['joints_camera'][0]
    require(vertices.ndim == 3 and vertices.shape[0] == 1 and np.isfinite(vertices).all(), 'Invalid camera vertices.')
    require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in 'iu' and faces.min()>=0 and faces.max()<vertices.shape[1], 'Invalid faces.')
    require(np.isfinite(joints).all() and (joints[:,2]>0).all(), 'Invalid projected joints.')
    homogeneous = joints.astype(np.float64) @ data['K_integer_centers'].T
    uv = homogeneous[:,:2]/homogeneous[:,2:]
    errors=np.linalg.norm(uv-target[:,:2],axis=1)
    positive=target[:,2]>0; active=positive.copy(); active[[1,9,12]]=False
    groups=dict(body17=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]),
        feet6=np.arange(19,25),left_hand21=np.arange(25,46),right_hand21=np.arange(46,67),face51=np.arange(67,118))
    group_errors={key:statistics(errors[ids[active[ids]]]) for key,ids in groups.items()}
    overlay=mesh(raw,np.array([0,0,raw.width,raw.height]),data)
    boxes=[('smpl-x','完整视野',[0,0,*raw.size]),
           ('smpl-x','全身',crop_box(row['initial_box_xyxy'],raw.size,padding=64))]
    for key,label in [('face','面部'),('left_hand','左手'),('right_hand','右手')]:
        require(key in crops, 'Missing frozen crop: '+key)
        boxes.append(('分析',label,crop_box(crops[key]['crop_integer'],raw.size)))
    ids=np.r_[19:25,11,14]; ids=ids[positive[ids]]
    require(len(ids)>0,'No observed foot crop support.')
    points=target[ids,:2]
    boxes.append(('分析','双脚',crop_box(np.r_[points.min(0),points.max(0)],raw.size,padding=32)))
    files=[]
    def emit(folder,label,box,panels,description):
        sheet=joined(panels)
        path=output/folder/f'20261030_{image_label}_{name}_视频帧{row["source_frame_index"]:06d}_{label}_从左到右_{description}.png'
        require(not path.exists(),'Preserve previous image: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
        with Image.open(path) as check:
            check.load();require(np.array_equal(np.asarray(check)[:,:panels[0].width],np.asarray(panels[0])),'Original panel changed.')
        files.append(dict(path=str(path),sha256=sha(path),crop_integer_xyxy=box,wh=list(sheet.size),panels=description))
    for folder,label,box in boxes:
        emit(folder,label,box,[raw.crop(box),overlay.crop(box)],'原图_官方SMPLX透明覆盖')
    observed=raw.copy(); projected=raw.copy()
    for panel,xy,color in [(observed,target[:,:2],(0,230,70)),(projected,uv,(235,30,210))]:
        draw=ImageDraw.Draw(panel)
        for p in xy[active]:draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
    box=boxes[1][2]
    emit('特征点','全身',box,[raw.crop(box),observed.crop(box),projected.crop(box)],'原图_二维训练观测绿_三维SMPLX投影紫')
    for p,h in hashes.items():require(sha(p)==h,'Read-only source changed: '+p)
    report=dict(created=datetime.now(timezone.utc).isoformat(),state='CPU_images_generated_visual_review_pending',
        frame_id=name,source_frame=row['source_frame_index'],source_hashes=hashes,script_sha256=sha(__file__),files=files,
        projection_saved_vs_sourceK_max_px=float(np.linalg.norm(uv-data['joints_projected_native'][0],axis=1).max()),
        errors_px=dict(all_positive_observations=statistics(errors[positive]),active_fit_observations=statistics(errors[active]),
            active_confidence_above_025=statistics(errors[active&(target[:,2]>.25)]),groups=group_errors),
        per_joint=[dict(index=i,confidence=float(target[i,2]),active_fit_observation=bool(active[i]),error_px=float(errors[i])) for i in range(118)],
        observation_scope='Residuals on fitted detector observations, not independent physical ground truth; no residual pruning.',
        crop_origin=crop_origin,sequence_root=str(relocate(args.sequence_root)) if args.sequence_root else None,
        image_text_drawn=False,resampling=False,image_warp=False,original_panels_pixel_identical=True,
        GPU_calls=0,model_forwards=0,fit_updates=0,
        renderer='Existing depth-sorted CPU transparent triangle overlay; not exact z-buffer or Gaussian rendering.',
        physical_calibration_accepted=False,real_quality_accepted=False,visual_review_pending=True)
    receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=str(receipt),errors_px=report['errors_px'],images=[x['path'] for x in files]),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fit-dir',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--crops',required=True);p.add_argument('--records',default='D:/00/记录/005')
    p.add_argument('--sequence-root',help='Read a completed named frame from the ongoing official batch; never modifies that batch.')
    main(p.parse_args())
