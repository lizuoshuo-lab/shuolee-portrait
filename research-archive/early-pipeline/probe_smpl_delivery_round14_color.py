"""Isolate SH0 LDR color on frozen R13 endpoints at native scale."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_smpl_delivery_round13 as r13
sys.path.insert(0,str(r13.PROJECT/'src'))

C0 = 0.28209479177387814
RECEIPT_SHA = '55ccd67daf8607f75f069a8d17ff9783327097fd56938ab19b57475bffdfbbd8'
AUDIT_SHA = {'016': '41f23cfcb6d7998a604301f1eea3b1e6ac788285a3c0635a8f675797d7fb4dd1',
             '006': '36780c7e70a66809c38b6582421e46beed49bb6277703ae67f99726711c88fa0'}
COMMON = ('static_reference_front', 'static_rotated_oblique', 'static_rotated_side')
pin, sha, read, require = r13.pin, r13.sha, r13.read, r13.require


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def rgb_metrics(rgb, alpha, source, matte, crop, brightness_box):
    maximum = rgb.max(2)
    xc0, yc0, xc1, yc1 = crop
    x0, y0, x1, y1 = brightness_box
    torso = rgb[y0:y1, x0:x1]
    torso_alpha = alpha[y0:y1, x0:x1]
    torso_luma = torso @ np.array([.2126, .7152, .0722], dtype=np.float32)
    bright = (torso_alpha >= 243/255) & (torso_luma >= 230/255)
    excess = maximum-alpha
    result = dict(
        min_rgb=float(rgb.min()), max_rgb=float(rgb.max()),
        min_rgb_by_channel=rgb.min(axis=(0,1)).tolist(), max_rgb_by_channel=rgb.max(axis=(0,1)).tolist(),
        any_rgb_above_1_pixels=int(np.count_nonzero(maximum > 1.000001)),
        max_rgb_minus_alpha=float(excess.max()),
        rgb_exceeds_alpha_by_5_levels_pixels=int(np.count_nonzero(excess > 5/255)),
        torso_bright_opaque_pixels=int(bright.sum()),
        bright_opaque_in_crop=int(np.count_nonzero(
            ((rgb[yc0:yc1, xc0:xc1] @ np.array([.2126, .7152, .0722], dtype=np.float32)) >= 230/255)
            & (alpha[yc0:yc1, xc0:xc1] >= 243/255))),
    )
    if source is not None:
        foreground = matte >= 253/255
        source_float = source.astype(np.float32)/255.
        result['foreground_rgb_mae_raw'] = float(np.abs(rgb[foreground]-source_float[foreground]).mean())
        result['foreground_rgb_mae_display'] = float(
            np.abs(rgb[foreground].clip(0,1)-source_float[foreground]).mean())
        src_luma = source_float[y0:y1,x0:x1] @ np.array([.2126,.7152,.0722], dtype=np.float32)
        result['torso_bright_opaque_dark_source_pixels'] = int((bright & (src_luma <= 204/255)).sum())
    return result


def brightness_box(crop, mask):
    if mask is None:
        x0,y0,x1,y1 = crop
    else:
        yy,xx = np.where(mask >= .5)
        x0,y0,x1,y1 = int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
    h = y1-y0
    return [x0, y0+round(.25*h), x1, y0+round(.65*h)]


def panel(path, crop, source, rgb, cap, restored):
    x0,y0,x1,y1 = crop
    images = [np.rint(np.clip(x[y0:y1,x0:x1],0,1)*255).astype(np.uint8)
              for x in (rgb, cap, 4*(rgb-cap), restored)]
    columns = ['original', 'constrained', 'excess_rgb_x4', 'restored']
    if source is not None:
        images.insert(0, source[y0:y1,x0:x1])
        columns.insert(0, 'source_rgb')
    Image.fromarray(np.concatenate(images,axis=1)).save(path)
    return dict(file=str(path), sha256=sha(path), columns=columns, crop_xyxy=crop)


def difference_metrics(original, constrained):
    delta = original-constrained
    return dict(max_abs=float(np.abs(delta).max()), mean_abs=float(np.abs(delta).mean()),
                changed_pixels_gt_one_level=int(np.count_nonzero(np.abs(delta).max(2) > 1/255)),
                pixels_gt_five_levels=int(np.count_nonzero(np.abs(delta).max(2) > 5/255)))


def load_plan(subject):
    delivery = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round13'
    receipt = read(pin(delivery/'receipt.json',RECEIPT_SHA))
    arm = receipt['subjects'][subject]
    review = pin(arm['review'],arm['review_sha256'])
    result = read(review)
    plan_path = pin(review.with_name('plan.json'),result['plan_sha256'])
    plan = read(plan_path)
    run = Path(arm['result']).parent
    source_rows = read(pin(run/'data/training-inputs.json',plan['new']['training_inputs_sha256']))['records']
    require(len(source_rows) == 24 and {x['frame'] for x in source_rows} == set(arm['selected_frame_ids']),
            'The exact R13 24-frame source roster is required.')
    terminal = read(pin(arm['result'],arm['result_sha256']))
    saved = r13.round1.delivered(run,Path(plan['asset']),source_rows,arm['retained_holdouts'])
    ply = pin(saved.snapshot/'bound-local.ply',AUDIT_SHA[subject])
    require(terminal['steps'] == 10000 and terminal['sh_degree'] == 0 and arm['points'] == terminal['points'],
            'The pinned terminal is not the audited R13 SH0 output.')
    rois = pin(plan_path.with_name('rois.npz'),plan['roi_sha256'])
    return arm, plan, plan_path, rois, source_rows, saved, ply


def run(subject):
    arm, plan, plan_path, rois_path, rows, saved, ply = load_plan(subject)
    out = r13.review.LAB/f'shuolee-smpl-delivery-round14-{subject}'/'endpoint-color-isolation'
    out.mkdir(parents=True,exist_ok=False)
    (out/'float').mkdir()
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import RAW_PARAMETERS, _load_model, inspect_bound_ply
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    asset = Path(plan['asset'])
    rt.configure(asset.parents[1])
    from gaussian_renderer import render
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    with gpu_lock(), torch.no_grad(), np.load(rois_path,allow_pickle=False) as rois:
        require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA device.')
        model, parity = _load_model(ply,saved.snapshot,saved.fit,saved.mesh,saved.facts,asset)
        black = torch.zeros(3,device='cuda')
        pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        pairs = {item['frame']:item for item in plan['pairs']}
        captures = {item['frame']:item for item in saved.replay['records']}
        effective_c0 = max(C0,float(np.float32(C0)))
        cmax = np.float32(.5/effective_c0)
        if float(cmax)*effective_c0 > .5:
            cmax = np.nextafter(cmax,np.float32(0))
        cmax = float(cmax)
        original_dc = model._features_dc.detach().clone()
        constrained_dc = original_dc.clamp(-cmax,cmax)
        rgb_points = C0*original_dc[:,0,:]+.5
        constrained_points = C0*constrained_dc[:,0,:]+.5
        require(float(constrained_points.min()) >= 0 and float(constrained_points.max()) <= 1,
                'The conservative DC bounds do not preserve the float SH0 RGB range.')
        active = model.get_opacity.reshape(-1) > .1
        color = dict(points=len(rgb_points),dc_limit=cmax,min_rgb=float(rgb_points.min()),max_rgb=float(rgb_points.max()),
            constrained_min_rgb=float(constrained_points.min()),constrained_max_rgb=float(constrained_points.max()),
            active_opacity_threshold=.1,active_points=int(active.sum()),
            points_any_above_1=int((rgb_points.max(1).values>1).sum()),
            active_points_any_above_1=int(((rgb_points.max(1).values>1)&active).sum()),
            points_any_below_0=int((rgb_points.min(1).values<0).sum()),
            active_points_any_below_0=int(((rgb_points.min(1).values<0)&active).sum()),
            changed_dc_points=int((original_dc != constrained_dc).flatten(1).any(1).sum()))
        # CUDA's original SH0 conversion clamps below zero. The override is
        # precisely the lost per-point emitted color under that conversion.
        excess = (rgb_points.clamp_min(0)-constrained_points.clamp_min(0)).clamp_min(0).contiguous()
        raw_unchanged = [name for name in RAW_PARAMETERS if name != '_features_dc']
        raw_state = {name:getattr(model,name).detach().clone() for name in raw_unchanged}
        binding = model.binding.detach().clone()

        def native_data(item):
            with Image.open(pin(item['source'],item['source_sha256'])) as im:
                source = np.asarray(im.convert('RGB'))
            with Image.open(pin(item['score_mask'],item['score_mask_sha256'])) as im:
                matte = np.asarray(im.convert('L'),dtype=np.float32)/255.
            return source,matte

        def colors(current, cam, slot, override=None):
            kwargs = {} if override is None else {'override_color':override}
            value = render_pose(render,cam,current,pipe,black,slot,**kwargs)['render']
            require(torch.isfinite(value).all(),'Nonfinite native output.')
            return value

        def diagnostic(current, row, slot, name, crop, item=None, archived=None, head=False, frame_record=None):
            cam, camera_receipt = camera(row,0,pose_index=slot)
            ones = torch.ones_like(current.get_xyz)
            current._features_dc.copy_(original_dc)
            original_t = colors(current,cam,slot)
            alpha_t = colors(current,cam,slot,ones)
            try:
                current._features_dc.copy_(constrained_dc)
                clipped_t = colors(current,cam,slot)
                clipped_alpha_t = colors(current,cam,slot,ones)
            finally:
                current._features_dc.copy_(original_dc)
            restored_t = colors(current,cam,slot)
            restored_alpha_t = colors(current,cam,slot,ones)
            require(torch.equal(original_t,restored_t) and torch.equal(alpha_t,clipped_alpha_t)
                    and torch.equal(alpha_t,restored_alpha_t) and torch.equal(alpha_t,alpha_t[:1].expand_as(alpha_t)),
                    'RGB restoration or alpha parity failed.')
            check_t = colors(current,cam,slot,excess)
            contribution_max = float((check_t-(original_t-clipped_t)).abs().max())
            require(contribution_max < 5e-5,'Removed color does not match the rendered per-point excess.')
            require(torch.equal(current._features_dc,original_dc) and torch.equal(current.binding,binding)
                    and all(torch.equal(getattr(current,key),value) for key,value in raw_state.items()),
                    'A geometry, binding, opacity or non-DC parameter changed.')
            original,clipped,restored = [x.permute(1,2,0).cpu().numpy() for x in (original_t,clipped_t,restored_t)]
            alpha = alpha_t[0].cpu().numpy()
            source,matte = native_data(item) if item else (None,None)
            archive_delta = None
            if archived:
                require(archived['pose_index'] == slot and all(
                    archived['camera'][key] == camera_receipt[key] for key in ('K','w2c','wh')),
                    'Archived source camera or pose slot changed.')
                with Image.open(pin(saved.snapshot/'posed-replay'/archived['image'],archived['image_sha256'])) as im:
                    reference = np.asarray(im.convert('RGB'),dtype=np.int16)
                archive_delta = int(np.abs(np.rint(np.clip(original,0,1)*255).astype(np.int16)-reference).max())
                require(archive_delta <= 1,'Untouched arm does not reproduce the archived R13 replay.')
            brightness = brightness_box(crop,matte)
            metrics = {label:rgb_metrics(img,alpha,source,matte,crop,brightness)
                       for label,img in (('original',original),('constrained',clipped))}
            regions = {}
            if item and item['role'] in plan['native_regions']:
                for region,roi_hash in plan['native_regions'][item['role']].items():
                    mask = rois[f"{item['role']}_{region}"]
                    require(r13.r9.native.hash_mask(mask) == roi_hash,'Locked native face ROI changed.')
                    target = source[mask].astype(np.float32)/255.
                    regions[region] = dict(pixels=int(mask.sum()),roi_sha256=roi_hash,
                        original_rgb_mae_raw=float(np.abs(original[mask]-target).mean()),
                        constrained_rgb_mae_raw=float(np.abs(clipped[mask]-target).mean()),
                        original_rgb_mae_display=float(np.abs(original[mask].clip(0,1)-target).mean()),
                        constrained_rgb_mae_display=float(np.abs(clipped[mask].clip(0,1)-target).mean()))
            geometry_metrics = None if matte is None else r13.r10.mask_metrics(alpha,matte,np.zeros_like(matte,dtype=bool))
            raw_path = out/'float'/f'{name}.npz'
            np.savez_compressed(raw_path,original_rgb=original,constrained_rgb=clipped,alpha=alpha)
            figures = [panel(out/f'{name}.png',crop,source,original,clipped,restored)]
            if head and item['head_crop_xyxy']:
                figures.append(panel(out/f'{name}_head.png',item['head_crop_xyxy'],source,original,clipped,restored))
            return dict(name=name,camera=camera_receipt,frame=row['frame'],reference_role=frame_record,
                source_sha256=item['source_sha256'] if item else None,
                score_mask_sha256=item['score_mask_sha256'] if item else None,
                crop_xyxy=crop,brightness_box_xyxy=brightness,metrics=metrics,regions=regions,
                alpha_geometry_metrics=geometry_metrics,restored_bitwise_equal=True,alpha_bitwise_equal=True,
                original_rgb_sha256=digest(original),restored_rgb_sha256=digest(restored),alpha_sha256=digest(alpha),
                archive_max_abs_8bit=archive_delta,removed_contribution_max_abs_error=contribution_max,
                change=difference_metrics(original,clipped),float_capture=dict(file=str(raw_path),sha256=sha(raw_path),
                    dtype='float32',shape=list(original.shape),restored_reuses_identical_original=True),panels=figures)

        native = []
        for row in rows:
            frame = row['frame']
            item = pairs.get(frame)
            if item is None:
                a,b,c,d = row['initial_box_xyxy']
                item = dict(role='native_unpaired',source=row['image'],source_sha256=row['source_sha256'],
                    score_mask=row['mask'],score_mask_sha256=row['mask_sha256'],
                    crop_xyxy=r13.r10.native_box((a-64,b-64,c+64,d+64),row['wh']),head_crop_xyxy=None)
            require(item['source_sha256'] == row['source_sha256'] and item['score_mask_sha256'] == row['mask_sha256'],
                    'The frozen source frame or mask changed.')
            slot = captures[frame]['pose_index']
            native.append(diagnostic(model,row,slot,f'native_{frame}',item['crop_xyxy'],item,captures[frame],head=True,
                                     frame_record='R13_selected_named_pose'))
            print(subject,frame,'native complete',flush=True)

        reference = plan['common_reference']
        with np.load(pin(reference['file'],reference['sha256']),allow_pickle=False) as mesh:
            vertices,faces = mesh['vertices_world'],mesh['faces']
            require(str(mesh['reference_frame_id'].item()) == reference['frame'],'The shared reference frame changed.')
        poses = [vertices]
        specials = plan['common_special']
        for item in specials:
            with np.load(pin(item['common_mesh_file'],item['common_mesh_file_sha256']),allow_pickle=False) as mesh:
                slot = mesh['frame_ids'].tolist().index(item['frame'])
                require(np.array_equal(faces,mesh['faces']) and mesh['pose_indices'][slot] == item['old_slot'],
                        'The controlled f051/f055 pose identity changed.')
                poses.append(mesh['vertices_world'][slot])
        ids,counts = inspect_bound_ply(ply,len(faces))
        common_model = SMPLXGaussianModel(0)
        common_model.vertices_sequence = torch.as_tensor(np.stack(poses),device='cuda')
        common_model.faces = torch.as_tensor(faces,dtype=torch.long,device='cuda')
        common_model.num_timesteps = len(poses)
        common_model.load_ply(str(ply))
        common_model.binding_counter = torch.as_tensor(counts,dtype=torch.int32,device='cuda')
        common_model.select_mesh_by_timestep(0)
        require(torch.equal(common_model.binding.long(),torch.as_tensor(ids,device='cuda'))
                and torch.equal(common_model._features_dc,original_dc) and common_model.active_sh_degree == 0,
                'The common reference model did not load the exact bound SH0 parameters.')
        common = {}
        for name in COMMON:
            view = plan['static_views'][name]
            common[name] = diagnostic(common_model,view['fixed_world_camera'],0,f'common_{name}',
                                      view['crop_xyxy'],frame_record='R10_common_reference_mesh')
        controlled = {}
        for slot,item in enumerate(specials,start=1):
            row = dict(item['camera'],frame=item['frame'])
            controlled[item['frame']] = diagnostic(common_model,row,slot,item['role'],item['crop_xyxy'],item,head=True,
                                                  frame_record=f"R10_controlled_mesh_R13_{item['R13']}")
        require(sha(ply) == AUDIT_SHA[subject],'The endpoint PLY changed.')
    all_views = native+list(common.values())+list(controlled.values())
    result = dict(state='round14_r13_endpoint_color_isolation_complete',subject=subject,round13_commit=arm['commit'],
        round13_terminal_ply=str(ply),round13_terminal_ply_sha256=sha(ply),round13_review_plan=str(plan_path),
        round13_review_plan_sha256=sha(plan_path),common_reference=reference,optimizer_updates=0,
        intervention='In-memory stock SH0 DC projection to RGB [0,1], then restoration',
        float_contract='Raw full-frame black-background raster float32 RGB, and independently rendered alpha. Restored RGB matches original bitwise.',
        common_has_native_source_truth=False,color=color,parity=parity,native=native,common=common,controlled=controlled,
        views=len(all_views),source_views=len(native),common_views=len(common),controlled_views=len(controlled),
        alpha_bitwise_equal=all(x['alpha_bitwise_equal'] for x in all_views),
        restored_bitwise_equal=all(x['restored_bitwise_equal'] for x in all_views),
        max_removed_contribution_error=max(x['removed_contribution_max_abs_error'] for x in all_views),
        script_sha256=sha(Path(__file__)))
    save_json(out/'result.json',result)
    print(subject,'complete',len(all_views),sha(out/'result.json'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subjects',nargs='+',choices=('016','006'))
    args = parser.parse_args()
    with r13.review.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for subject in args.subjects:
            run(subject)


if __name__ == '__main__':
    main()

