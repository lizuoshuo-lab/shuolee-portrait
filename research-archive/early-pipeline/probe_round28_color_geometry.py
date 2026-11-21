"""Factor the exact 016 R22 to R27 rendered change into SH0 and geometry swaps."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from compare_round26_fixed_face_rays import LAB, LUMA, descriptor, load_endpoint, world_pose_in_viewer
from trace_round22_face_pixels import DELIVERY, PICKS, REVIEW_SHA, VIEW, checked, code, load_subject, need, pin, read, sha


ROUND27=DELIVERY.with_name('smpl-delivery-round27')
OUT=DELIVERY.with_name('smpl-delivery-round28')/'input-audit'/'016-color-geometry-swap'
PROJECT=Path('/mnt/d/Projects/Shuolee-smpl-face-surface-r27')
LEASE=Path('/tmp/shuolee-native122-gpu.lock')
SIGNED_RAYS_SHA='42f610c0b2876301c4c617a2eb89cd28ec04ebf9b2684355bcb930bd0a9f0bb2'
PIPE=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
ARMS=('r22_geometry_r22_color','r22_geometry_r27_color','r27_geometry_r22_color','r27_geometry_r27_color')


def png(path,mode='RGB'):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode)).copy()


def crop_pixels(tensor,box):
    x0,y0,x1,y1=box
    return tensor[:,y0:y1,x0:x1].permute(1,2,0).contiguous().cpu().numpy()


def render_swaps(models, colors, transforms, world, crop):
    from gaussian_renderer import render
    from shuolee_portrait.training import camera

    black=torch.zeros(3,device='cuda')
    images,alpha={},{}
    for geometry,model in models.items():
        row=world_pose_in_viewer(world,transforms[geometry])
        cam,actual=camera(row,0,pose_index=0)
        need(all(actual[key]==row[key] for key in ('K','wh','w2c')),'The fixed camera changed in the stock renderer.')
        alpha[geometry]=crop_pixels(render(cam,model,PIPE,black,override_color=torch.ones_like(model.get_xyz))['render'],crop)[:,:,0]
        for material in models:
            model._features_dc.copy_(colors[(geometry,material)])
            pixels=render(cam,model,PIPE,black)['render']
            need(bool(torch.isfinite(pixels).all()),'Nonfinite output in a color/geometry swap.')
            images[f'{geometry}_geometry_{material}_color']=crop_pixels(pixels,crop)
        model._features_dc.copy_(colors[(geometry,geometry)])
    return images,alpha


def parity(actual,expected):
    delta=np.abs(code(actual)-expected.astype(np.int16))
    return dict(max_code=int(delta.max()),changed_pixels=int(np.count_nonzero(np.max(delta,axis=-1) if delta.ndim==3 else delta)),
                mean_abs_code=float(delta.mean()),pixels=int(np.prod(delta.shape[:2])))


def board(path, images, title):
    width,height=images[0][1].shape[1],images[0][1].shape[0]
    result=Image.new('RGB',(width*len(images),height+30),'#181818')
    draw=ImageDraw.Draw(result)
    for i,(label,image) in enumerate(images):
        x=i*width
        result.paste(Image.fromarray(image.astype(np.uint8)),(x,30))
        draw.text((x+5,7),label,fill='#f1f1f1')
    result.save(path,compress_level=2)
    return dict(descriptor(path),title=title)


def effects(images,points):
    signed=[]
    for artifact,control in points:
        ax,ay=artifact; cx,cy=control
        colors={key:dict(artifact=code(rgb[ay,ax]).tolist(),control=code(rgb[cy,cx]).tolist()) for key,rgb in images.items()}
        contrast={key:float(LUMA@(np.asarray(value['artifact'])-np.asarray(value['control']))) for key,value in colors.items()}
        y00,y01,y10,y11=[contrast[name] for name in ARMS]
        signed.append(dict(artifact_xy=artifact,control_xy=control,rgb_codes=colors,contrast_luma_codes=contrast,
            color_with_r22_geometry=y01-y00,color_with_r27_geometry=y11-y10,
            geometry_with_r22_color=y10-y00,geometry_with_r27_color=y11-y01,
            equal_weight_color_effect=.5*((y01-y00)+(y11-y10)),
            equal_weight_geometry_effect=.5*((y10-y00)+(y11-y01)),interaction=y11-y10-y01+y00))
    return signed


def run(source_control):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    signed=read(pin(ROUND27/'ray-comparison/016-18000-signed.json',SIGNED_RAYS_SHA))
    context=load_subject('016',read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    review_path=checked(signed['qa']['result']); review=read(review_path)
    plan_path=checked(review['plan']); plan=read(plan_path)
    qa=plan['static_grid']; roster=qa['camera_roster']
    key=VIEW+'p30'; static=roster[key]
    crop=static['head_crop_xyxy']; width=crop[2]-crop[0]; height=crop[3]-crop[1]
    arms={name:qa['arms'][name] for name in ('r22','r27')}
    ref=context['plan']['poses']['r22']['new']['reference_frame_id']
    candidate=load_endpoint(LAB/'shuolee-smpl-delivery-round27-016',18000,ref)
    need(qa['camera_roster']==context['plan']['static_grid']['camera_roster'] and
         arms['r22']['ply_sha256']==context['hashes']['portable'] and
         arms['r27']['ply_sha256']==candidate['hashes']['portable_ply']['sha256'], 'The diagnostic endpoints differ from the signed replay.')
    ids={'r22':context['stable'],'r27':candidate['stable']}
    need(len(ids['r22'])==len(ids['r27']) and np.array_equal(np.sort(ids['r22']),np.sort(ids['r27'])),
         '016 no longer has one-to-one stable lineage across the two endpoints.')
    order={}
    for destination,stable in ids.items():
        for origin,source_ids in ids.items():
            lookup={int(i):n for n,i in enumerate(source_ids)}
            order[(destination,origin)]=np.asarray([lookup[int(i)] for i in stable],np.int64)
    frozen_by_id=np.array_equal(context['local']['binding_0'],candidate['local']['binding_0'][order[('r22','r27')]])
    opacity_by_id=np.array_equal(context['local']['opacity'],candidate['local']['opacity'][order[('r22','r27')]])
    need(frozen_by_id and opacity_by_id,'016 owner triangles or raw opacity changed across the two geometry arms.')
    transforms={name:arm['world_to_viewer'] for name,arm in arms.items()}
    input_paths={name:pin(arm['ply'],arm['ply_sha256']) for name,arm in arms.items()}
    with LEASE.open('a+') as lease,torch.no_grad():
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from shuolee_portrait import runtime as rt
        rt.configure(Path(plan['asset']).parents[1])
        from scene.gaussian_model import GaussianModel
        need(torch.cuda.device_count()==1,'Use exactly one GPU for the swap.')
        models={name:GaussianModel(0) for name in arms}
        for name,model in models.items():
            model.load_ply(str(input_paths[name]))
            need(model.optimizer is None and model.binding is None and model.active_sh_degree==0,
                 'The swap requires an actual static SH0 PLY and no optimizer.')
        dc={name:model._features_dc.detach().clone() for name,model in models.items()}
        colors={(dst,src):dc[src][torch.as_tensor(index,device='cuda')] for (dst,src),index in order.items()}
        images,alpha=render_swaps(models,colors,transforms,static['world_camera'],crop)
        control=None
        if source_control:
            row=next(item for item in plan['frozen']['rows'] if item['frame']==ref)
            need(row['role']=='selected_fit','The reference source control is not an original selected observation.')
            native={name:review['native'][name][ref]['regions']['face'] for name in arms}
            box=native['r22']['box']
            need(box==native['r27']['box'],'The signed native face crop changed.')
            rgb=png(checked(row['image']))[box[1]:box[3],box[0]:box[2]]
            matte=png(checked(row['mask']),'L')[box[1]:box[3],box[0]:box[2]]>=252
            native_rgb,native_alpha=render_swaps(models,colors,transforms,row,box)
            native_parity={}
            h,w=rgb.shape[:2]
            for name,meta in native.items():
                panel=png(checked(meta['panel']))
                need(panel.shape==(h,4*w,3) and np.array_equal(panel[:,:w],rgb),'Signed original native crop or layout differs.')
                native_parity[name]=dict(black=parity(native_rgb[f'{name}_geometry_{name}_color'],panel[:,w:2*w]),
                                         alpha=parity(native_alpha[name],panel[:,3*w:4*w,0]))
            need(all(entry['max_code']<=1 for proof in native_parity.values() for entry in proof.values()),
                 'The source control PLY replay differs from its archived native render.')
            metrics={name:dict(foreground_mae_rgb_codes=float(np.abs(code(value)-rgb.astype(np.int16))[matte].mean()),
                               foreground_pixels=int(matte.sum())) for name,value in native_rgb.items()}
            control=dict(frame=ref,box=box,source=row['image'],mask=row['mask'],parity=native_parity,metrics=metrics)
    parity_receipt={}
    for name in arms:
        item=review['static'][name]['records'][key]
        image=png(checked(item['panel']))
        need(item['fixed_world_camera']==static['world_camera'] and item['panel']['crop_xyxy']==crop
             and image.shape==(height,4*width,3),'Signed static view or layout differs.')
        proof=dict(black=parity(images[f'{name}_geometry_{name}_color'],image[:,:width]),
                   alpha=parity(alpha[name],image[:,2*width:3*width,0]))
        need(max(entry['max_code'] for entry in proof.values())<=1,'Endpoint swap rerender differs from the signed GPU result.')
        parity_receipt[name]=proof
    OUT.mkdir(parents=True,exist_ok=False)
    figure=board(OUT/'fixed-plus30.png',[(key.replace('_geometry_',' G / ').replace('_color',' C').upper(),code(images[key]))
        for key in ARMS],'Registered R22/R27 +30° color/geometry swap')
    if control is not None:
        control['board']=board(OUT/'original-reference.png',[('Original RGB',rgb)]+[(key.replace('_geometry_',' G / ').replace('_color',' C').upper(),code(native_rgb[key]))
            for key in ARMS],'Original camera reference frame and paired swaps')
    receipt=dict(schema='shuolee-r28-color-geometry-swap/1',subject='016',step=18000,
        registered_view=key,world_camera=static['world_camera'],crop_xyxy=crop,point_count=len(ids['r22']),
        stable_ids_one_to_one=True,exact_shared_owner_triangles=frozen_by_id,exact_raw_opacity_by_id=opacity_by_id,
        endpoint_parity=parity_receipt,figure=figure,native_control=control,
        sampled_effects=effects(images,PICKS['016']['p30'][1]),optimizer_updates=0,
        provenance=dict(signed_rays_sha256=SIGNED_RAYS_SHA,review=descriptor(review_path),plan=descriptor(plan_path),
                        r22=descriptor(input_paths['r22']),r27=descriptor(input_paths['r27']),source_sha256=sha(__file__)),
        scope='Registered synthetic marked rays. Each contrast is artifact minus a nearby control. Swaps keep model pose, camera, opacity and stable row alignment fixed.')
    target=OUT/'result.json'
    with target.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(dict(file=str(target),sha256=sha(target),parity=parity_receipt,native=control,
               effects=[item['contrast_luma_codes'] for item in receipt['sampled_effects']]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-control',action='store_true')
    run(parser.parse_args().source_control)
