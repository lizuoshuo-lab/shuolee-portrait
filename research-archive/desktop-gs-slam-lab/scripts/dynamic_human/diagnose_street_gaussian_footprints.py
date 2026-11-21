"""Read-only target Gaussian footprint and compositing intervention. No training."""
from pathlib import Path
import json
import os
import shutil
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import train_street_deformable as helper
from resume_street_deformable import cuda_tree, equal_tree, sha
from models.gaussians.basics import dataclass_gs
from models.trainers.base import rasterization

LAB=helper.LAB
OUT=LAB/'outputs/dynamic-human-street-footprint-diagnostic-017/linear-compositing-v2'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
ORIGINAL=LAB/'outputs/dynamic-human-street-smpl-joint-011/articulated'
BASELINE=LAB/'outputs/dynamic-human-street-joint-009/resume-10000-expandable'
VIEWS=[(129,4),(130,4),(131,4),(149,6),(150,6),(151,6)]


def status(value,**fields):
    helper.write_json(OUT/'state.json',dict(status=value,pid=os.getpid(),**fields))
    print(json.dumps(dict(status=value,**fields)),flush=True)


def stats(value):
    value=value.detach().float().flatten().cpu()
    assert len(value) and torch.isfinite(value).all()
    return dict(count=len(value),min=float(value.min()),median=float(value.median()),
        p90=float(torch.quantile(value,.9)),p95=float(torch.quantile(value,.95)),
        p99=float(torch.quantile(value,.99)),max=float(value.max()),mean=float(value.mean()))


def panel(directory,stem,box,keys,labels,name,columns):
    crops=[Image.open(directory/(stem+'-'+key+'.png')).convert('RGB').crop(box) for key in keys]
    scale=min(3.,260/crops[0].width,620/crops[0].height)
    size=tuple(round(x*scale) for x in crops[0].size)
    rows=(len(keys)+columns-1)//columns
    canvas=Image.new('RGB',(columns*280,rows*(size[1]+44)+40),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17)
    for i,(crop,label) in enumerate(zip(crops,labels)):
        x,y=i%columns*280,i//columns*(size[1]+44)
        draw.text((x+8,y+8),label,font=font,fill='#182d39')
        canvas.paste(crop.resize(size,Image.Resampling.NEAREST),(x+(280-size[0])//2,y+36))
    draw.text((8,canvas.height-31),stem+' · 同区域；未训练的诊断干预',font=font,fill='#182d39')
    canvas.save(directory/(stem+'-'+name+'.png'))


def main():
    assert not (OUT/'protocol.json').exists(),'Preserve earlier attempts'
    assert json.loads((SOURCE/'independent-state-audit.json').read_text())['heldout_poses_and_other_people_exactly_unchanged']
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    checkpoint=SOURCE/'checkpoint-refined.pt'
    helper.write_json(OUT/'protocol.json',dict(status='sealed_before_read_only_render_interventions',
        checkpoint_sha256=sha(checkpoint),script_sha256=sha(Path(__file__)),source='Pose diagnostic016, best in-bounds training objective',
        views=[dict(frame=f,camera=c,heldout=f%10==0) for f,c in VIEWS],person=79,optimized_steps=0,
        interventions=['Full unchanged scene','Target opacity set to zero','Target visible alpha with all occluder opacities retained',
            'Target premultiplied RGB contribution with all occluders retained','Complement RGB contribution including sky',
            'Target Gaussian scales multiplied by0.5 and0.25 with all other attributes fixed'],
        previous_attempt='Parent attempt preserved: additive check on already-clipped contributions failed at0.00580013. This attempt reproduces pre-clamp linear compositing and verifies the original clamp-before-sky order.',
        boundaries=['Scale changes also change optical coverage; they are untrained diagnostic interventions, not improved model candidates.',
            'Projected bounding radii do not measure true silhouette or final visible contribution.',
            'No human segmentation ground truth; fixed boxes include occluders and background.',
            'Existing heldout views are exploratory diagnostics, not an untouched test benchmark. No new training or checkpoint.']))
    started=time.perf_counter()
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    status('loading_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(BASELINE/'config.yaml').data)
    cfg=OmegaConf.load(SOURCE/'config.yaml')
    trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
    equal_tree(saved,trainer.state_dict(only_model=True),'before')
    assert 'Affine' not in trainer.models and 'CamPose' not in trainer.models
    trainer.completed_updates=30000;trainer.render_each_class=False;trainer.set_eval()
    node=trainer.models['SMPLNodes']
    initialization=json.loads((ORIGINAL/'initialization.json').read_text())
    model_index=next(r['model_index'] for r in initialization['instances']['SMPLNodes'] if r['author_instance']==79)
    point_mask=node.point_ids[:,0]==model_index
    assert int(point_mask.sum())==6890
    canonical,_=node.template.get_init_vf()
    model_stats=dict(scales_m=stats(node._scales[point_mask].exp()),
        largest_axis_scale_m=stats(node._scales[point_mask].exp().max(-1).values),
        opacity=stats(node._opacities[point_mask].sigmoid()),
        canonical_center_offset_m=stats((node._means[point_mask]-canonical[model_index]).norm(dim=-1)))
    helper.write_json(OUT/'model-statistics.json',model_stats)
    regions=json.loads((SOURCE/'training-rois.json').read_text())['regions']+[
        r for r in json.loads((SOURCE/'person-rois.json').read_text())['regions'] if r['instance']==79]
    lookup={(r['frame'],r['camera']):r for r in regions}
    capture={};original_render=trainer.render_gaussians
    def capture_render(gs,cam,**kwargs):
        capture.update(gs=gs,cam=cam,kwargs=kwargs)
        return original_render(gs,cam,**kwargs)
    trainer.render_gaussians=capture_render
    directory=OUT/'renders';directory.mkdir()
    rows=[]
    with torch.no_grad():
        for frame,camera in VIEWS:
            status('rendering_interventions',frame=frame,camera=camera)
            stem=f'frame{frame:03d}-camera{camera}'
            info,cam=dataset.full_image_set.get_image(frame*7+camera,1.)
            info,cam=helper.to_cuda(info),helper.to_cuda(cam)
            result=trainer(info,cam,novel_view=True)
            gs=capture['gs'];processed=capture['cam'];kwargs=capture['kwargs']
            radii=trainer.info['radii'].clone()
            labels=trainer.pts_labels
            node_positions=torch.where(labels==trainer.gaussian_classes['SMPLNodes'])[0]
            assert node.filter_mask.all() and len(node_positions)==len(point_mask)
            mask=torch.zeros_like(labels,dtype=torch.bool);mask[node_positions]=point_mask
            assert int(mask.sum())==6890
            roi=lookup[(frame,camera)];box=roi['box'];x0,y0,x1,y1=box
            raw=result['rgb'].clamp(0,1)
            expected_path=SOURCE/('evaluation/final' if frame%10==0 else 'training-evaluation/after')/(stem+'-prediction.png')
            expected=np.asarray(Image.open(expected_path))
            actual=np.uint8(raw.cpu().numpy()*255+.5)
            difference=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
            assert difference.max()<=1,stem
            images=dict(observed=info['pixels'],full=raw)
            def render(colors=None,opacities=None,scales=None,sky=True,linear=False):
                if linear:
                    rendered,alphas,_=rasterization(means=gs.means,quats=gs.quats,scales=gs.scales,
                        opacities=gs.opacities.squeeze(),colors=colors,
                        viewmats=torch.linalg.inv(processed.camtoworlds)[None],Ks=processed.Ks[None],
                        width=processed.W,height=processed.H,packed=trainer.render_cfg.packed,
                        absgrad=trainer.render_cfg.absgrad,sparse_grad=trainer.render_cfg.sparse_grad,
                        rasterize_mode='antialiased' if trainer.render_cfg.antialiased else 'classic',**kwargs)
                    return rendered[0,:,:,:3]
                changed=dataclass_gs(_means=gs.means,_quats=gs.quats,
                    _scales=gs.scales if scales is None else scales,
                    _rgbs=gs.rgbs if colors is None else colors,
                    _opacities=gs.opacities if opacities is None else opacities,detach_keys=[],extras=None)
                output,_=original_render(changed,processed,**kwargs)
                return output['rgb_gaussians']+(result['rgb_sky']*(1-output['opacity']) if sky else 0)
            target_colors=gs.rgbs.clone();target_colors[~mask]=0
            other_colors=gs.rgbs.clone();other_colors[mask]=0
            alpha_colors=torch.zeros_like(gs.rgbs);alpha_colors[mask]=1
            target_linear=render(colors=target_colors,sky=False,linear=True)
            other_linear=render(colors=other_colors,sky=False,linear=True)
            sky_linear=result['rgb_sky']*(1-result['opacity'])
            images['target-contribution']=target_linear
            images['other-contribution']=other_linear+sky_linear
            images['visible-alpha']=render(colors=alpha_colors,sky=False)
            reconstructed=((target_linear+other_linear).clamp(max=1)+sky_linear).clamp(0,1)
            decomposition_error=float((reconstructed-raw).abs().max())
            assert decomposition_error<2e-5,decomposition_error
            np.savez_compressed(directory/(stem+'-linear-components.npz'),
                target=target_linear.cpu().numpy(),other=other_linear.cpu().numpy(),
                sky=sky_linear.cpu().numpy(),full=raw.cpu().numpy())
            opacity=gs.opacities.clone();opacity[mask]=0
            images['without-target']=render(opacities=opacity)
            for factor,label in [(.5,'half-scale'),(.25,'quarter-scale')]:
                scales=gs.scales.clone();scales[mask]*=factor
                images[label]=render(scales=scales)
            for key,value in images.items():helper.save_rgb(directory/(stem+'-'+key+'.png'),value.clamp(0,1).cpu().numpy())
            target_radii=radii[0,mask];positive=target_radii[target_radii>0]
            row=dict(frame=frame,camera=camera,heldout=frame%10==0,box=box,
                target_count=6890,projected_nonzero_radius_count=int((target_radii>0).sum()),
                projected_radius_px=stats(positive),radii_over20px=int((positive>20).sum()),
                radii_over50px=int((positive>50).sum()),reload_max8bit_difference=int(difference.max()),
                contribution_sum_with_author_clamp_order_max_error=decomposition_error,
                contribution_clipping_pixel_count=int(((target_linear+other_linear)>1).any(-1).sum()),
                visible_alpha_region_mean=float(images['visible-alpha'][y0:y1,x0:x1].mean()),
                variants={key:helper.score(info['pixels'][y0:y1,x0:x1].cpu().numpy(),value[y0:y1,x0:x1].clamp(0,1).cpu().numpy())
                    for key,value in images.items() if key in ['full','without-target','half-scale','quarter-scale']})
            rows.append(row)
            panel(directory,stem,box,['observed','full','without-target','visible-alpha','target-contribution','other-contribution'],
                ['原始观测','完整渲染','移除目标人物','目标可见贡献权重','目标颜色贡献（裁剪显示）','其他颜色贡献（含天空）'],'contributions',3)
            panel(directory,stem,box,['observed','full','half-scale','quarter-scale'],
                ['原始观测','原高斯尺度','目标尺度乘以 0.5','目标尺度乘以 0.25'],'scales',4)
            helper.write_json(OUT/'progress.json',dict(completed=len(rows),rows=rows))
            del result,images,gs,info,cam
    equal_tree(saved,trainer.state_dict(only_model=True),'after_all_read_only_interventions')
    assert sha(checkpoint)==json.loads((OUT/'protocol.json').read_text())['checkpoint_sha256']
    helper.write_json(OUT/'completion.json',dict(status='six_view_footprint_and_compositing_diagnostic_complete_visual_review_pending',
        model=model_stats,rows=rows,optimized_steps=0,all_model_tensors_unchanged=True,quality_accepted=False,
        seconds=time.perf_counter()-started))
    status('completed_visual_review_pending')


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():status('failed_needs_inspection',error=repr(error))
        raise
