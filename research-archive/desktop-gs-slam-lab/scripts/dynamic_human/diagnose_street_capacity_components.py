"""Read-only component substitutions of a rejected two-view fit; no optimization."""
from pathlib import Path
import json, os, shutil, time, traceback
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import refit_street_human_surface as surface
from resume_street_deformable import cuda_tree, equal_tree, sha

helper=surface.helper
LAB=helper.LAB
OUT=LAB/'outputs/dynamic-human-street-capacity-components-022'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
CANDIDATE=LAB/'outputs/dynamic-human-street-local-capacity-020/adjacent_pair'
FIELDS=('_features_dc','_features_rest','_means','_scales','_quats','_opacities')
MODES={
    'source':(),
    'candidate':FIELDS,
    'color_only':FIELDS[:2],
    'shape_opacity_only':FIELDS[2:],
    'candidate_source_directional_color':tuple(x for x in FIELDS if x!='_features_rest'),
    'candidate_source_dc_color':tuple(x for x in FIELDS if x!='_features_dc'),
    'candidate_source_opacity':tuple(x for x in FIELDS if x!='_opacities'),
}

def state(status,**extra):
    helper.write_json(OUT/'state.json',dict(status=status,pid=os.getpid(),**extra))
    print(json.dumps(dict(status=status,**extra)),flush=True)

def main():
    assert not OUT.exists(),'Preserve existing diagnostic'
    assert (CANDIDATE/'independent-state-audit.json').exists()
    assert (CANDIDATE/'independent-image-audit.json').exists()
    OUT.mkdir(parents=True);shutil.copy2(__file__,OUT/Path(__file__).name)
    source_path=SOURCE/'checkpoint-refined.pt';candidate_path=CANDIDATE/'checkpoint-refined.pt'
    source_hash=sha(source_path);candidate_hash=sha(candidate_path)
    train=json.loads((SOURCE/'training-rois.json').read_text())['regions']
    rois=[r for r in json.loads((SOURCE/'person-rois.json').read_text())['regions'] if r['instance']==79]
    rois += [r for r in train if (r['frame'],r['camera']) in [(129,4),(131,4)]]
    assert len(rois)==15
    cfg=OmegaConf.load(SOURCE/'config.yaml')
    helper.write_json(OUT/'protocol.json',dict(status='sealed_before_read_only_interventions',source_checkpoint_sha256=source_hash,
        candidate_checkpoint_sha256=candidate_hash,script_sha256=sha(Path(__file__)),target=79,views=rois,
        modes=MODES,optimizer_updates=0,origin='Rejected two-training-view fit020; source016',
        frozen='All cameras, body poses, skinning, background, other instances and all target fields not named in each intervention.',
        boundaries=['Diagnostic substitutions of jointly learned parameters are not independent training ablations or additive causal effects.',
            'Repeatedly inspected heldout times are diagnostic views, not an untouched final benchmark.',
            'Both checkpoints share body poses, camera geometry, point identities and topology.',
            'No sharpening or generative postprocessing; native anonymous face texture retained.',
            'No new trained model or acceptance claim; fixed 3D-box ROIs include background and occluders.']))
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(12_000_000_000/torch.cuda.get_device_properties(0).total_memory)
    helper.install_capture(helper.ArgoVerseCameraData);helper.install_training_only(helper.DrivingDataset)
    state('loading_dataset')
    dataset=helper.DrivingDataset(OmegaConf.load(surface.common.BASELINE/'config.yaml').data)
    trainer=helper.MultiTrainer(**cfg.trainer,num_timesteps=156,model_config=OmegaConf.create(OmegaConf.to_container(cfg.model)),
        num_train_images=987,num_full_images=1092,test_set_indices=dataset.test_timesteps,scene_aabb=dataset.aabb.reshape(2,3),device=torch.device('cuda'))
    saved=torch.load(source_path,map_location='cpu',mmap=True,weights_only=False)
    other=torch.load(candidate_path,map_location='cpu',mmap=True,weights_only=False)
    trainer.load_state_dict(cuda_tree(saved),load_only_model=True,strict=True)
    equal_tree(saved,trainer.state_dict(only_model=True),'initial_source_restoration')
    trainer.completed_updates=30000;trainer.render_each_class=False;trainer.set_eval()
    for model in trainer.models.values():
        for parameter in model.parameters():parameter.requires_grad_(False)
    node=trainer.models['SMPLNodes']
    init=json.loads((surface.ORIGINAL/'initialization.json').read_text())
    index=next(r['model_index'] for r in init['instances']['SMPLNodes'] if r['author_instance']==79)
    mask=node.point_ids[:,0]==index;cpu_mask=mask.cpu();assert int(mask.sum())==6890
    # Independently establish the source/candidate difference boundary before intervention.
    count=0
    for model,values in saved['models'].items():
        for key,value in values.items():
            candidate=other['models'][model][key]
            if model=='SMPLNodes' and key in FIELDS:
                assert torch.equal(value[~cpu_mask],candidate[~cpu_mask])
            else: equal_tree(value,candidate,model+'.'+key);count+=int(torch.is_tensor(value))
    for key,value in saved.items():
        if key!='models':equal_tree(value,other[key],key)
    original={k:getattr(node,k).detach().clone() for k in FIELDS}
    changed={k:other['models']['SMPLNodes'][k][cpu_mask].cuda() for k in FIELDS}
    assert changed['_features_rest'].shape[1:]==(3,3),'Expected degree-one SH (three directional bases x RGB)'
    rows=[];checks=[]
    with torch.no_grad():
        for name,fields in MODES.items():
            state('rendering_component_substitution',mode=name)
            for field in FIELDS:
                getattr(node,field).copy_(original[field])
                if field in fields:getattr(node,field)[mask]=changed[field]
            frozen=surface.frozen_check(trainer,saved,cpu_mask,fields)
            for field in FIELDS:
                expected=changed[field] if field in fields else original[field][mask]
                assert torch.equal(getattr(node,field)[mask],expected)
            if name=='candidate':equal_tree(other,trainer.state_dict(only_model=True),'candidate_reproduction')
            checks.append(dict(mode=name,frozen_entries=frozen,target_fields_exact=True))
            for roi in rois:
                frame,camera=roi['frame'],roi['camera'];stem=f'frame{frame:03d}-camera{camera}'
                info,cam=dataset.full_image_set.get_image(frame*7+camera,1.)
                info,cam=helper.to_cuda(info),helper.to_cuda(cam)
                result=trainer(info,cam,novel_view=True)
                pred=result['rgb'].clamp(0,1).cpu().numpy();gt=info['pixels'].cpu().numpy()
                directory=OUT/'renders'/name
                helper.save_rgb(directory/(stem+'-prediction.png'),pred)
                x0,y0,x1,y1=roi['box']
                helper.save_rgb(directory/(stem+'-person79.png'),pred[y0:y1,x0:x1])
                if name=='source':helper.save_rgb(OUT/'ground-truth'/(stem+'.png'),gt)
                match=None
                if name in ('source','candidate'):
                    folder=SOURCE if name=='source' else CANDIDATE
                    folder=folder/('evaluation/final' if frame in dataset.test_timesteps else 'training-evaluation/after')
                    baseline=np.asarray(Image.open(folder/(stem+'-prediction.png')))
                    actual=np.uint8(pred*255+.5);match=int(np.abs(baseline.astype(np.int16)-actual.astype(np.int16)).max())
                    assert match<=1,(name,stem,match)
                rows.append(dict(mode=name,frame=frame,camera=camera,box=roi['box'],heldout=frame in dataset.test_timesteps,
                    **helper.score(gt[y0:y1,x0:x1],pred[y0:y1,x0:x1]),source_png_max_byte_difference=match))
            surface.frozen_check(trainer,saved,cpu_mask,fields)
        for field in FIELDS:getattr(node,field).copy_(original[field])
        equal_tree(saved,trainer.state_dict(only_model=True),'final_restoration')
    assert source_hash==sha(source_path) and candidate_hash==sha(candidate_path)
    summary={name:dict(heldout13_mean_psnr_db=float(np.mean([r['psnr_db'] for r in rows if r['mode']==name and r['heldout']])),
        training2_mean_psnr_db=float(np.mean([r['psnr_db'] for r in rows if r['mode']==name and not r['heldout']]))) for name in MODES}
    helper.write_json(OUT/'results.json',dict(rows=rows,summary=summary))
    helper.write_json(OUT/'tensor-audit.json',dict(source_candidate_shared_frozen_entries=count,per_mode=checks,
        final_source_restored_exact=True,checkpoint_files_unchanged=True,sh_degree=1,target_points=6890))
    helper.write_json(OUT/'completion.json',dict(status='105_read_only_renders_complete_image_audit_pending',views=15,modes=7,
        optimizer_updates=0,summary=summary,quality_accepted=False))
    state('completed_image_and_visual_audit_pending',summary=summary)

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():state('failed_needs_inspection',error=repr(error))
        traceback.print_exc();raise
