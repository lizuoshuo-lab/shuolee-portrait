"""Independent CPU checkpoint and saved-image verification for joint024."""
from pathlib import Path
import argparse, hashlib, json, math, statistics
import numpy as np

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-street-joint-pose-surface-024'
SOURCE=LAB/'outputs/dynamic-human-street-pose-convergence-016'
CONTROL=LAB/'outputs/dynamic-human-street-surface-refit-018/surface-retry-v2/surface_and_appearance'
SCENE=LAB/'data/dynamic-human/argoverse2/processed/training/000'
FIELDS=('_features_dc','_features_rest','_means','_scales','_quats','_opacities')
POSES=('instances_trans','instances_quats','smpl_qauts')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**23),b''):h.update(chunk)
    return h.hexdigest()

def write(name,obj):
    p=OUT/name
    if p.exists():raise FileExistsError('Preserve audit '+str(p))
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def state_audit():
    import torch
    from resume_street_deformable import equal_tree
    torch.set_num_threads(3)
    protocol=json.loads((OUT/'protocol.json').read_text())
    selection=json.loads((OUT/'training-selection.json').read_text())
    active=json.loads((OUT/'active-parameters.json').read_text())
    assert sha(SOURCE/'checkpoint-refined.pt')==protocol['source_checkpoint_sha256']
    assert sha(CONTROL/'checkpoint-refined.pt')==protocol['control_checkpoint_sha256']
    source=torch.load(SOURCE/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
    actual=torch.load(OUT/'checkpoint-refined.pt',map_location='cpu',mmap=True,weights_only=False)
    initial=json.loads((LAB/'outputs/dynamic-human-street-smpl-joint-011/articulated/initialization.json').read_text())
    idx=next(r['model_index'] for r in initial['instances']['SMPLNodes'] if r['author_instance']==79)
    assert idx==active['model_index']
    before=source['models']['SMPLNodes'];after=actual['models']['SMPLNodes']
    mask=before['points_ids'][:,0]==idx
    assert int(mask.sum())==6890 and torch.where(mask)[0].tolist()==active['point_indices']
    pose_mask=torch.zeros(before['instances_trans'].shape[:2],dtype=torch.bool)
    pose_mask[active['training_pose_frames'],idx]=True
    assert int(pose_mask.sum())==90 and not pose_mask[active['test_timesteps']].any()
    assert all(f not in active['test_timesteps'] for f in active['training_pose_frames'])
    count=0;changed={}
    for name,state in source['models'].items():
        other=actual['models'][name];assert state.keys()==other.keys()
        for field,value in state.items():
            if name=='SMPLNodes' and field in FIELDS+POSES:
                m=mask if field in FIELDS else pose_mask
                assert value.shape==other[field].shape and value.dtype==other[field].dtype
                assert torch.isfinite(other[field]).all() and torch.equal(value[~m],other[field][~m]),field
                changed[field]=int(torch.count_nonzero(value[m]!=other[field][m]))
            else:equal_tree(value,other[field],name+'.'+field);count+=int(isinstance(value,torch.Tensor))
    for key,value in source.items():
        if key!='models':equal_tree(value,actual[key],'trainer.'+key)
    def stats(v):
        v=v.float().flatten();return dict(min=float(v.min()),median=float(v.median()),p95=float(torch.quantile(v,.95)),max=float(v.max()))
    def degrees(a,b):
        aa=torch.nn.functional.normalize(a.double(),dim=-1);bb=torch.nn.functional.normalize(b.double(),dim=-1)
        v=torch.acos((aa*bb).sum(-1).abs().clamp(max=1))*360/math.pi
        return torch.where(torch.all(a==b,dim=-1),0.,v)
    centers=(after['_means'][mask]-before['_means'][mask]).norm(dim=-1)
    scales=(after['_scales'][mask]-before['_scales'][mask]).exp()
    rotations=degrees(before['_quats'][mask],after['_quats'][mask])
    trans=(after['instances_trans'][pose_mask]-before['instances_trans'][pose_mask]).norm(dim=-1)
    roots=degrees(before['instances_quats'][pose_mask],after['instances_quats'][pose_mask])
    joints=degrees(before['smpl_qauts'][pose_mask],after['smpl_qauts'][pose_mask])
    assert centers.max()<.030001 and scales.min()>=.49999 and scales.max()<=2.00001 and rotations.max()<20.01
    assert trans.max()<.100001 and roots.max()<10.01 and joints.max()<20.01
    assert selection['optimizer_updates']==100 and selection['region_backward_evaluations']==10100
    assert selection['evaluated_training_states']==101 and not selection['heldout_selection_used']
    assert [r['iteration'] for r in selection['history']]==list(range(101))
    best=min(selection['history'],key=lambda x:x['total'])
    assert best['iteration']==selection['selected_iteration'] and best['total']==selection['selected_total']
    assert set(selection['history'][0]['gradient_norms'])==set(FIELDS+POSES)
    assert all(v>0 for v in selection['history'][0]['gradient_norms'].values())
    result=dict(status='independent_target_fields_pose_bounds_frozen_state_and_selection_verified',
        source_checkpoint_sha256=protocol['source_checkpoint_sha256'],checkpoint_sha256=sha(OUT/'checkpoint-refined.pt'),
        all_non_target_attributes_background_voxel_cameras_and_stored_test_poses_unchanged=True,
        frozen_tensor_entries=count,changed_components=changed,selected_iteration=best['iteration'],
        target_center_displacement_m=stats(centers),target_scale_ratio=stats(scales),target_rotation_degrees=stats(rotations),
        additional_pose_translation_m=stats(trans),additional_root_degrees=stats(roots),additional_joint_degrees=stats(joints),
        updates=100,region_backward_evaluations=10100,quality_accepted=False,
        boundary='Heldout rendering still uses author interpolation from neighboring training poses; frozen stored test poses alone do not establish fixed test geometry.')
    write('independent-state-audit.json',result)
    print(json.dumps(result,indent=2),flush=True)

def image_audit():
    from PIL import Image,ImageDraw,ImageFont
    from audit_street_joint_images import check_quantization
    assert (OUT/'completion.json').exists()
    rois=json.loads((OUT/'person-rois.json').read_text())['regions']
    lookup={(r['frame'],r['camera'],r['instance']):r for r in rois};assert len(lookup)==435
    metrics=json.loads((OUT/'evaluation/final/metrics.json').read_text())
    scores={(r['frame'],r['camera'],r['instance']):r for r in metrics['person_box_regions']}
    assert scores.keys()==lookup.keys() and len(metrics['views'])==105
    assert {(r['frame'],r['camera']) for r in metrics['views']}=={(f,c) for f in range(10,151,10) for c in range(7)}
    images=[];regions=[]
    for view in metrics['views']:
        f,c=view['frame'],view['camera'];stem=f'frame{f:03d}-camera{c}'
        p=OUT/'evaluation/final'/(stem+'-prediction.png');pred=np.asarray(Image.open(p).convert('RGB'))
        gt=np.asarray(Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((pred.shape[1],pred.shape[0]),Image.Resampling.BILINEAR))
        assert np.array_equal(gt,np.asarray(Image.open(OUT/'ground-truth'/(stem+'.png'))))
        value,diff=check_quantization(gt,pred,view['psnr_db'])
        images.append(dict(frame=f,camera=c,sha256=sha(p),png_psnr_db=value,quantization_difference_db=diff))
        for key,r in lookup.items():
            if key[:2]!=(f,c):continue
            record=scores[key];assert r['box']==record['box'] and r['image_size']==record['image_size']
            x0,y0,x1,y1=r['box'];value,diff=check_quantization(gt[y0:y1,x0:x1],pred[y0:y1,x0:x1],record['psnr_db'])
            if key[2]==79:
                assert np.array_equal(pred[y0:y1,x0:x1],np.asarray(Image.open(OUT/'evaluation/final'/(stem+'-person79.png'))))
                assert np.array_equal(gt[y0:y1,x0:x1],np.asarray(Image.open(OUT/'ground-truth'/(stem+'-person79.png'))))
            regions.append(dict(frame=f,camera=c,instance=key[2],psnr_db=record['psnr_db'],ssim=record['ssim'],png_psnr_db=value,quantization_difference_db=diff))
    training=[]
    for label in ('before','after'):
        result=json.loads((OUT/'training-evaluation'/label/'metrics.json').read_text());assert result['count']==len(result['rows'])==101
        for f,c in [(129,4),(131,4),(149,6),(151,6)]:
            r=next(r for r in result['rows'] if (r['frame'],r['camera'])==(f,c));stem=f'frame{f:03d}-camera{c}'
            pred=np.asarray(Image.open(OUT/'training-evaluation'/label/(stem+'-prediction.png')))
            gt=np.asarray(Image.open(SCENE/'images'/f'{f:03d}_{c}.jpg').convert('RGB').resize((pred.shape[1],pred.shape[0]),Image.Resampling.BILINEAR))
            assert np.array_equal(gt,np.asarray(Image.open(OUT/'training-ground-truth'/(stem+'.png'))))
            x0,y0,x1,y1=r['box'];check_quantization(gt[y0:y1,x0:x1],pred[y0:y1,x0:x1],r['psnr_db'])
        training.append(dict(label=label,mean_psnr_db=result['mean_person_psnr_db'],mean_mae=result['mean_mae'],saved_views_checked=4))
    target=[r for r in regions if r['instance']==79];assert len(target)==13
    report=dict(status='all105heldoutPNGs435regions_and8saved_training_views_independently_verified',
        summary=dict(target13_mean_psnr_db=statistics.mean(r['psnr_db'] for r in target),target13_mean_ssim=statistics.mean(r['ssim'] for r in target)),
        images=images,regions=regions,training=training,quality_accepted=False,
        boundaries=['PSNR independently checked with PNG quantization bounds; SSIM remains original floating-point evaluator.','Pixel boxes include background and occlusion; visual and temporal acceptance separate.'])
    write('independent-image-audit.json',report)
    directory=OUT/'paired-audit';directory.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);figures=[]
    for r in target:
        f,c=r['frame'],r['camera'];stem=f'frame{f:03d}-camera{c}'
        panels=[Image.open(OUT/'ground-truth'/(stem+'-person79.png')).convert('RGB')]+[
            Image.open(run/'evaluation/final'/(stem+'-person79.png')).convert('RGB') for run in [SOURCE,CONTROL,OUT]]
        scale=min(3.,310/panels[0].width,620/panels[0].height);size=tuple(round(v*scale) for v in panels[0].size)
        canvas=Image.new('RGB',(1320,size[1]+86),'white');draw=ImageDraw.Draw(canvas)
        for col,(label,panel) in enumerate(zip(['原始观测','共同起点','表面与颜色','姿态、表面与颜色'],panels)):
            assert panel.size==panels[0].size
            draw.text((col*330+10,8),label,font=font,fill='#182d39')
            canvas.paste(panel.resize(size,Image.Resampling.NEAREST),(col*330+(330-size[0])//2,38))
        draw.text((10,size[1]+49),f'帧 {f} / 相机 {c} · 两支各 100 次全区域更新 · 原像素等比例显示，未锐化',font=font,fill='#182d39')
        name=stem+'-person79-comparison.png';canvas.save(directory/name);figures.append(dict(file=name,sha256=sha(directory/name),frame=f,camera=c))
    (directory/'figures.json').write_text(json.dumps(dict(figures=figures,visual_review_pending=True),indent=2)+'\n')
    print(json.dumps(dict(summary=report['summary'],training=training,figures=len(figures)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['state','images'],required=True);a=p.parse_args()
    state_audit() if a.part=='state' else image_audit()
