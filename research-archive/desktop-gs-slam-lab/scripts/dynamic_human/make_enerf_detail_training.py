"""Derive one controlled loss change from the retained graph training script."""
from pathlib import Path
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-detail-weight-076';B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
enabled=['01','03','04','07','08','09','10','11','14','16']
cams=['01','02','03','04','05','07','08','09','10','11','13','14','15','16','17']
save(R/'region-visual-review.json',dict(status='15native_contact_sheets45source_images_actually_viewed',
    files=[dict(path=f'review/camera{c}-three-times.png',sha256=sha(R/'review'/f'camera{c}-three-times.png')) for c in cams],
    extra_weight_cameras=enabled,extra_weight_images=70,global_only_cameras=[c for c in cams if c not in enabled],
    findings_zh='扩大窗口后，可见手掌、指缘和鞋部多数落入窗口，仍包含衣袖、玩偶和背景。02、05、13、15、17机位在重叠处的黑衣人物预测明显偏向白衣人物，或漏掉黑衣鞋/手，因此这些机位只保留原全图损失。其他机位的窗口仍是近似图像区域，不是精确部位分割或可见性标签。',
    accepted_for_image_window_weight_only=True,anatomical_or_instance_labels_accepted=False,goal_complete=False))
protocol=read(R/'protocol.json');protocol.update(extra_weight_cameras=enabled,extra_weight_images=70,global_only_images=35,
    visual_scope='15cameras x3source times744750756=45images. All105region arrays numerically checked. Other even times not individually viewed; windows remain weak photometric emphasis, not anatomical annotations.')
save(R/'protocol.json',protocol)
source=S/'train_enerf_graph_control.py';text=source.read_text(encoding='utf8')
def replace(a,b):
    global text
    assert text.count(a)==1,(a[:80],text.count(a))
    text=text.replace(a,b,1)
replace("records=[r for r in read(OUT/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105 and all(r['frame']%2==0 for r in records);obs=[]", "records=[r for r in read(OUT/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105 and all(r['frame']%2==0 for r in records);obs=[]\n detail_protocol=read(a.result_root/'protocol.json');review=read(a.result_root/'region-visual-review.json');assert review['accepted_for_image_window_weight_only'];enabled=set(detail_protocol['extra_weight_cameras']);region_rows={(r['camera'],r['frame']):r for r in read(a.result_root/'preparation.json')['records']};detail_smoked=(root/'detail-gradient-audit.json').exists()")
replace("obs.append(dict(camera=r['camera'],frame=r['frame'],rgb=", "region_file=a.result_root/'regions'/f\"camera{r['camera']}-frame{r['frame']:06d}.npz\";assert sha(region_file)==region_rows[r['camera'],r['frame']]['region_sha256'];roi=np.load(region_file)['mask'];assert roi.shape==z['valid'].shape and not np.any(roi&~z['valid']);roi=roi if r['camera'] in enabled else np.zeros_like(roi)\n  obs.append(dict(camera=r['camera'],frame=r['frame'],detail_roi=tensor(roi[y0:y1,x0:x1])[...,None],rgb=")
replace("background=background.cuda();start=0;history=[];elapsed=0", "save(root/'detail-protocol.json',dict(ROI_weight=.2,region_manifest_sha256=sha(a.result_root/'preparation.json'),region_review_sha256=sha(a.result_root/'region-visual-review.json'),extra_weight_cameras=sorted(enabled),definition='meanRGBabsoluteerror over union of detailwindows; originalvalidmask andtarget',goal_complete=False))\n background=background.cuda();start=0;history=[];elapsed=0")
replace("photometric=loss;regularization=", "base_photometric=loss;detail_loss=((pred-target).abs()*o['detail_roi']).sum()/(3*o['detail_roi'].sum()).clamp_min(1)\n  if not detail_smoked and o['detail_roi'].sum()>0:\n   expected=(pred-target).abs()[o['detail_roi'][...,0].bool()].mean();assert torch.allclose(detail_loss,expected,atol=1e-7)\n   g=torch.autograd.grad(detail_loss,pred,retain_graph=True)[0];outside=o['detail_roi'][...,0]==0;assert g[outside].abs().max()==0 and torch.isfinite(g).all() and g.abs().sum()>0\n   save(root/'detail-gradient-audit.json',dict(status='realnativefirstROIloss_equals_boolean_pixelmean_and_zero_gradient_outside_window',camera=o['camera'],frame=o['frame'],roi_pixels=int(o['detail_roi'].sum()),max_gradient_outside=float(g[outside].abs().max()),pixel_gradient_l1=float(g.abs().sum()),goal_complete=False));detail_smoked=True\n  loss=loss+.2*detail_loss;photometric=loss;regularization=")
replace("loss=float(loss),photometric_loss=float(photometric),rigidity=", "loss=float(loss),photometric_loss=float(photometric),base_photometric_loss=float(base_photometric),detail_L1=float(detail_loss),rigidity=")
replace("['train_enerf_graph_control.py','enerf_articulated_model.py'", "['train_enerf_detail_weight.py','enerf_articulated_model.py'")
target=S/'train_enerf_detail_weight.py';target.write_text(text,encoding='utf8')
save(R/'source-change-audit.json',dict(status='controlled_derivation_prepared',baseline_source_sha256=sha(source),new_source_sha256=sha(target),
    modifications=['load fixed native union ROI for70images;35zero masks','add0.2normalizedlocalRGBL1','actualselectedpixelgradient verification andrecords','source snapshot filename'],
    unchanged='renderer,body/graph,initialGaussianparameters,optimizer,rates,draws,randombackground,globalRGB/alpha losses,regularization,densityrules',goal_complete=False))
plan=read(B/'visual-review-plan.json');plan['status']='fixed_before_detail_weight_training';plan['decision']='Compare076to074 same correctedRGBmotion; also check retained067 context. No clarity claim from lowerweightedtrainingloss.';save(R/'visual-review-plan.json',plan)
shutil.copy2(__file__,R/Path(__file__).name)
print('Controlled trainer and45-image review saved;70weighted +35global-only images.')
