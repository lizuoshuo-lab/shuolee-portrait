from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-consensus-097'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'final-summary.json').exists()
preview=read(O/'native-preview.json');assert preview['contexts']==66
for r in preview['records']:
 p=O/r['file'];assert sha(p)==r['sha256'];assert sha(L/r['source'])==r['source_sha256'];source=np.load(L/r['source'])['rgb'];x0,y0,x1,y1=r['bbox'];photo=source[y0:y1,x0:x1];im=np.array(Image.open(p));assert np.array_equal(im[46:,:160],photo)
 z=np.load(p.with_name(p.stem+'-render.npz'))
 for k,b in enumerate(['all_views','excluded_views'],2):
  a=z[b+'_alpha'][:,:,None];rgb=z[b+'_rgb'];composite=np.rint(np.clip(rgb*a+photo/255*(1-a),0,1)*255).astype(np.uint8);assert np.array_equal(im[46:,160*k:160*(k+1)],composite)
save(O/'pixel-audit.json',dict(status='all66native_photos_and132_shared_mesh_composites_exact',photo_crops=66,mesh_composites=132,scope='Display integrity only; no reconstruction PSNR or finger GT.',goal_complete=False))
findings=[
 'All22contact sheets (66contexts) actually inspected:45even training-camera cases,18odd cases across04/09/13,3previously heldout camera cases at732.',
 '03/04 at726: shared model is more flexed than the monocular open-hand candidate; consistent as a shared grasping hypothesis, but fingers behind toy are not independently confirmed.',
 '01/03/04/09/10 at732: shared models give connected hands and visible finger divisions, yet palm width, fingertip placement and exposed native outlines retain offsets. Blue material is not reconstructed skin.',
 'Odd04/09 at729: source shows opening/exposed fingers while shared hypothesis is still strongly flexed. Smooth temporal coefficients have not recovered the precise opening event.',
 'Excluded-view and all-view candidates look similar in most inspected views. This is not a substitute for scene geometry truth; all still inherit correlated HaMeR biases.',
 '00/06/12 at732: predicted hand remains coherent but source contour and fingertips do not coincide exactly. Native display crops centered on model prediction are for inspection only, not fixed-region metrics.',
 'Overlay has no toy/body depth occlusion, and poses have no explicit collision or anatomical joint-limit constraints. Neither candidate may be substituted as exact hand geometry.'
]
save(O/'visual-review.json',dict(status='all66native_contexts_actually_viewed',files=[dict(file=f'review-{i:02d}.png',sha256=sha(O/f'review-{i:02d}.png')) for i in range(22)],findings=findings,adopt_as_hard_geometry=False,Gaussian_clarity_accepted=False,goal_complete=False))
im=Image.new('RGB',(640,412),'white')
for i,stem in enumerate(['camera03-frame000726','camera04-frame000732']):
 p=O/'native-panels'/f'{stem}.png';im.paste(Image.open(p),(0,i*206));assert np.array_equal(np.array(im)[i*206:(i+1)*206],np.array(Image.open(p)))
im.save(O/'shared-hand-versus-source.png')
a=read(O/'all_views/summary.json');b=read(O/'excluded_views/summary.json')
summary=dict(status='continuous_shared_MANO_fit_and66native_checks_complete_candidate_only',variants=2,updates_each=1200,shared_parameters=470,all_views=dict(images=105,cameras=15,initial_predicted_joint_residual=a['fit_initial'],final_predicted_joint_residual=a['fit_final']),excluded_views=dict(fit_images=84,fit_cameras=12,excluded_images=21,excluded_cameras=['04','09','13'],initial_predicted_joint_residual=b['diagnostic_initial'],final_predicted_joint_residual=b['diagnostic_final']),native_visual_contexts=66,shared_shape_and_scale=True,continuous_pose=True,camera_and_estimated_times_frozen=True,monocular_depth_used=False,source_time_is_estimated_not_hardware_GT=True,adopt_as_hard_geometry=False,Gaussian_parameters_changed=False,visual_review='visual-review.json',metric_scope='Residuals to HaMeR skeletal predictions only; not actual fingertip accuracy or appearance reconstruction.',next='Qualify a bounded soft surface/motion constraint against the original RGB loss and frozen079 baseline. Preserve body/toy occlusion and non-hand Gaussians; reject hard replacement. Actual heldout native GS renders must establish any gain.',goal_complete=False)
save(O/'final-summary.json',summary)
save(O/'next-stage-contract.json',dict(stage='098_not_started',objective='native photometric test of a weak articulated hand prior',candidate='all_views/parameters.npz',candidate_sha256=sha(O/'all_views/parameters.npz'),constraints=['No per-camera hand deformation or time retuning','No rigid replacement of whole6330Gaussian context: it contains toy/forearm contributions','Audit canonical hand association and soft distance before training','Use exact original RGB, known native cameras and training-only even-frame supervision','Compare actual heldout/odd-time GS RGB and native fingers, not blue mesh appearance','Maintain original079 baseline and reject if heldout views/continuity degrade'],not_acceptance='Current anatomical hypothesis still misaligns visible contours and opening timing; no photo-consistent surface yet.',goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name)
print('097 complete:66original-photo checks; no Gaussian clarity acceptance.')
