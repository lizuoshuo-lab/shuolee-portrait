"""CPU restore/projection audit and native photograph mesh overlays, not GS results."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np, torch, pyrender, trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from hamer.utils.renderer import create_raymond_lights
from fit_hamer_consensus import L,O,P,S,read,save,sha,inputs,basis,SharedHand,project,summarize

torch.set_num_threads(6)
rows,aa,times,xy,weights,spread=inputs();manifest={(r['camera'],r['frame']):r for r in rows}
prediction_receipts={(r['camera'],r['frame']):r for r in read(P/'observations.json')['records']}
for r in rows:
    key=r['camera'],r['frame'];assert sha(P/prediction_receipts[key]['file'])==prediction_receipts[key]['sha256']
    assert sha(L/r['source'])==r['source_sha256']
models={};audit=[]
for branch in ['all_views','excluded_views']:
    out=O/branch;z=np.load(out/'parameters.npz');init={k:z[k] for k in ['root_base','pose','translation','betas','log_scale']};m=SharedHand(init,device='cpu');models[branch]=m
    old=np.load(out/'predictions.npz');b=basis(old['capture_time'])
    with torch.no_grad():J,V,R,root=[x.numpy() for x in m(torch.from_numpy(b).float())]
    difference=max(float(abs(J-old['joints_world']).max()),float(abs(V-old['vertices_world']).max()));assert difference<2e-6,difference
    K=np.array([a['K'] for a in aa]);W=np.array([a['w2c'] for a in aa]);uv,depth=project(J.astype(float),K,W);err=float(abs(uv-old['projected_xy']).max());assert err<.001,err
    report=read(out/'summary.json');residual=np.linalg.norm(uv-xy,axis=-1);mask=old['fit'];assert abs(summarize(residual[mask])['median_px']-report['fit_final']['median_px'])<.001
    rotcheck=float(abs(R@R.transpose(0,1,3,2)-np.eye(3)).max());assert rotcheck<1e-5 and np.allclose(np.linalg.det(R),1,atol=1e-5)
    # This model takes time alone: equal physical time is bit-identical across cameras.
    with torch.no_grad():same=m(torch.from_numpy(basis([726,726])).float())[1].numpy();assert np.array_equal(same[0],same[1])
    grid=np.linspace(719,734,301)
    with torch.no_grad():jg,vg,_,_=m(torch.from_numpy(basis(grid)).float());jg=jg.numpy();vg=vg.numpy()
    velocity=np.diff(jg,axis=0)/.05;acc=np.diff(jg,n=2,axis=0)/.05**2
    joint_drift=np.linalg.norm(jg[-1]-jg[0],axis=-1)
    np.savez(out/'continuous-trajectory.npz',time=grid,joints_world=jg,vertices_world=vg)
    audit.append(dict(variant=branch,CPU_restore_max_world_difference=difference,independent_native_projection_max_px=err,rotation_orthogonality_max=rotcheck,same_time_vertices_camera_independent=True,minimum_depth=float(depth.min()),maximum_joint_speed_world_per_frame=float(np.linalg.norm(velocity,axis=-1).max()),maximum_joint_acceleration_world_per_frame2=float(np.linalg.norm(acc,axis=-1).max()),note='Continuity and finite derivatives do not establish true motion or anatomical plausibility.'))
save(O/'geometry-audit.json',dict(status='both_CPU_restores_independent_projection_and_rotations_pass',predictions_checked=105,source_hashes_checked=105,records=audit,heldout_images_used_for_fitting=False,no_Gaussian_updates=True,goal_complete=False))

views=[(r['camera'],r['frame']) for r in rows if r['frame'] in [720,726,732]]
views += [(c,f) for c in ['04','09','13'] for f in range(721,732,2)]
views += [(c,732) for c in ['00','06','12']]
oldboxes={(r['camera'],r['frame']):list(map(int,r['crop'])) for r in read(L/'outputs/dynamic-human-enerf-hand-motion-082/observations.json')['records']}
dest=O/'native-panels';dest.mkdir(exist_ok=True);font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
renderer=pyrender.OffscreenRenderer(160,160);records=[];maxcameraerr=0
for c,f in views:
    stem=f'camera{c}-frame{f:06d}';path=S/'native-input'/f'{stem}.npz'
    if not path.exists():path=L/'outputs/dynamic-human-enerf-hand-time-scope-088/native-input'/f'{stem}.npz'
    z=np.load(path);W=z['w2c'].astype(float);K=z['K'].astype(float);raw=z['rgb'];B=torch.from_numpy(basis([times[c,f]])).float();geometry={}
    for branch,m in models.items():
        with torch.no_grad():J,V,_,_=m(B)
        geometry[branch]=(J[0].numpy().astype(float),V[0].numpy().astype(float))
    box=oldboxes.get((c,f))
    if box is None:
        jc=geometry['all_views'][0]@W[:3,:3].T+W[:3,3];q=jc@K.T;uv=q[:,:2]/q[:,2:];center=np.rint(uv.mean(0)).astype(int);box=[int(center[0]-80),int(center[1]-80),int(center[0]+80),int(center[1]+80)]
        roi_rule='160px centered on shared-hand predicted joints for display only; no image metric'
    else:roi_rule='pre-existing082 native160px context, no fitted-model selection'
    x0,y0,x1,y1=box;assert 0<=x0<x1<=1920 and 0<=y0<y1<=1080 and (x1-x0,y1-y0)==(160,160)
    photo=raw[y0:y1,x0:x1];columns=[photo.copy()];labels=['Native photograph']
    mono=P/'mesh-panels'/f'{stem}-rgba.npz'
    if mono.exists():
        a=np.load(mono);assert np.array_equal(a['source_bbox'],box);rgba=a['rgba'];columns.append(np.rint(np.clip(rgba[:,:,:3]*rgba[:,:,3:]+photo/255*(1-rgba[:,:,3:]),0,1)*255).astype(np.uint8));labels.append('Single-view HaMeR')
    else:
        columns.append(np.full_like(photo,245));labels.append('No monocular preview')
    saved={}
    for branch,m in models.items():
        J,V=geometry[branch];cam=V@W[:3,:3].T+W[:3,3];gl=cam*np.array([1,-1,-1]);mesh=trimesh.Trimesh(gl,m.mano.faces,process=False)
        material=pyrender.MetallicRoughnessMaterial(metallicFactor=0,roughnessFactor=.8,baseColorFactor=(.5,.75,1,1))
        scene=pyrender.Scene(bg_color=[1,1,1,0],ambient_light=(.3,.3,.3));scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=True))
        camera=pyrender.IntrinsicsCamera(float(K[0,0]),float(K[1,1]),float(K[0,2]-x0),float(K[1,2]-y0),znear=.01,zfar=100)
        scene.add(camera,pose=np.eye(4))
        for light in create_raymond_lights():scene.add_node(light)
        rgba,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);rgb=rgba[:,:,:3].astype(float)/255;alpha=(depth>0).astype(float);overlay=np.rint(np.clip(rgb*alpha[:,:,None]+photo/255*(1-alpha[:,:,None]),0,1)*255).astype(np.uint8)
        columns.append(overlay);labels.append('Shared: 15 views' if branch=='all_views' else 'Shared: excludes04/09/13')
        saved[branch+'_rgb']=rgb;saved[branch+'_alpha']=alpha;saved[branch+'_vertices_camera']=cam
        # Independent OpenGL -> native pixel projection, including principal-point crop.
        proj=camera.get_projection_matrix(160,160);clip=np.c_[gl,np.ones(len(gl))]@proj.T;ndc=clip[:,:3]/clip[:,3:];pixels=np.c_[(ndc[:,0]+1)*80,(1-ndc[:,1])*80];q=cam@K.T;expected=q[:,:2]/q[:,2:]-[x0,y0];err=float(abs(pixels-expected).max());assert err<1e-8,err;maxcameraerr=max(maxcameraerr,err)
    im=Image.new('RGB',(640,206),'white');draw=ImageDraw.Draw(im);draw.text((4,4),f'{c}/{f} | Shared MANO prior; NOT Gaussian reconstruction',font=font,fill='black')
    for k,(a,label) in enumerate(zip(columns,labels)):im.paste(Image.fromarray(a),(160*k,46));draw.text((160*k+2,24),label,font=font,fill='black')
    p=dest/f'{stem}.png';im.save(p);assert np.array_equal(np.array(Image.open(p))[46:,:160],photo)
    np.savez_compressed(dest/f'{stem}-render.npz',**saved,bbox=np.array(box))
    records.append(dict(camera=c,frame=f,capture_time=times[c,f],bbox=box,roi_rule=roi_rule,file=str(p.relative_to(O)),sha256=sha(p),source=str(path.relative_to(L)),source_sha256=sha(path),photo_native_pixels_exact=True,monocular_preview_available=mono.exists(),scope='No body/toy depth occlusion in mesh overlay; blue is a prior, not reconstructed skin.'))
    print('NATIVE_MANO',c,f,flush=True)
renderer.delete()
# Contact sheets preserve every native pixel, three rows each; all66 contexts are retained.
for offset in range(0,len(records),3):
    rr=records[offset:offset+3];im=Image.new('RGB',(640,206*len(rr)),'white')
    for k,r in enumerate(rr):im.paste(Image.open(O/r['file']),(0,k*206))
    im.save(O/f'review-{offset//3:02d}.png')
save(O/'native-preview.json',dict(status='66native_context_panels_complete',contexts=len(records),projection_matrix_max_px=maxcameraerr,records=records,photometric_reconstruction=False,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name);print('CONSENSUS_AUDIT_AND_PREVIEW_COMPLETE',len(records),flush=True)
