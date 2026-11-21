"""Exercise real CUDA kernels and the complete capture-time image loader."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import torch
from PIL import Image

ROOT=Path('/opt/gs-slam-lab/dynamic-human')
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=LAB/'outputs/dynamic-human-street-005'
DATA=LAB/'data/dynamic-human/argoverse2/processed/training/000'
sys.path.insert(0,str(ROOT/'repos/drivestudio'))
os.chdir(ROOT/'repos/drivestudio')

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def operators():
    from pytorch3d.ops import knn_points
    import nvdiffrast.torch as dr
    import gsplat
    torch.set_num_threads(4)
    torch.manual_seed(73)
    x=torch.rand((1,9,3),device='cuda',requires_grad=True)
    y=torch.rand((1,13,3),device='cuda',requires_grad=True)
    distances=knn_points(x,y,K=2).dists
    reference=((x[:,:,None,:]-y[:,None,:,:])**2).sum(-1).topk(2,largest=False).values
    torch.testing.assert_close(distances,reference,rtol=1e-5,atol=1e-6)
    actual_grad=torch.autograd.grad(distances.sum(),(x,y),retain_graph=True)
    reference_grad=torch.autograd.grad(reference.sum(),(x,y))
    for a,b in zip(actual_grad,reference_grad):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    print('PYTORCH3D CUDA FORWARD/BACKWARD PASSED',flush=True)
    context=dr.RasterizeCudaContext(device='cuda')
    vertices=torch.tensor([[[-.8,-.7,0,1],[.8,-.7,0,1],[0,.8,0,1]]],device='cuda',requires_grad=True)
    triangles=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
    attrs=torch.tensor([[[.2,.4,.6],[.8,.2,.3],[.5,.7,.1]]],device='cuda',requires_grad=True)
    rast,rast_db=dr.rasterize(context,vertices,triangles,resolution=[32,32])
    rendered,_=dr.interpolate(attrs,rast,triangles)
    visible=rast[...,3]>0
    assert int(visible.sum())>100
    # Verify barycentric interpolation from the reported raster coordinates.
    bary=torch.stack((rast[...,0],rast[...,1],1-rast[...,0]-rast[...,1]),dim=-1)
    exact=bary@attrs[0]
    torch.testing.assert_close(rendered[visible],exact[visible],rtol=1e-5,atol=1e-6)
    rendered.square().sum().backward()
    assert torch.isfinite(attrs.grad).all() and float(attrs.grad.abs().sum())>0
    assert torch.isfinite(vertices.grad).all() and float(vertices.grad.abs().sum())>0
    print('NVDIFFRAST CUDA FORWARD/BACKWARD PASSED',flush=True)
    means=torch.tensor([[.15,-.1,2.]],device='cuda',requires_grad=True)
    colors=torch.tensor([[.3,.5,.7]],device='cuda',requires_grad=True)
    kwargs=dict(means=means,quats=torch.tensor([[1.,0,0,0]],device='cuda'),
        scales=torch.tensor([[.14,.17,.12]],device='cuda'),opacities=torch.tensor([.7],device='cuda'),
        colors=colors,viewmats=torch.eye(4,device='cuda')[None],
        Ks=torch.tensor([[[30.,0,16.],[0,30.,16.],[0,0,1.]]],device='cuda'),
        width=32,height=32,packed=False,near_plane=.01,far_plane=100.,rasterize_mode='classic')
    rgb,alpha,_=gsplat.rasterization(**kwargs)
    assert torch.isfinite(rgb).all() and 0<float(alpha.max())<=1
    torch.testing.assert_close(rgb,alpha*colors[0],rtol=2e-5,atol=2e-6)
    rgb.sum().backward()
    torch.testing.assert_close(colors.grad,alpha.detach().sum().expand_as(colors),rtol=2e-5,atol=2e-5)
    assert torch.isfinite(means.grad).all()
    receipt=dict(status='real_cuda_operators_verified',device=torch.cuda.get_device_name(),torch=torch.__version__,
        gsplat=gsplat.__version__,pytorch3d_knn_forward_max_abs=float((distances-reference).abs().max()),
        pytorch3d_knn_gradient_max_abs=max(float((a-b).abs().max()) for a,b in zip(actual_grad,reference_grad)),
        nvdiffrast_triangle_covered_pixels=int(visible.sum()),
        nvdiffrast_position_gradient_nonzero=True,nvdiffrast_attribute_gradient_nonzero=True,
        gsplat_alpha_peak=float(alpha.max()),gsplat_color_gradient_analytic_match=True,
        limitation='Small deterministic kernel smoke; no street training or model-quality claim')
    save('linux-cuda-smoke.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)

def cameras():
    from datasets.argoverse.argoverse_sourceloader import ArgoVerseCameraData
    from street_capture_time_adapter import install,receipt
    install(ArgoVerseCameraData)
    records=json.loads((DATA/'capture_timestamps.json').read_text())['frames']
    rows=[]
    # Actual full constructor, mask/image loads and camera-ray generation.
    for frame,scale in [(0,4),(134,4)]:
        for camera_id in range(7):
            camera=ArgoVerseCameraData(dataset_name='argoverse',data_path=str(DATA),cam_id=camera_id,
                start_timestep=frame,end_timestep=frame+1,load_dynamic_mask=True,load_sky_mask=True,
                downscale_when_loading=scale,device=torch.device('cpu'))
            camera.set_unique_ids(camera_id,torch.tensor([camera_id]))
            camera.load_time(torch.tensor([frame/155.]))
            images,calibration=camera.get_image(0)
            image_path=DATA/'images'/f'{frame:03d}_{camera_id}.jpg'
            with Image.open(image_path) as image:
                image=image.convert('RGB').resize((camera.WIDTH,camera.HEIGHT),Image.Resampling.BILINEAR)
                expected=torch.from_numpy(np.array(image))/255
            assert torch.equal(images['pixels'],expected)
            assert 'fine_dynamic_masks' in str(camera.human_mask_filepaths[0])
            for key,folder,interpolation in [('human_masks','fine_dynamic_masks/human',Image.Resampling.BILINEAR),
                    ('dynamic_masks','fine_dynamic_masks/all',Image.Resampling.BILINEAR),
                    ('vehicle_masks','fine_dynamic_masks/vehicle',Image.Resampling.BILINEAR),
                    ('sky_masks','sky_masks',Image.Resampling.NEAREST)]:
                with Image.open(DATA/folder/f'{frame:03d}_{camera_id}.png') as mask:
                    wanted=np.array(mask.convert('L').resize((camera.WIDTH,camera.HEIGHT),interpolation))>0
                assert np.array_equal(images[key].numpy(),wanted)
            record=next(r for r in records[frame]['cameras'] if int(r['camera_id'])==camera_id)
            expected_pose=np.linalg.inv(np.loadtxt(DATA/'ego_pose'/f'{frame:03d}.txt'))@np.loadtxt(DATA/record['capture_pose'])
            np.testing.assert_allclose(calibration['camera_to_world'].numpy(),expected_pose,rtol=1e-6,atol=2e-6)
            assert int(camera.capture_timestamps_ns[0])==int(record['timestamp_ns'])
            assert torch.isfinite(images['viewdirs']).all()
            torch.testing.assert_close(images['viewdirs'].norm(dim=-1),torch.ones((camera.HEIGHT,camera.WIDTH)),rtol=1e-5,atol=1e-6)
            # Independent center-pixel ray and source pose check.
            yy,xx=camera.HEIGHT//2,camera.WIDTH//2
            k=calibration['intrinsics'].numpy().astype(np.float64)
            vector=np.linalg.inv(k)@np.array([xx+.5,yy+.5,1.])
            direction=expected_pose[:3,:3]@vector
            direction/=np.linalg.norm(direction)
            np.testing.assert_allclose(images['viewdirs'][yy,xx].numpy(),direction,atol=1e-6)
            rows.append(dict(frame=frame,camera=camera_id,load_size=camera.load_size,
                image_exact=True,masks_exact=True,pose_reference='image_exposure',rays_finite_and_unit=True,
                center_ray_independent_check=True,person_predicted_pixels=int(images['human_masks'].sum())))
            del camera,images
            print('FULL CAMERA LOADER',frame,camera_id,'PASSED',flush=True)
    save('full-camera-loader-audit.json',dict(status='full_author_constructor_and_rays_verified',cases=rows,
        adapter=receipt(DATA),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['14 actual image/mask/ray cases at downscale4; separate matrix audit covers all1092 capture poses',
            'Masks are semantic predictions, not ground truth; author human category includes riders',
            'No SMPL mesh, moving-object time interpolation, or joint training verified']))

if __name__=='__main__':
    operators()
    cameras()
