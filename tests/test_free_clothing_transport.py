"""Independent known-geometry controls, no body forward or real training."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import warnings
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shuolee_portrait.free_clothing_transport import FreeClothingTransport

FRAMES = ('side', 'reference', 'turned')
IDS = ('free-10', 'free-2', 'free-37', 'free-4')
FACES = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
SUPPORT = np.array([0, 0, 1, 1], dtype=np.int64)
V = np.array([[0., 0., 4.], [1., 0., 4.], [0., 1., 4.],
              [2., 0., 4.], [2., 1., 4.], [2., 0., 5.]])
XYZ = np.array([[.2, .3, 4.25], [1.4, -.4, 4.6], [2.4, .5, 4.5], [-.2, 1.2, 2.]])
L = np.array([[.21, 0., 0.], [.06, .12, 0.], [-.03, .04, .08]])
COV = np.stack([(L*(i+1)/2)@(L*(i+1)/2).T for i in range(4)])
CAM = np.array([.2, -1.4, -.7])
Q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
Q180 = np.diag([1., -1., -1.])
TRANS = np.array([.4, -.7, .2])
METRICS = {}

def digest(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def construct(vertices=None, frame_ids=FRAMES, **kwargs):
    vertices = np.stack([2*V@Q.T+TRANS, V, V@Q180.T]) if vertices is None else vertices
    args = dict(frame_ids=frame_ids, reference_frame_id='reference', vertices_world=vertices,
                faces=FACES, motion_support_triangle_ids=SUPPORT, gaussian_ids=IDS,
                translation_unit='meters', frame_source_sha256={f:digest(v) for f,v in zip(frame_ids,vertices)})
    args.update(kwargs)
    return FreeClothingTransport(**args)

def run_np(obj, frame='side', xyz=XYZ, cov=COV, camera=CAM, ids=IDS):
    return obj.transport_numpy(frame, xyz, cov, camera, gaussian_ids=ids)

def project(x, K):
    p=x@K.T
    return p[:,:2]/p[:,2:]

def dense_rotation():
    axis=np.array([.37,-.53,.71]);axis/=np.linalg.norm(axis);angle=.41
    x,y,z=axis;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross

class TransportControl(unittest.TestCase):
    def test_reference_is_exact_for_off_surface_points(self):
        p=run_np(construct(), 'reference')
        np.testing.assert_array_equal(p['centers'],XYZ)
        np.testing.assert_array_equal(p['covariance'],COV)
        self.assertEqual(p['frame_id'],'reference')

    def test_known_similarity_and_nonzero_free_offsets(self):
        p=run_np(construct())
        np.testing.assert_allclose(p['centers'],2*XYZ@Q.T+TRANS,atol=2e-14,rtol=0)
        # First support is horizontal; the first two points stay off its plane.
        self.assertAlmostEqual(p['centers'][0,2]-(2*4+TRANS[2]),.5)
        self.assertAlmostEqual(p['centers'][1,2]-(2*4+TRANS[2]),1.2)
        self.assertEqual(len(p['centers']),4)  # no points created per triangle

    def test_full_covariance_independent_factor_transform(self):
        p=run_np(construct())
        expected=np.stack([(2*Q@(L*(i+1)/2))@(2*Q@(L*(i+1)/2)).T for i in range(4)])
        np.testing.assert_allclose(p['covariance'],expected,atol=1e-14,rtol=0)
        self.assertTrue(np.all(np.linalg.eigvalsh(p['covariance'])>0))
        self.assertGreater(np.max(abs(p['covariance']-COV)),.01)

    def test_legal_half_turn_is_not_rejected_as_flip(self):
        p=run_np(construct(),'turned')
        np.testing.assert_allclose(p['centers'],XYZ@Q180.T,atol=1e-14,rtol=0)

    def test_sh_view_direction_matches_inverse_camera(self):
        p=run_np(construct())
        camera_ref=(CAM-TRANS)@Q/2
        d=XYZ-camera_ref;d/=np.linalg.norm(d,axis=1,keepdims=True)
        np.testing.assert_allclose(p['reference_view_directions'],d,atol=2e-14,rtol=0)
        # Independent official l=1 polynomial, odd terms catch ray sign.
        sh=np.array([.6,.2,-.4,.8])
        expected=.28209479177387814*sh[0]+.4886025119029199*(-d[:,1]*sh[1]+d[:,2]*sh[2]-d[:,0]*sh[3])
        actual_d=p['reference_view_directions']
        actual=.28209479177387814*sh[0]+.4886025119029199*(-actual_d[:,1]*sh[1]+actual_d[:,2]*sh[2]-actual_d[:,0]*sh[3])
        world=p['centers']-CAM;world/=np.linalg.norm(world,axis=1,keepdims=True)
        wrong=.28209479177387814*sh[0]+.4886025119029199*(-world[:,1]*sh[1]+world[:,2]*sh[2]-world[:,0]*sh[3])
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
        METRICS['wrong_world_SH_degree1_max_difference']=float(np.max(abs(wrong-expected)))
        self.assertGreater(METRICS['wrong_world_SH_degree1_max_difference'],.01)

    def test_nonzero_degree3_SH_uses_same_reference_direction(self):
        source=ROOT/'src/avatar_core/utils/sh_utils.py'
        spec=importlib.util.spec_from_file_location('independent_official_sh',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sh=np.random.default_rng(112).normal(size=(4,3,16))*.1
        p=run_np(construct());cr=(CAM-TRANS)@Q/2
        d=XYZ-cr;d/=np.linalg.norm(d,axis=1,keepdims=True)
        np.testing.assert_allclose(module.eval_sh(3,sh,p['reference_view_directions']),module.eval_sh(3,sh,d),atol=1e-14,rtol=0)

    def test_frame_order_is_explicit(self):
        vertices=np.stack([2*V@Q.T+TRANS,V,V@Q180.T])
        order=[2,0,1]
        p=run_np(construct(vertices[order],tuple(FRAMES[i] for i in order)))
        np.testing.assert_allclose(p['centers'],2*XYZ@Q.T+TRANS,atol=2e-14,rtol=0)
        with self.assertRaises((ValueError,KeyError)):run_np(construct(),'missing')

    def test_point_identity_cannot_silently_reorder(self):
        with self.assertRaises(ValueError):run_np(construct(),ids=IDS[::-1])
        with self.assertRaises(ValueError):construct(gaussian_ids=('duplicate',)*4)
        with self.assertRaises(ValueError):run_np(construct(),xyz=XYZ[:3],cov=COV[:3])

    def test_units_and_source_identity_required(self):
        with self.assertRaises((ValueError,TypeError)):construct(translation_unit=None)
        with self.assertRaises(ValueError):construct(translation_unit='unknown_sfm')
        with self.assertRaises(ValueError):construct(frame_source_sha256={'side':'bad'})

    def test_invalid_support_and_degenerate_triangle_rejected(self):
        with self.assertRaises(ValueError):construct(motion_support_triangle_ids=np.array([0,0,1,2]))
        v=np.stack([2*V@Q.T+TRANS,V,V@Q180.T]);v[0,2]=v[0,1]
        with self.assertRaises(ValueError):construct(v)

    def test_non_similarity_is_explicit_approximation(self):
        v=np.stack([V.copy(),V,V@Q180.T]);v[0,1,0]+=1
        p=run_np(construct(v))
        self.assertGreater(np.max(p['support_vertex_residual']),.01)
        # It remains a declared isotropic field, not exact affine shear.
        eye=np.eye(3)[None]
        np.testing.assert_allclose(p['rotations']@np.swapaxes(p['rotations'],-1,-2),np.broadcast_to(eye,(4,3,3)),atol=1e-14)

    def test_source_array_mutation_cannot_change_field(self):
        v=np.stack([2*V@Q.T+TRANS,V,V@Q180.T]);obj=construct(v)
        before=run_np(obj)['centers'].copy();v[:]=100
        np.testing.assert_array_equal(run_np(obj)['centers'],before)

    def test_bad_covariance_or_zero_ray_is_rejected(self):
        c=COV.copy();c[0,0,0]=-1
        with self.assertRaises(ValueError):run_np(construct(),cov=c)
        c=COV.copy();c[0,0,1]+=.1
        with self.assertRaises(ValueError):run_np(construct(),cov=c)
        with self.assertRaises(ValueError):run_np(construct(),'reference',camera=XYZ[0])

    def test_small_covariance_cannot_hide_relative_asymmetry(self):
        cov=np.broadcast_to(np.eye(3)*1e-12,(4,3,3)).copy()
        np.testing.assert_array_equal(run_np(construct(),'reference',cov=cov)['covariance'],cov)
        cov[:,0,1]=1e-13
        # The lower-triangle eigensolver still reports positive eigenvalues;
        # this does not make the asymmetric input a valid covariance.
        self.assertTrue(np.all(np.linalg.eigvalsh(cov)>0))
        with self.assertRaisesRegex(ValueError,'must be symmetric'):
            run_np(construct(),'reference',cov=cov)
        with self.assertRaisesRegex(ValueError,'must be symmetric'):
            construct().transport_torch('reference',torch.tensor(XYZ),torch.tensor(cov),
                                        torch.tensor(CAM),gaussian_ids=IDS)

    def test_numpy_covariance_underflow_is_rejected(self):
        v=np.stack([V*1e-50,V,V@Q180.T])
        cov=np.broadcast_to(np.eye(3)*1e-230,(4,3,3)).copy()
        with self.assertRaisesRegex(ValueError,'[Pp]ositive|[Cc]ovariance'):
            run_np(construct(v),cov=cov,camera=np.zeros(3))

    def test_torch_covariance_underflow_is_rejected(self):
        v=np.stack([V*1e-20,V,V@Q180.T]);obj=construct(v)
        cov=torch.eye(3,dtype=torch.float32).repeat(4,1,1)*1e-6
        with self.assertRaisesRegex(ValueError,'[Pp]ositive|[Cc]ovariance'):
            obj.transport_torch('side',torch.tensor(XYZ,dtype=torch.float32),cov,
                                torch.zeros(3,dtype=torch.float32),gaussian_ids=IDS)

    def test_cpu_transport_ignores_ambient_default_meta_device(self):
        obj=construct();x=torch.tensor(XYZ,dtype=torch.float32,device='cpu')
        cov=torch.tensor(COV,dtype=torch.float32,device='cpu')
        for camera in (CAM,torch.tensor(CAM,dtype=torch.float32,device='cpu')):
            baseline=obj.transport_torch('side',x,cov,camera,gaussian_ids=IDS)
            with torch.device('meta'):
                actual=obj.transport_torch('side',x,cov,camera,gaussian_ids=IDS)
                self.assertEqual(torch.empty(0).device.type,'meta')
            for key,value in baseline.items():
                if isinstance(value,torch.Tensor):
                    self.assertEqual(actual[key].device.type,'cpu',key)
                    torch.testing.assert_close(actual[key],value,atol=0,rtol=0)

    def test_cpu_autocast_does_not_lower_transport_precision_or_gradients(self):
        rotation=dense_rotation();scale=1.7
        obj=construct(np.stack([scale*V@rotation.T+TRANS,V,V@Q180.T]))
        x=torch.tensor(XYZ,dtype=torch.float32,requires_grad=True)
        cov=torch.tensor(COV,dtype=torch.float32,requires_grad=True)
        baseline=obj.transport_torch('side',x,cov,CAM,gaussian_ids=IDS)
        x2=x.detach().clone().requires_grad_();c2=cov.detach().clone().requires_grad_()
        previous=torch.is_autocast_enabled('cpu')
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            actual=obj.transport_torch('side',x2,c2,CAM,gaussian_ids=IDS)
            self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(torch.is_autocast_enabled('cpu'),previous)
        for key,value in baseline.items():
            if isinstance(value,torch.Tensor):
                self.assertEqual(actual[key].dtype,value.dtype,key)
                torch.testing.assert_close(actual[key],value,atol=0,rtol=0)
        def loss(result):
            return result['centers'].square().sum()+result['covariance'].square().sum()+result['reference_view_directions'][:,0].sum()
        expected_grads=torch.autograd.grad(loss(baseline),(x,cov))
        actual_grads=torch.autograd.grad(loss(actual),(x2,c2))
        for actual_grad,expected_grad in zip(actual_grads,expected_grads):
            torch.testing.assert_close(actual_grad,expected_grad,atol=0,rtol=0)

    def test_float32_dense_rotation_covariance_and_gradients(self):
        rotation=dense_rotation();scale=1.7
        obj=construct(np.stack([scale*V@rotation.T+TRANS,V,V@Q180.T]))
        x=torch.tensor(XYZ,dtype=torch.float32,requires_grad=True)
        cov=torch.tensor(COV,dtype=torch.float32,requires_grad=True)
        actual=obj.transport_torch('side',x,cov,CAM,gaussian_ids=IDS)
        q=torch.tensor(rotation,dtype=torch.float32);shift=torch.tensor(TRANS,dtype=torch.float32)
        x2=x.detach().clone().requires_grad_();c2=cov.detach().clone().requires_grad_()
        expected_x=scale*x2@q.T+shift
        expected_cov=(scale*q)@c2@(scale*q).T
        camera_ref=(torch.tensor(CAM,dtype=torch.float32)-shift)@q/scale
        d=x2-camera_ref;d=d/torch.linalg.vector_norm(d,dim=-1,keepdim=True)
        torch.testing.assert_close(actual['centers'],expected_x,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(actual['covariance'],expected_cov,atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(actual['reference_view_directions'],d,atol=2e-6,rtol=2e-6)
        self.assertGreater(float(actual['covariance'][:,0,2].abs().max()),.001)
        weight=torch.arange(1,13,dtype=torch.float32).reshape(4,3)/10
        loss=(actual['centers']*weight).sum()+actual['covariance'].square().sum()+actual['reference_view_directions'][:,0].sum()
        direct=(expected_x*weight).sum()+expected_cov.square().sum()+d[:,0].sum()
        gradients=torch.autograd.grad(loss,(x,cov));expected_gradients=torch.autograd.grad(direct,(x2,c2))
        for actual_grad,expected_grad in zip(gradients,expected_gradients):
            torch.testing.assert_close(actual_grad,expected_grad,atol=2e-6,rtol=2e-5)
        METRICS['float32_dense_rotation_center_max_difference_m']=float((actual['centers'].detach()-expected_x.detach()).abs().max())

    def test_cpu_autograd_matches_independent_affine_chain(self):
        obj=construct();x=torch.tensor(XYZ,requires_grad=True);cov=torch.tensor(COV,requires_grad=True)
        p=obj.transport_torch('side',x,cov,torch.tensor(CAM),gaussian_ids=IDS)
        weight=torch.arange(1,13,dtype=torch.float64).reshape(4,3)/10
        loss=(p['centers']*weight).sum()+p['covariance'].square().sum()+p['reference_view_directions'][:,0].sum()
        gx,gc=torch.autograd.grad(loss,(x,cov))
        x2=torch.tensor(XYZ,requires_grad=True);c2=torch.tensor(COV,requires_grad=True)
        q=torch.tensor(Q);expected_x=2*x2@q.T+torch.tensor(TRANS)
        ll=torch.tensor(CAM-TRANS)@q/2
        d=x2-ll;d=d/torch.linalg.norm(d,dim=1,keepdim=True)
        expected_c=(2*q)@c2@(2*q).T
        direct=(expected_x*weight).sum()+expected_c.square().sum()+d[:,0].sum()
        ex,ec=torch.autograd.grad(direct,(x2,c2))
        torch.testing.assert_close(gx,ex,atol=1e-12,rtol=0)
        torch.testing.assert_close(gc,ec,atol=1e-12,rtol=0)
        METRICS['center_gradient_max_difference']=float((gx-ex).abs().max())

    def test_serialized_readback_and_mutation_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'transport.json';obj=construct();obj.save(path)
            restored=FreeClothingTransport.load(path)
            np.testing.assert_array_equal(run_np(restored)['centers'],run_np(obj)['centers'])
            np.testing.assert_array_equal(run_np(restored)['covariance'],run_np(obj)['covariance'])
            restored.assert_unchanged()
            with self.assertRaises((ValueError,FileExistsError)):obj.save(path)
            path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaises((ValueError,RuntimeError)):restored.assert_unchanged()

    def test_payload_crc_corruption_is_explicit_input_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'transport.json';construct().save(path);payload=path.with_suffix('.npz')
            damaged=bytearray(payload.read_bytes())
            # Change the central-directory CRC without changing compressed
            # bytes, then update the outer hash so validation reaches ZIP CRC.
            offset=damaged.find(b'PK\x01\x02');self.assertGreaterEqual(offset,0)
            damaged[offset+16]^=1
            payload.write_bytes(damaged)
            metadata=json.loads(path.read_text())
            metadata['payload']['sha256']=hashlib.sha256(damaged).hexdigest()
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'Invalid transport NPZ') as caught:
                FreeClothingTransport.load(path)
            self.assertIsInstance(caught.exception.__cause__,zipfile.BadZipFile)

    def test_duplicate_npz_fields_rejected_even_when_last_copy_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'transport.json';obj=construct();obj.save(path);payload=path.with_suffix('.npz')
            with zipfile.ZipFile(payload) as archive:
                entries=[(name,archive.read(name)) for name in archive.namelist()]
            different=io.BytesIO();np.save(different,obj.vertices_world+1.)
            buffer=io.BytesIO()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('vertices_world.npy',different.getvalue())
                    for name,data in entries:archive.writestr(name,data)
            payload.write_bytes(buffer.getvalue())
            metadata=json.loads(path.read_text())
            metadata['payload']['sha256']=hashlib.sha256(buffer.getvalue()).hexdigest()
            path.write_text(json.dumps(metadata))
            # Ordinary NumPy silently chooses the last member, which matches
            # all declared array hashes; the contract must reject ambiguity.
            with np.load(payload,allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['vertices_world'],obj.vertices_world)
            with self.assertRaisesRegex(ValueError,'Duplicate transport NPZ fields'):
                FreeClothingTransport.load(path)

    def test_complete_chain_gauge_invariance_and_missing_warp_control(self):
        K=np.array([[900.,7.,321.],[0.,850.,199.],[0.,0.,1.]])
        rotations=np.stack([Q,np.eye(3),Q180])
        translations=np.array([[.4,-.6,.7],[-.2,.1,.4],[.8,.2,-.3]])
        local_vertices=np.stack([2*V@Q.T+TRANS,V,V@Q180.T+np.array([0.,0.,10.])])
        values=[];negatives=[]
        for gauge in [1.,2.]:
            world=np.stack([(v-gauge*t)@r for v,t,r in zip(local_vertices,translations,rotations)])
            obj=construct(world)
            xr=(XYZ-gauge*translations[1])@rotations[1]
            r,t=rotations[0],gauge*translations[0]
            cam=-t@r
            p=run_np(obj,xyz=xr,camera=cam)
            xc=p['centers']@r.T+t
            covc=r@p['covariance']@r.T
            values.append(dict(x=xc,cov=covc,d=p['reference_view_directions'],uv=project(xc,K)))
            negatives.append(project(xr@r.T+t,K))
        for key in values[0]:
            np.testing.assert_allclose(values[0][key],values[1][key],atol=2e-12,rtol=0)
        expected=2*XYZ@Q.T+TRANS
        np.testing.assert_allclose(values[0]['x'],expected,atol=2e-14,rtol=0)
        METRICS['consistent_gauge_pixel_difference']=float(np.max(np.linalg.norm(values[0]['uv']-values[1]['uv'],axis=1)))
        METRICS['omitted_warp_gauge_pixel_difference']=float(np.max(np.linalg.norm(negatives[0]-negatives[1],axis=1)))
        self.assertGreater(METRICS['omitted_warp_gauge_pixel_difference'],1)

    def test_common_world_scale_preserves_pixels_and_projected_covariance(self):
        K=np.array([[900.,7.,321.],[0.,850.,199.],[0.,0.,1.]])
        camera_R=dense_rotation();camera_t=-camera_R@CAM
        vertices=np.stack([2*V@Q.T+TRANS,V,V@Q180.T])
        results=[]
        for gauge in (1.,2.,.25):
            obj=construct(vertices*gauge)
            result=run_np(obj,xyz=XYZ*gauge,cov=COV*gauge**2,camera=CAM*gauge)
            camera_points=result['centers']@camera_R.T+camera_t*gauge
            camera_covariance=camera_R@result['covariance']@camera_R.T
            homogeneous=camera_points@K.T
            jacobian=(K[None,:2,:]*homogeneous[:,2,None,None]-homogeneous[:,:2,None]*K[None,2:3,:])/homogeneous[:,2,None,None]**2
            image_covariance=jacobian@camera_covariance@np.swapaxes(jacobian,-1,-2)
            results.append(dict(uv=project(camera_points,K),image_covariance=image_covariance,
                                view=result['reference_view_directions']))
            np.testing.assert_allclose(result['centers'],gauge*(2*XYZ@Q.T+TRANS),atol=4e-14,rtol=0)
        for result in results[1:]:
            np.testing.assert_allclose(result['uv'],results[0]['uv'],atol=2e-12,rtol=0)
            np.testing.assert_allclose(result['image_covariance'],results[0]['image_covariance'],atol=2e-10,rtol=0)
            np.testing.assert_allclose(result['view'],results[0]['view'],atol=2e-14,rtol=0)
        METRICS['common_world_scale_pixel_max_difference']=max(float(abs(r['uv']-results[0]['uv']).max()) for r in results)
        METRICS['common_world_scale_projected_covariance_max_difference']=max(float(abs(r['image_covariance']-results[0]['image_covariance']).max()) for r in results)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransportControl))
    seconds=time.monotonic()-start
    data=dict(tests_run=result.testsRun,passed=result.wasSuccessful() and seconds<120,failures=len(result.failures),errors=len(result.errors),seconds=seconds,
              metrics=METRICS,model_forwards=0,GPU_jobs=0,real_fit=0,gaussian_steps=0,physical_quality_accepted=False,
              source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/shuolee_portrait/free_clothing_transport.py',Path(__file__)]})
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    sys.exit(0 if data['passed'] else 1)
