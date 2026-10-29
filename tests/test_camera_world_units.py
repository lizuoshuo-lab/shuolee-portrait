"""Deterministic CPU unit integration; no SMPL-X/VPoser/NN forward or fitting."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.camera_world_units import normalize_camera_w2c, validate_camera_length_contract
from shuolee_portrait.sequence_fitting import SharedIdentitySequence, camera_to_world


def declaration(unit='sfm_units', factor=.4, basis='synthetic_known'):
    return dict(translation_unit=unit, meters_per_camera_unit=factor,
                basis=basis, evidence='analytic metric cameras and 0.4 metres per synthetic SfM unit')


def scene():
    # Independently specified physical scene; changing unit affects only SfM t.
    points = np.array([[-.3,-.8,4.], [.25,-.7,4.2], [-.2,.6,4.1],
                       [.3,.7,3.8], [0.,0.,3.7]], dtype=np.float64)
    centers = np.array([[0.,0.,0.], [.8,.15,0.], [-.4,.2,.1]])
    E = np.tile(np.eye(4),(3,1,1))
    for i, theta in enumerate([0.,.17,-.23]):
        c,s = np.cos(theta),np.sin(theta)
        E[i,:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
        E[i,:3,3]=-E[i,:3,:3]@centers[i]
    camera = np.einsum('nij,vj->nvi',E[:,:3,:3],points)+E[:,None,:3,3]
    sfm=E.copy();sfm[:,:3,3]/=.4
    K=np.array([[1317.3,4.2,1353.4],[0.,1299.8,759.2],[0.,0.,1.]])
    return points, E, sfm, camera, K


def project(points,K):
    q=points@K.T
    return q[...,:2]/q[...,2:]


def analytic_control():
    points,E,sfm,camera,K=scene()
    converted=normalize_camera_w2c(sfm,declaration())
    tensor=torch.tensor(camera,dtype=torch.float64)
    correct=camera_to_world(tensor,torch.tensor(converted.w2c_meters.copy()),translation_unit='meters').numpy()
    # Replay the old arithmetic explicitly, without declaring these mixed values metres.
    mixed=np.einsum('nvi,nij->nvj',camera-sfm[:,None,:3,3],sfm[:,:3,:3])
    own=np.einsum('nij,nvj->nvi',sfm[:,:3,:3],mixed)+sfm[:,None,:3,3]
    # Render the reference mixed geometry into every camera: own-frame equality
    # does not imply a single correct shared world representation.
    wrong_cross=np.einsum('nij,vj->nvi',sfm[:,:3,:3],mixed[0])+sfm[:,None,:3,3]
    good_cross=np.einsum('nij,vj->nvi',E[:,:3,:3],correct[0])+E[:,None,:3,3]
    return dict(known_metres_per_sfm_unit=.4,
        correct_world_max_error_m=float(np.max(np.linalg.norm(correct-points[None],axis=-1))),
        old_mixed_world_max_numeric_disagreement=float(np.max(np.linalg.norm(mixed-mixed[0],axis=-1))),
        old_own_frame_max_px=float(np.max(np.linalg.norm(project(own,K)-project(camera,K),axis=-1))),
        old_shared_reference_cross_view_max_px=float(np.max(np.linalg.norm(project(wrong_cross,K)-project(camera,K),axis=-1))),
        correct_shared_reference_cross_view_max_px=float(np.max(np.linalg.norm(project(good_cross,K)-project(camera,K),axis=-1))),
        real_person_scale_estimated=False)


class Placeholder(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('marker',torch.zeros(1));self.num_pca_comps=12
    def forward(self,*args,**kwargs):
        raise AssertionError('No model forward is allowed in this CPU unit check.')


def sequence(contract=None, *, dtype=torch.float64, input_K=None):
    _,_,sfm,_,K=scene()
    initial={k:np.zeros((3,n)) for k,n in [('global_orient',3),('transl',3),('latent',32),('jaw_pose',3),('expression',10)]}
    return SharedIdentitySequence(Placeholder(),Placeholder(),['source:0','source:155','source:310'],
        np.tile(K,(3,1,1)) if input_K is None else input_K,sfm,initial,
        camera_length_contract=contract,dtype=dtype)


def camera_consumers(model):
    out=SimpleNamespace(joints=torch.tensor(scene()[3],dtype=torch.float64),
                        vertices=torch.tensor(scene()[3],dtype=torch.float64))
    return dict(forward=model.forward_state,
                project=lambda:model.project(out),
                world_vertices=lambda:model.world_vertices(out),
                snapshot=lambda:model.native_snapshot('source:0'),
                extra_state=model.get_extra_state,
                checkpoint=model.state_dict)


class UnitChecks(unittest.TestCase):
    def test_undeclared_rejected_before_model_access(self):
        with self.assertRaisesRegex(ValueError,'camera_length_contract'):
            SharedIdentitySequence(None,None,['a'],np.eye(3)[None],np.eye(4)[None],{})

    def test_explicit_function_unit_required(self):
        x=torch.zeros((1,2,3));e=torch.eye(4)[None]
        for unit in [None,'sfm_units','unknown']:
            with self.subTest(unit=unit),self.assertRaises(ValueError):
                camera_to_world(x,e,translation_unit=unit)

    def test_one_common_positive_scale_only(self):
        for factor in [0.,-1.,np.nan,np.inf,True,[.4,.4,.4],np.ones(3),'.4']:
            with self.subTest(factor=str(factor)),self.assertRaises(ValueError):
                validate_camera_length_contract(declaration(factor=factor))

    def test_missing_factor_and_unknown_unit(self):
        for field in ['translation_unit','meters_per_camera_unit','basis','evidence']:
            d=declaration();del d[field]
            with self.subTest(field=field),self.assertRaises(ValueError):validate_camera_length_contract(d)
        with self.assertRaises(ValueError):validate_camera_length_contract(declaration(unit='pixels'))

    def test_provenance_is_required_not_acceptance(self):
        d=declaration();d['evidence']=' '
        with self.assertRaises(ValueError):validate_camera_length_contract(d)
        d=declaration();d['physical_camera_accepted']=True
        with self.assertRaises(ValueError):validate_camera_length_contract(d)
        r=normalize_camera_w2c(np.eye(4),declaration(basis='model_conditioned')).report()
        self.assertFalse(r['physical_camera_accepted']);self.assertFalse(r['training_admitted'])

    def test_meter_identity_is_explicit(self):
        E=scene()[1]
        r=normalize_camera_w2c(E,declaration(unit='meters',factor=1.))
        np.testing.assert_array_equal(r.w2c_meters,E)
        with self.assertRaises(ValueError):normalize_camera_w2c(E,declaration(unit='meters',factor=.4))

    def test_only_translation_changed_original_preserved(self):
        _,E,sfm,_,_=scene();r=normalize_camera_w2c(sfm,declaration())
        np.testing.assert_array_equal(r.source_w2c,sfm)
        np.testing.assert_array_equal(r.w2c_meters[:,:3,:3],sfm[:,:3,:3])
        np.testing.assert_allclose(r.w2c_meters,E,atol=1e-15,rtol=0)

    def test_unit_conversion_is_immutable_copy(self):
        d=declaration();e=scene()[2];r=normalize_camera_w2c(e,d);before=r.w2c_meters.copy()
        e[:,:3,3]=9.;d['meters_per_camera_unit']=8.
        np.testing.assert_array_equal(r.w2c_meters,before)
        with self.assertRaises(TypeError):r.contract['evidence']='new'
        with self.assertRaises(ValueError):r.w2c_meters.setflags(write=True)

    def test_invalid_extrinsics_and_overflow_rejected(self):
        for e in [np.empty((0,4,4)),np.eye(3),np.diag([-1.,1,1,1]),np.eye(4)*2]:
            with self.subTest(shape=e.shape),self.assertRaises(ValueError):normalize_camera_w2c(e,declaration())
        e=np.eye(4);e[0,3]=1e308
        with self.assertRaises(ValueError):normalize_camera_w2c(e,declaration(factor=1e308))

    def test_analytic_wrong_units_preserve_pixels_but_split_world(self):
        a=analytic_control()
        self.assertLess(a['old_own_frame_max_px'],1e-9)
        self.assertGreater(a['old_mixed_world_max_numeric_disagreement'],1.)
        self.assertGreater(a['old_shared_reference_cross_view_max_px'],20.)
        self.assertLess(a['correct_world_max_error_m'],1e-12)
        self.assertLess(a['correct_shared_reference_cross_view_max_px'],1e-9)

    def test_torch_vertex_gradient_preserved(self):
        _,E,_,camera,_=scene();x=torch.tensor(camera,requires_grad=True,dtype=torch.float64)
        y=camera_to_world(x,torch.tensor(E),translation_unit='meters')
        weights=torch.arange(y.numel(),dtype=torch.float64).reshape(y.shape)/20
        (y*weights).sum().backward()
        expected=np.einsum('nvj,nij->nvi',weights.numpy(),E[:,:3,:3])
        np.testing.assert_allclose(x.grad.numpy(),expected,atol=1e-14,rtol=0)

    def test_sequence_uses_metric_E_and_keeps_full_K(self):
        _,E,sfm,_,K=scene();m=sequence(declaration())
        np.testing.assert_allclose(m.w2c.numpy(),E,atol=1e-15,rtol=0)
        np.testing.assert_array_equal(m.source_w2c.numpy(),sfm)
        np.testing.assert_array_equal(m.K.numpy(),np.tile(K,(3,1,1)))
        self.assertEqual(m.state['betas'].shape,(1,10))

    def test_frozen_K_has_no_upstream_gradient_but_points_do(self):
        _,_,_,camera,K=scene()
        input_K=torch.tensor(np.tile(K,(3,1,1)),dtype=torch.float64,requires_grad=True)
        m=sequence(declaration(),input_K=input_K)
        points=torch.tensor(camera,dtype=torch.float64,requires_grad=True)
        m.project(SimpleNamespace(joints=points)).sum().backward()
        self.assertFalse(m.K.requires_grad)
        self.assertIsNone(m.K.grad_fn)
        self.assertIsNone(input_K.grad)
        q=camera@K.T
        expected=(K[0]+K[1])/q[...,2,None]-(q[...,0]+q[...,1])[...,None]*K[2]/q[...,2,None]**2
        np.testing.assert_allclose(points.grad.numpy(),expected,atol=1e-10,rtol=0)

    def test_runtime_camera_mutations_rejected_by_every_consumer(self):
        for key in ('K','source_w2c','w2c'):
            m=sequence(declaration())
            with torch.no_grad():getattr(m,key)[0,0,2 if key=='K' else 3]+=.1
            for name,consume in camera_consumers(m).items():
                with self.subTest(buffer=key,consumer=name),self.assertRaisesRegex(RuntimeError,'Frozen camera'):
                    consume()

    def test_runtime_frame_reordering_rejected_by_every_consumer(self):
        m=sequence(declaration());m.frame_ids=tuple(reversed(m.frame_ids))
        for name,consume in camera_consumers(m).items():
            with self.subTest(consumer=name),self.assertRaisesRegex(RuntimeError,'frame identities/order'):
                consume()

    def test_runtime_camera_requires_grad_cannot_be_enabled(self):
        for key in ('K','source_w2c','w2c'):
            m=sequence(declaration());getattr(m,key).requires_grad_(True)
            for name,consume in camera_consumers(m).items():
                with self.subTest(buffer=key,consumer=name),self.assertRaisesRegex(RuntimeError,'cannot require gradients'):
                    consume()

    def test_dtype_changes_rejected_before_any_child_or_parameter_changes(self):
        for dtype,convert in ((torch.float64,lambda m:m.float()),
                              (torch.float64,lambda m:m.half()),
                              (torch.float32,lambda m:m.double())):
            m=sequence(declaration(),dtype=dtype)
            before={name:value.detach().clone() for name,value in
                    list(m.named_buffers())+list(m.named_parameters())}
            with self.subTest(dtype=dtype),self.assertRaisesRegex(RuntimeError,'dtype is fixed at construction'):
                convert(m)
            for name,value in list(m.named_buffers())+list(m.named_parameters()):
                self.assertEqual(value.dtype,before[name].dtype,name)
                self.assertTrue(torch.equal(value,before[name]),name)
            m.load_state_dict(copy.deepcopy(m.state_dict()))

    def test_constructor_dtype_and_cpu_device_transfer_preserve_source(self):
        for dtype in (torch.float32,torch.float64):
            m=sequence(declaration(),dtype=dtype)
            source=m.source_w2c.clone();K=m.K.clone();meter=m.w2c.clone()
            self.assertIs(m.to(device='cpu'),m)
            self.assertEqual(m.source_w2c.dtype,torch.float64)
            self.assertEqual(m.K.dtype,dtype)
            self.assertTrue(torch.equal(m.source_w2c,source))
            self.assertTrue(torch.equal(m.K,K));self.assertTrue(torch.equal(m.w2c,meter))
            other=sequence(declaration(),dtype=dtype)
            other.load_state_dict(copy.deepcopy(m.state_dict()))
            self.assertTrue(torch.equal(other.source_w2c,source))

    def test_reference_snapshot_carries_scale_and_source(self):
        points,E,sfm,camera,_=scene();m=sequence(declaration())
        # Bypass neural/body execution: inject only the explicitly known fixture.
        m.forward_state=lambda:SimpleNamespace(vertices=torch.tensor(camera,dtype=torch.float64))
        snapshot=m.native_snapshot('source:155')
        self.assertEqual(snapshot['reference_frame'],'source:155')
        self.assertEqual(snapshot['frame_ids'],list(m.frame_ids))
        self.assertEqual(snapshot['coordinate_system'],'opencv_world_meters')
        self.assertFalse(snapshot['physical_camera_accepted'])
        self.assertFalse(snapshot['training_admitted'])
        self.assertFalse(snapshot['gaussian_export'])
        np.testing.assert_array_equal(snapshot['source_w2c'],sfm)
        np.testing.assert_allclose(snapshot['reference_vertices_world'],points,atol=1e-14,rtol=0)
        self.assertEqual(snapshot['camera_length_contract'],declaration())
        snapshot['frame_ids'].reverse()
        self.assertEqual(m.frame_ids,('source:0','source:155','source:310'))

    def test_checkpoint_serialized_roundtrip_preserves_units(self):
        m=sequence(declaration())
        with torch.no_grad():m.state['transl'].fill_(.31)
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
        restored=torch.load(buffer,weights_only=True,map_location='cpu')
        other=sequence(declaration());other.load_state_dict(restored)
        self.assertEqual(other.camera_length_contract,declaration())
        np.testing.assert_array_equal(other.state['transl'].detach(),m.state['transl'].detach())
        np.testing.assert_array_equal(other.w2c,m.w2c)

    def test_checkpoint_missing_metadata_even_nonstrict_rejected(self):
        m=sequence(declaration());data=copy.deepcopy(m.state_dict());data.pop('_extra_state')
        with self.assertRaises(RuntimeError):m.load_state_dict(data,strict=False)

    def test_checkpoint_conflicting_factor_does_not_relabel_E(self):
        data=sequence(declaration()).state_dict();other=sequence(declaration(factor=.7))
        original=other.w2c.clone()
        with self.assertRaises(RuntimeError):other.load_state_dict(data,strict=False)
        self.assertTrue(torch.equal(original,other.w2c))

    def test_checkpoint_frozen_camera_tampering_rejected(self):
        m=sequence(declaration())
        for key in ['K','source_w2c','w2c']:
            data=copy.deepcopy(m.state_dict());data[key][1,0,2]+=.1
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.load_state_dict(data)

    def test_nested_checkpoint_wrong_frame_order_rejected(self):
        parent=nn.Module();parent.add_module('portrait',sequence(declaration()))
        data=copy.deepcopy(parent.state_dict())
        data['portrait._extra_state']['frame_ids'].reverse()
        with self.assertRaises(RuntimeError):parent.load_state_dict(data,strict=False)

    def test_exported_metadata_cannot_mutate_instance(self):
        m=sequence(declaration());d=m.camera_length_contract;d['meters_per_camera_unit']=9
        d2=m.get_extra_state();d2['frame_ids'].reverse();d2['contract']['evidence']='wrong'
        self.assertEqual(m.camera_length_contract,declaration())
        self.assertEqual(m.frame_ids,('source:0','source:155','source:310'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path);args=ap.parse_args()
    torch.set_num_threads(1)
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(UnitChecks))
    if args.report:
        root=Path(__file__).resolve().parents[1]
        paths=[root/'src/shuolee_portrait/camera_world_units.py',root/'src/shuolee_portrait/sequence_fitting.py',Path(__file__).resolve()]
        report=dict(tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
            seconds=time.monotonic()-start,synthetic_control=analytic_control(),
            model_forwards=0,GPU_jobs=0,real_fit=0,gaussian_steps=0,physical_quality_accepted=False,
            source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
