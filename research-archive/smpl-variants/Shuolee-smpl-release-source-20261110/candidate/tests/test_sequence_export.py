"""Independent synthetic CPU readers for standard PLY/reference-camera export."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from shuolee_portrait.frame_pose_contract import load_frame_pose_bundle
from shuolee_portrait.sequence_export import export_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def rotation_wxyz(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class SequenceExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rows = []
        self.K = np.array([[51., 2.25, 13.2], [0., 47., 10.3], [0., 0., 1.]])
        self.vertices = np.array([[0., 0., 3.], [.5, .1, 3.], [0., .5, 3.]])
        for i, name in enumerate(('moving', 'reference')):
            image, mask = self.root/(name+'.png'), self.root/(name+'-mask.png')
            Image.new('RGB', (32, 24), (20, 50+i, 90)).save(image)
            Image.new('L', (32, 24), 255).save(mask)
            E = np.eye(4)
            angle = .31 + .2*i
            E[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
            E[:3, 3] = [.2, -.1, .3+i*.1]
            rows.append(dict(frame=name, source_frame=4+13*i, sequence_id='synthetic-export', video_sha256='c'*64,
                wh=[32, 24], image=image.name, source_sha256=sha(image), mask=mask.name, mask_sha256=sha(mask),
                K_integer_centers=self.K.tolist(), w2c=E.tolist()))
        write(self.root/'frames.json', dict(records=rows))
        np.savez(self.root/'poses.npz', frame_ids=np.array([r['frame'] for r in rows]),
            source_frames=np.array([r['source_frame'] for r in rows]), image_sha256=np.array([r['source_sha256'] for r in rows]),
            sequence_id=np.array('synthetic-export'), video_sha256=np.array('c'*64),
            vertices_world=np.stack((self.vertices+[.2, 0, 0], self.vertices)), faces=np.array([[0, 1, 2]]),
            shared_betas=np.zeros((1, 10)), K_integer_centers=np.stack([self.K]*2), w2c=np.array([r['w2c'] for r in rows]))
        write(self.root/'bundle.json', dict(schema='shuolee-frame-pose/1', purpose='synthetic_contract_test',
            sequence_id='synthetic-export', video_sha256='c'*64, coordinate_system='opencv_world_meters',
            pixel_domain='rectified_pinhole', pixel_convention='integer_centers', reference_frame_id='reference',
            frames=dict(path='frames.json', rows_key='records', sha256=sha(self.root/'frames.json')),
            poses=dict(path='poses.npz', sha256=sha(self.root/'poses.npz'))))
        self.bundle = load_frame_pose_bundle(self.root/'bundle.json')
        self.state = self.make_state(3)

    def tearDown(self):
        self.temporary.cleanup()

    def make_state(self, degree):
        quats = np.array([[1., 2., 3., 4.], [.8, -.3, .2, .1]], dtype=np.float64)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        ids = ['free-second', 'free-first']
        provenance = dict(bundle=self.bundle.report(), reference_frame_id='reference', gaussian_ids=ids)
        binding = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return dict(schema='shuolee-framewise-native-training/1/reference', frame_id='reference', step=3,
            coordinate_system='opencv_world_meters', posed_clouds_averaged=False, gaussian_ids=ids,
            binding_sha256=binding, provenance=provenance, training_quality_accepted=False,
            parameters=dict(xyz=np.array([[-.2, .13, 3.1], [.7, -.25, 4.2]]), quats=quats,
                scales=np.array([[.13, .08, .04], [.21, .17, .09]]), opacity=np.array([.23, .81]),
                sh=(np.arange(2*3*(degree+1)**2).reshape(2, 3, (degree+1)**2)-11)/137.))

    def export(self, name='export', state=None, bundle=None):
        return export_reference(self.state if state is None else state,
            self.bundle if bundle is None else bundle, self.root/name)

    def test_all_degrees_standard_binary_fields_and_independent_covariance_reader(self):
        for degree in range(4):
            with self.subTest(degree=degree):
                state = self.make_state(degree)
                receipt = self.export('degree-'+str(degree), state)
                path = self.root/('degree-'+str(degree))/'point_cloud.ply'
                with path.open('rb') as stream:
                    self.assertEqual(stream.readline(), b'ply\n')
                    self.assertEqual(stream.readline(), b'format binary_little_endian 1.0\n')
                data = PlyData.read(path, mmap=False)['vertex'].data
                expected_names = (['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+
                    [f'f_rest_{i}' for i in range(3*((degree+1)**2-1))]+['opacity']+
                    [f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)])
                self.assertEqual(list(data.dtype.names), expected_names)
                self.assertTrue(all(data.dtype[name].itemsize == 4 for name in expected_names))
                p = state['parameters']
                np.testing.assert_array_equal(np.column_stack([data[k] for k in ('x','y','z')]), p['xyz'].astype(np.float32))
                np.testing.assert_array_equal(np.column_stack([data[k] for k in ('nx','ny','nz')]), np.zeros((2,3)))
                dc = np.column_stack([data[f'f_dc_{i}'] for i in range(3)])[:,:,None]
                rest = np.column_stack([data[f'f_rest_{i}'] for i in range(3*((degree+1)**2-1))]).reshape(2,3,-1) if degree else np.zeros((2,3,0))
                np.testing.assert_array_equal(np.concatenate((dc,rest),axis=2), p['sh'].astype(np.float32))
                scales = np.exp(np.column_stack([data[f'scale_{i}'] for i in range(3)]).astype(float))
                opacity = 1/(1+np.exp(-data['opacity'].astype(float)))
                np.testing.assert_allclose(scales, p['scales'], rtol=2e-7, atol=0)
                np.testing.assert_allclose(opacity, p['opacity'], rtol=0, atol=2e-8)
                quats = np.column_stack([data[f'rot_{i}'] for i in range(4)]).astype(float)
                for i in range(2):
                    R, original_R = rotation_wxyz(quats[i]), rotation_wxyz(p['quats'][i])
                    np.testing.assert_allclose(R@np.diag(scales[i]**2)@R.T,
                        original_R@np.diag(p['scales'][i]**2)@original_R.T, rtol=3e-6, atol=3e-9)
                self.assertFalse(receipt['quality_accepted'])
                self.assertFalse(receipt['coordinate_transform_applied'])
                self.assertEqual(receipt['sh_degree'], degree)
                self.assertEqual(receipt['ply']['sha256'], sha(path))
                self.assertEqual(set(x.name for x in path.parent.iterdir()), {'point_cloud.ply','cameras.json','result.json'})

    def test_cameras_preserve_full_K_and_same_world_projection(self):
        self.export()
        cameras = json.loads((self.root/'export/cameras.json').read_text())
        self.assertIsInstance(cameras, list)
        points = np.array([[.1, -.2, 3.], [.4, .7, 4.]])
        for camera in cameras:
            frame = self.bundle.frame(camera['frame_id'])
            E, K = np.array(camera['w2c']), np.array(camera['K_integer_centers'])
            np.testing.assert_array_equal(E, frame.w2c)
            np.testing.assert_array_equal(K, self.K)
            np.testing.assert_allclose(E@np.array(camera['c2w']), np.eye(4), atol=2e-15)
            xyz1 = points@E[:3,:3].T+E[:3,3]
            pixels1 = xyz1@K.T; pixels1 = pixels1[:,:2]/pixels1[:,2,None]
            xyz2 = (points-camera['position'])@np.array(camera['rotation'])
            pixels2 = xyz2@np.array(camera['K_renderer_corners']).T
            pixels2 = pixels2[:,:2]/pixels2[:,2,None]-.5
            np.testing.assert_allclose(pixels1,pixels2,atol=2e-14)
            self.assertEqual(camera['reference_frame_id'],'reference')
            self.assertEqual(camera['source_frame'],frame.source_frame)
            self.assertFalse(camera['static_reference_multiview_ground_truth'])
        self.assertFalse(cameras[0]['is_reference_frame'])
        self.assertTrue(cameras[1]['is_reference_frame'])

    def test_reference_identity_zero_steps_and_cross_sequence_fail_before_output(self):
        for update in ({'frame_id':'moving'},{'step':0},{'step':True},{'step':-1}, {'posed_clouds_averaged':True}):
            with self.subTest(update=update):
                with self.assertRaises(ValueError): self.export(state=dict(self.state,**update))
                self.assertFalse((self.root/'export').exists())
        with self.assertRaises(ValueError): self.export(bundle=replace(self.bundle,sequence_id='another'))
        self.assertFalse((self.root/'export').exists())

    def test_invalid_physical_parameters_and_float32_overflow_fail_before_output(self):
        cases = [('scales',0.),('scales',-1.),('scales',1e300),('scales',1e-300),
                 ('opacity',0.),('opacity',1.),('opacity',1-1e-12),('quats',0.),('quats',.6),
                 ('xyz',np.nan),('xyz',1e100),('sh',np.inf)]
        for key,value in cases:
            with self.subTest(key=key,value=value):
                state=copy.deepcopy(self.state);state['parameters'][key].fill(value)
                with self.assertRaises(ValueError): self.export(state=state)
                self.assertFalse((self.root/'export').exists())

    def test_changed_source_and_provenance_cannot_be_exported(self):
        state=copy.deepcopy(self.state);state['provenance']['reference_frame_id']='moving'
        with self.assertRaisesRegex(ValueError,'binding hash'): self.export(state=state)
        with self.bundle.reference.mask.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Bound input changed'):self.export()

    def test_corrupt_serialized_field_prevents_result_receipt(self):
        original=PlyData.write
        def corrupt(data, stream):
            original(data,stream);stream.flush()
            with Path(stream.name).open('r+b') as handle:
                while handle.readline()!=b'end_header\n': pass
                handle.write(struct.pack('<f',9.5))
        with patch.object(PlyData,'write',corrupt):
            with self.assertRaisesRegex(ValueError,'field roundtrip failed'):self.export()
        self.assertFalse((self.root/'export/result.json').exists())

    def test_existing_export_is_preserved(self):
        self.export(); before=sha(self.root/'export/point_cloud.ply')
        with self.assertRaisesRegex(ValueError,'must be new'):self.export()
        self.assertEqual(before,sha(self.root/'export/point_cloud.ply'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SequenceExportTests))
    report=dict(state='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-started,
        GPU_tasks=0,model_forwards=0,training_steps=0,quality_accepted=False,
        source_sha256=sha(ROOT/'src/shuolee_portrait/sequence_export.py'),tests_sha256=sha(__file__),
        failure_details=[str(test)+'\n'+detail for test,detail in result.failures+result.errors])
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2));raise SystemExit(not result.wasSuccessful())
