"""CPU-only original-prior provenance and upstream control-flow checks.

Synthetic tensors replace the network and body model; no weights or video
decoder are loaded and no physical alignment or reconstruction is accepted.
"""
import ast
import copy
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shuolee_portrait import prior, runtime as rt


def upstream_control():
    """Execute the vendored forward's branch logic with synthetic CPU kernels."""
    source = ROOT / 'third_party/gvhmr/hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Pipeline')
    forward = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward'))
    calls = dict(network=0, static=0, ik=0)
    def tensor(width, value=0.):
        return torch.full((1, 3, width), value, dtype=torch.float32, device='cpu')
    decoded = dict(body_pose=tensor(63, 1.), betas=tensor(10, .2),
                   global_orient=tensor(3), global_orient_gv=tensor(3),
                   local_transl_vel=tensor(3))
    def network(**kwargs):
        calls['network'] += 1
        return dict(pred_x=tensor(1), pred_cam=tensor(3), static_conf_logits=tensor(6))
    def static(outputs, decoder):
        calls['static'] += 1
        return tensor(3, 8.)
    def ik(outputs, decoder):
        calls['ik'] += 1
        return outputs['decode_dict']['body_pose'] + 5.
    namespace = dict(compute_bbox_info_bedlam=lambda *args: tensor(3),
                     compute_transl_full_cam=lambda *args: tensor(3, 2.),
                     get_smpl_params_w_Rt_v2=lambda **kwargs: dict(global_orient=tensor(3), transl=tensor(3)),
                     pp_static_joint=static, pp_static_joint_cam=static, process_ik=ik)
    module = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    exec(compile(module, str(source), 'exec'), namespace)
    pipeline = SimpleNamespace(args=SimpleNamespace(normalize_cam_angvel=False),
                               denoiser3d=network,
                               endecoder=SimpleNamespace(decode=lambda values: decoded))
    pipeline.forward = MethodType(namespace['forward'], pipeline)
    batch = dict(length=torch.tensor([3], device='cpu'), bbx_xys=tensor(3),
                 K_fullimg=torch.eye(3, device='cpu').expand(1, 3, 3, 3),
                 cam_angvel=tensor(6), obs=tensor(3), f_imgseq=tensor(4))
    body = lambda **params: SimpleNamespace(joints=params['transl'][:, None, :].expand(-1, 2, -1))
    return pipeline, batch, body, calls


class OriginalPriorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'inference').mkdir()
        self.video = self.root / 'uploaded.mp4'
        self.video.write_bytes(b'synthetic video identity; not decoded')
        self.rows = []
        for i in range(3):
            image = self.root / f'frame-{i}.png'
            Image.new('RGB', (24, 32), (i, 40, 80)).save(image)
            E = np.eye(4); E[0, 3] = i * .1
            self.rows.append(dict(frame=f'frame_{i:06}', source_frame_index=i * 5,
                                  official_prediction_slot=i, time_seconds=i / 6,
                                  image=str(image), source_sha256=rt.sha(image), wh=[24, 32],
                                  K=[[30., 0., 12.5], [0., 31., 16.5], [0., 0., 1.]],
                                  w2c=E.tolist(), initial_box_xyxy=[2., 3., 20., 29.]))
        rt.save(self.root / 'run.json', dict(video=str(self.video), video_sha256=rt.sha(self.video)))
        self.cfg = dict(frame_stride=5, pixel_center_convention='renderer_corner_to_detector_integer')
        self.save_inputs()

    def save_inputs(self, **extra):
        rt.save(self.root / 'data/inputs.json', dict(records=self.rows, **extra))

    def binding(self):
        return prior._fresh_source_binding(self.root, self.rows, self.cfg)

    def prediction(self, postprocess=False):
        pipeline, batch, body, calls = upstream_control()
        result = prior._forward_prediction(pipeline, batch, postprocess=postprocess)
        return prior._prediction_arrays(result, body, 3), calls

    def test_original_uses_one_upstream_forward_without_ik_or_static_postprocess(self):
        arrays, calls = self.prediction()
        self.assertEqual(calls, dict(network=1, static=0, ik=0))
        np.testing.assert_array_equal(arrays['incam_body_pose'], np.ones((3, 63)))
        np.testing.assert_array_equal(arrays['incam_transl'], np.full((3, 3), 2.))

    def test_legacy_default_really_uses_upstream_ik(self):
        pipeline, batch, body, calls = upstream_control()
        result = prior._forward_prediction(pipeline, batch)
        arrays = prior._prediction_arrays(result, body, 3)
        self.assertEqual(calls, dict(network=1, static=1, ik=1))
        np.testing.assert_array_equal(arrays['incam_body_pose'], np.full((3, 63), 6.))
        self.assertIs(inspect.signature(prior.infer).parameters['postprocess'].default, True)

    def test_original_npz_has_only_incam_arrays_and_source_receipt(self):
        binding = self.binding()
        arrays, _ = self.prediction()
        report = prior._save_prediction(self.root, self.rows, self.cfg, arrays,
                                        postprocess=False, source_binding=binding)
        path = self.root / 'inference/prediction-original.npz'
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), {'incam_body_pose', 'incam_betas',
                'incam_global_orient', 'incam_transl', 'incam_joints'})
            np.testing.assert_array_equal(saved['incam_body_pose'], arrays['incam_body_pose'])
        self.assertEqual(report, rt.read(self.root / 'inference/original-receipt.json'))
        self.assertEqual(report['prediction_sha256'], rt.sha(path))
        self.assertEqual(report['inputs_sha256'], rt.sha(self.root / 'data/inputs.json'))
        self.assertEqual(report['video_sha256'], rt.sha(self.video))
        self.assertEqual(report['frame_ids'], [row['frame'] for row in self.rows])
        self.assertEqual(report['source_frames'], [0, 5, 10])
        self.assertEqual(report['image_sha256'], [rt.sha(row['image']) for row in self.rows])
        self.assertEqual(report['prediction_variant'], 'original_incam')
        for key in ('postprocessing_applied', 'previous_person_parameters_loaded',
                    'cached_prediction_loaded', 'quality_accepted', 'physical_calibration_accepted',
                    'training_started'):
            self.assertIs(report[key], False)
        self.assertIs(report['fresh_inference'], True)
        self.assertIs(report['from_this_video'], True)
        self.assertFalse((self.root / 'inference/prediction-postprocessed.npz').exists())

    def test_legacy_output_paths_and_receipt_are_preserved(self):
        arrays, _ = self.prediction(postprocess=True)
        report = prior._save_prediction(self.root, self.rows, self.cfg, arrays)
        self.assertTrue((self.root / 'inference/prediction-postprocessed.npz').is_file())
        self.assertTrue((self.root / 'inference/receipt.json').is_file())
        self.assertEqual(report['prediction_variant'], 'postprocessed_incam')
        self.assertTrue(report['postprocessing_applied'])
        self.assertEqual(report['frames'], 3)
        self.assertFalse((self.root / 'inference/prediction-original.npz').exists())

    def test_camera_pixel_convention_is_bound_without_mutating_rows(self):
        before = copy.deepcopy(self.rows)
        binding = self.binding()
        self.assertEqual(self.rows, before)
        self.assertEqual(binding['renderer_K'], [row['K'] for row in self.rows])
        self.assertEqual(binding['w2c'], [row['w2c'] for row in self.rows])
        self.assertEqual(binding['prior_K'][0][0][2], 12.)
        self.assertEqual(binding['prior_K'][0][1][2], 16.)

    def test_changed_video_is_rejected(self):
        self.video.write_bytes(b'different uploaded video')
        with self.assertRaisesRegex(ValueError, 'video'):
            self.binding()

    def test_other_video_manifest_or_row_is_rejected(self):
        self.save_inputs(video_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'different video'):
            self.binding()
        self.rows[0]['video_sha256'] = '0' * 64
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'different video'):
            self.binding()

    def test_changed_source_image_is_rejected(self):
        Image.new('RGB', (24, 32), 'red').save(self.rows[1]['image'])
        with self.assertRaisesRegex(ValueError, 'source image'):
            self.binding()

    def test_duplicate_ids_wrong_order_and_prediction_slot_are_rejected(self):
        original = copy.deepcopy(self.rows)
        for field, value in [('frame', self.rows[0]['frame']), ('source_frame_index', 0),
                             ('official_prediction_slot', 0), ('time_seconds', 0.)]:
            self.rows = copy.deepcopy(original)
            self.rows[1][field] = value
            self.save_inputs()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.binding()

    def test_source_binding_cannot_change_during_inference(self):
        binding = self.binding()
        arrays, _ = self.prediction()
        self.rows[1]['K'][0][2] += 1.
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            prior._save_prediction(self.root, self.rows, self.cfg, arrays,
                                   postprocess=False, source_binding=binding)
        self.assertFalse((self.root / 'inference/prediction-original.npz').exists())

    def test_original_requires_preinference_source_binding(self):
        arrays, _ = self.prediction()
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            prior._save_prediction(self.root, self.rows, self.cfg, arrays, postprocess=False)

    def test_existing_prediction_is_never_overwritten(self):
        binding = self.binding()
        arrays, _ = self.prediction()
        prior._save_prediction(self.root, self.rows, self.cfg, arrays,
                               postprocess=False, source_binding=binding)
        path = self.root / 'inference/prediction-original.npz'
        digest = rt.sha(path)
        with self.assertRaises(FileExistsError):
            prior._save_prediction(self.root, self.rows, self.cfg, arrays,
                                   postprocess=False, source_binding=binding)
        self.assertEqual(rt.sha(path), digest)

    def test_nonboolean_mode_is_rejected_before_inference(self):
        pipeline = Mock()
        for flag in (0, 1, None, 'false'):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                prior._forward_prediction(pipeline, {}, postprocess=flag)
            with self.assertRaises(TypeError):
                prior.infer(self.root, self.cfg, postprocess=flag)
        pipeline.forward.assert_not_called()

    def test_existing_inference_directory_blocks_fresh_model_import(self):
        with self.assertRaises(FileExistsError):
            prior.infer(self.root, self.cfg, postprocess=False)

    def test_bad_incam_shape_and_nonfinite_prediction_are_rejected(self):
        pipeline, batch, body, _ = upstream_control()
        result = prior._forward_prediction(pipeline, batch, postprocess=False)
        result['pred_smpl_params_incam']['transl'] = torch.zeros((1, 2, 3), device='cpu')
        with self.assertRaisesRegex(ValueError, 'shape'):
            prior._prediction_arrays(result, body, 3)
        result['pred_smpl_params_incam']['transl'] = torch.full((1, 3, 3), float('nan'), device='cpu')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            prior._prediction_arrays(result, body, 3)

    def test_arrays_do_not_alias_upstream_prediction_tensors(self):
        pipeline, batch, body, _ = upstream_control()
        result = prior._forward_prediction(pipeline, batch, postprocess=False)
        arrays = prior._prediction_arrays(result, body, 3)
        result['pred_smpl_params_incam']['body_pose'].add_(10.)
        np.testing.assert_array_equal(arrays['incam_body_pose'], np.ones((3, 63)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

