"""Deployment/orchestration checks. Numerical CUDA stages are not emulated as evidence."""
from contextlib import nullcontext
import io
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

from test_route119 import PreparedFixture
from shuolee_portrait import route119
from shuolee_portrait import runtime as rt


class DeploymentTests(PreparedFixture):
    def test_removed_manual_commands_and_flags_are_rejected(self):
        from contextlib import redirect_stderr
        for args in (['review'], ['stage'], ['init-prepared'], ['train', 'video.mp4', '--review']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                route119.main(args)
            self.assertEqual(error.exception.code, 2)

    def prepare_fixture(self, video, root, cfg, assets):
        # Actual RGB/mask/NPZ consistency is checked; CUDA stages below are spies.
        for filename in ('inputs.json', 'training-inputs.json', 'observations.npz'):
            shutil.copy2(self.root / 'data' / filename, root / 'data' / filename)

    def test_default_command_reaches_training_and_export_without_review_files(self):
        target = self.base / 'automatic'
        visited = []
        def stage(name, root, assets):
            visited.append(name)
            if name == 'train':
                rt.save(root / 'gaussian-run/result.json', dict(steps=18000))
        with patch('shuolee_portrait.cli.gpu_lock', return_value=nullcontext()), \
             patch('shuolee_portrait.video119.prepare_video', side_effect=self.prepare_fixture), \
             patch.object(route119, 'run_stage', side_effect=stage):
            route119.main(['reconstruct', '--video', str(self.video), '--output', str(target),
                           '--assets', str(self.base / 'assets')])
        self.assertEqual(visited, ['prior', 'fit', 'fit-gate', 'surface', 'train', 'export'])
        run = rt.read(target / 'run.json')
        self.assertNotIn('review_mode', run)
        self.assertFalse(run['quality_accepted'])
        self.assertFalse((target / 'reviews').exists())


    def test_fit_numerical_failure_stops_automatic_pipeline_before_training(self):
        target = self.base / 'failed-numerical-gate'
        seen = []
        def stage(name, root, assets):
            seen.append(name)
            if name == 'fit-gate':
                raise RuntimeError('Fresh body fit failed coarse geometric gate.')
        with patch('shuolee_portrait.cli.gpu_lock', return_value=nullcontext()), \
             patch('shuolee_portrait.video119.prepare_video', side_effect=self.prepare_fixture), \
             patch.object(route119, 'run_stage', side_effect=stage):
            with self.assertRaisesRegex(RuntimeError, 'coarse geometric gate'):
                route119.reconstruct(self.video, target, self.base / 'assets')
        self.assertEqual(seen[-1], 'fit-gate')
        self.assertEqual(rt.read(target / 'status.json')['state'], 'failed')
        self.assertFalse((target / 'gaussian-run').exists())

    def test_short_training_cannot_be_exported_as_full_119_run(self):
        target = self.base / 'short'
        def stage(name, root, assets):
            if name == 'train':
                rt.save(root / 'gaussian-run/result.json', dict(steps=1000))
            self.assertNotEqual(name, 'export')
        with patch('shuolee_portrait.cli.gpu_lock', return_value=nullcontext()), \
             patch('shuolee_portrait.video119.prepare_video', side_effect=self.prepare_fixture), \
             patch.object(route119, 'run_stage', side_effect=stage):
            with self.assertRaisesRegex(RuntimeError, 'no completed 18000-step result'):
                route119.reconstruct(self.video, target, self.base / 'assets')

    def test_automatic_training_does_not_install_checkpoint_wait_callback(self):
        calls = []
        module = SimpleNamespace(train_clothed=lambda *a, **kw: calls.append(kw))
        with patch.dict(sys.modules, {'shuolee_portrait.clothed_training': module}), patch.object(rt, 'configure'):
            route119.execute_stage('train', self.root, self.base / 'assets')
        self.assertEqual(calls[0], {})
        self.assertFalse((self.root / 'reviews').exists())

    def test_one_video_command_uses_saved_assets_unique_runs_and_full_schedule(self):
        profile = self.base / 'deployment119.json'
        rt.save(profile, dict(schema='shuolee-deployment119-v1', assets=str(self.base / 'assets'),
                              runs=str(self.base / 'runs'), environment=str(Path(sys.prefix).resolve())))
        with patch.object(route119, 'reconstruct') as reconstruct:
            route119.train_video(self.video, profile)
            route119.train_video(self.video, profile)
        first, second = reconstruct.call_args_list
        self.assertEqual(first.args[2], str(self.base / 'assets'))
        self.assertEqual(first.args[1].parent, (self.base / 'runs').resolve())
        self.assertNotEqual(first.args[1], second.args[1])
        self.assertEqual(first.kwargs, {})

    def test_configuration_is_saved_only_after_successful_doctor(self):
        profile = self.base / 'deployment119.json'
        with patch('shuolee_portrait.cli.doctor', side_effect=RuntimeError('missing model')):
            with self.assertRaisesRegex(RuntimeError, 'missing model'):
                route119.configure_deployment(self.base / 'assets', self.base / 'runs', profile)
        self.assertFalse(profile.exists())

    def test_profile_cannot_bind_outputs_inside_git_checkout(self):
        with patch('shuolee_portrait.cli.doctor') as doctor:
            with self.assertRaisesRegex(ValueError, 'outside the Git checkout'):
                route119.configure_deployment(self.base / 'assets', rt.PROJECT / 'runs', self.base / 'profile.json')
        doctor.assert_not_called()
