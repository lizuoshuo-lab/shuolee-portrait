"""Bootstrap/profile controls: tiny files, no model download or CUDA work."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'scripts')]
from shuolee_portrait import official_deployment as deployment, runtime as rt
import prepare_assets


class OfficialDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit, real_pins = deployment.official_pin()
        self.payload = {name: ('synthetic '+name).encode() for name in real_pins}
        self.pins = {k: hashlib.sha256(v).hexdigest() for k, v in self.payload.items()}
        self.pin_patch = patch.object(deployment, 'official_pin', return_value=(self.commit, self.pins))
        self.pin_patch.start(); self.addCleanup(self.pin_patch.stop)
        self.source = self.root/'authorized source'
        self.write_source(self.source)

    def write_source(self, destination):
        for name, data in self.payload.items():
            path = destination/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)

    def test_bootstrap_reads_real_pin_without_numpy_or_torch(self):
        code = ('import sys,json;sys.path.insert(0,'+repr(str(PROJECT/'src'))+');'
            'from shuolee_portrait.official_deployment import official_pin;'
            'commit,pins=official_pin();print(json.dumps([commit,len(pins),"numpy" in sys.modules,"torch" in sys.modules]))')
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [self.commit, 7, False, False])

    def test_reuse_checks_all_sources_without_git_or_download(self):
        with patch.object(deployment.subprocess, 'run') as network:
            metadata = deployment.prepare_source(self.root/'runtime', explicit=self.source)
        network.assert_not_called()
        self.assertEqual(metadata['files_sha256'], self.pins)
        self.assertEqual(metadata['source_root'], str(self.source.resolve()))

    def test_changed_explicit_source_is_rejected_and_preserved(self):
        path = self.source/next(iter(self.payload)); path.write_bytes(b'changed')
        with patch.object(deployment.subprocess, 'run') as git:
            with self.assertRaisesRegex(ValueError, 'differs'):
                deployment.prepare_source(self.root, explicit=self.source)
        git.assert_not_called(); self.assertEqual(path.read_bytes(), b'changed')

    def test_new_source_fetch_is_pinned_then_validated_before_publication(self):
        calls = []
        def git(command, **kwargs):
            calls.append(command)
            if 'checkout' in command: self.write_source(Path(command[2]))
        with patch.object(deployment.subprocess, 'run', side_effect=git), \
             patch.object(deployment.subprocess, 'check_output', return_value=self.commit+'\n'):
            result = deployment.prepare_source(self.root/'runtime')
        self.assertTrue(result['source_hashes_verified'])
        self.assertTrue(any(c[3:] == ['fetch', '--depth', '1', 'origin', self.commit] for c in calls))
        self.assertTrue(any(c[3:] == ['remote', 'add', 'origin', deployment.REPOSITORY] for c in calls))
        self.assertEqual(list((self.root/'runtime/sources').glob('.smplify-x-*')), [])

    def test_failed_source_validation_never_publishes_or_overwrites(self):
        with patch.object(deployment.subprocess, 'run'), \
             patch.object(deployment.subprocess, 'check_output', return_value=self.commit+'\n'):
            with self.assertRaises(ValueError): deployment.prepare_source(self.root/'runtime')
        self.assertFalse((self.root/'runtime/sources/smplify-x').exists())
        occupied=self.root/'runtime/sources/smplify-x'; occupied.mkdir()
        (occupied/'keeper').write_bytes(b'keep')
        with patch.object(deployment.subprocess, 'run') as git, self.assertRaises(ValueError):
            deployment.prepare_source(self.root/'runtime')
        git.assert_not_called(); self.assertEqual((occupied/'keeper').read_bytes(), b'keep')

    def test_seven_assets_reused_and_legacy_files_not_required_or_removed(self):
        assets = self.root/'assets'; rows = []
        for name in deployment.ASSET_PATHS:
            raw=('asset fixture '+name).encode(); path=assets/name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            rows.append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        old = assets/'gvhmr/retained'; old.parent.mkdir(); old.write_bytes(b'keep old assets')
        rows.append(dict(path='gvhmr/unneeded.ckpt', bytes=999, sha256='missing', download={'url':'invalid'}))
        manifest=dict(files=rows)
        with patch.object(prepare_assets, 'download') as download, patch.object(prepare_assets, 'copy_verified') as copying:
            actual=prepare_assets.prepare(self.root/'runtime', [assets], manifest, profile=deployment.PROFILE)
        self.assertEqual(actual, assets); download.assert_not_called(); copying.assert_not_called()
        self.assertEqual(old.read_bytes(), b'keep old assets')
        self.assertIs(deployment.selected_manifest(manifest, 'legacy'), manifest)
        with self.assertRaises(ValueError): deployment.selected_manifest(dict(files=rows[:2]), deployment.PROFILE)

    def test_official_doctor_uses_selected_assets_and_no_legacy_prior(self):
        with patch.object(rt, 'configure'), patch.object(rt, 'verify_assets', return_value={'files':7}) as assets, \
             patch.object(deployment, 'gpu_dependencies', return_value={'CPU_test_stub':True}), \
             patch.object(deployment.shutil, 'which', return_value='/fixture/ffmpeg'), \
             patch('shuolee_portrait.cli.doctor') as legacy:
            result=deployment.doctor(self.root/'assets', self.source)
        assets.assert_called_once_with(deployment.ASSET_PATHS); legacy.assert_not_called()
        self.assertFalse(result['gvhmr_required']); self.assertFalse(result['training_started'])
        self.assertFalse(any('hmr4d' in name or 'gvhmr' in name for name in deployment.official_module_names()))

    def test_configure_records_default_source_without_choosing_shape_policy(self):
        output=self.root/'runtime/queue/deployment.json'
        with patch.object(deployment, 'doctor', return_value={'CPU_test_stub':True}) as doctor:
            profile=deployment.configure_deployment(self.root/'assets',self.root/'runtime/runs',output,self.source)
        self.assertEqual(rt.read(output), profile)
        self.assertEqual(profile['official_smplifyx']['source_root'],str(self.source.resolve()))
        self.assertNotIn('shape_policy',profile['official_smplifyx'])
        doctor.assert_called_once()

    def test_default_source_and_explicit_override_preserve_shape_policy(self):
        profile=dict(official_smplifyx={'source_root':'/deployed/pinned/source'})
        options=dict(shape_policy={'mode':'fit_reference'})
        actual=deployment.fitting_options(options,profile)
        self.assertEqual(actual,dict(options,source_root='/deployed/pinned/source'))
        self.assertNotIn('source_root',options)
        override=dict(options,source_root='/explicit/source')
        self.assertEqual(deployment.fitting_options(override,profile),override)
        self.assertIsNone(deployment.fitting_options(None,profile))


if __name__ == '__main__': unittest.main()
