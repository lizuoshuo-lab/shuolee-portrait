"""Real filesystem and child-process tests; CUDA reconstruction is tested separately."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import signal
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from shuolee_portrait.queue119 import VideoQueue, read, save, worker_lock

CHILD = '''
import argparse, json, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('video'); p.add_argument('--deployment'); p.add_argument('--output')
a=p.parse_args()
print('real child received '+a.video, flush=True)
time.sleep(0.05)
payload=Path(a.video).read_bytes()
if payload == b'fail': raise RuntimeError('original child error')
if payload == b'camera': raise RuntimeError('Missing real camera poses: [22, 23]')
if payload == b'killed':
 import os, signal
 os.kill(os.getpid(), signal.SIGKILL)
root=Path(a.output); (root/'export').mkdir(parents=True)
(root/'export/point_cloud.ply').write_text('orchestration fixture only')
steps=100 if payload == b'short' else 18000
(root/'result.json').write_text(json.dumps(dict(state='completed', steps=steps, format_passed=True)))
'''


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profile = self.root / 'deployment.json'
        save(self.profile, dict(schema='shuolee-deployment119-v1',
             runs=str(self.root/'runs'), assets=str(self.root/'assets'), environment=sys.prefix))
        child = self.root / 'child.py'
        child.write_text(CHILD, encoding='utf-8')
        self.command = [sys.executable, str(child)]
        self.q = self.new_queue()

    def tearDown(self):
        if self.q.child is not None:
            self.q.stop()
            self.q.child.wait(timeout=10)
            self.q.collect()
        self.temp.cleanup()

    def new_queue(self):
        return VideoQueue(self.profile, settle_seconds=30, command=self.command)

    def upload(self, name='person.mp4', content=b'video fixture'):
        path = self.q.inbox / name
        path.write_bytes(content)
        return path

    def admit(self):
        self.q.scan(0)
        self.q.scan(31)

    def finish(self):
        self.q.child.wait(timeout=10)
        self.q.collect()

    def test_partial_and_hidden_uploads_are_ignored_and_stability_is_required(self):
        self.upload('person.mp4.filepart')
        self.upload('.hidden.mp4')
        path = self.upload()
        self.q.scan(0)
        self.q.scan(29)
        self.assertEqual(self.q.jobs(), [])
        path.write_bytes(b'continued upload')
        self.q.scan(30)
        self.q.scan(59)
        self.assertEqual(self.q.jobs(), [])
        self.q.scan(61)
        jobs = self.q.jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(Path(jobs[0]['input']).read_bytes(), b'continued upload')
        self.assertTrue(path.is_file())

    def test_changed_source_during_snapshot_is_not_admitted(self):
        source = self.upload()
        with patch('shuolee_portrait.queue119.signature', return_value=(999, 0, 0)):
            self.assertIsNone(self.q.snapshot(source, (source.stat().st_size, 0, 0)))
        self.assertEqual(self.q.jobs(), [])
        self.assertEqual(list((self.q.root/'inputs').iterdir()), [])

    def test_renamed_duplicate_is_deduplicated_even_after_restart(self):
        self.upload()
        self.admit()
        original = self.q.jobs()[0]
        self.upload('copy.MOV')
        self.q = self.new_queue()
        self.admit()
        self.assertEqual(len(self.q.jobs()), 1)
        self.assertEqual(self.q.jobs()[0]['id'], original['id'])
        self.assertEqual(len(list((self.q.root/'inputs').iterdir())), 1)

    def test_real_children_run_sequentially_and_completed_upload_does_not_repeat(self):
        self.upload('one.mp4', b'one')
        self.upload('two.mp4', b'two')
        self.admit()
        self.q.start_next()
        pid = self.q.child.pid
        self.q.start_next()
        self.assertEqual(self.q.child.pid, pid)
        self.assertEqual([j['state'] for j in self.q.jobs()].count('queued'), 1)
        self.finish()
        self.q.start_next()
        self.finish()
        self.assertTrue(all(j['state'] == 'completed' for j in self.q.jobs()))
        self.q = self.new_queue()
        self.q.recover()
        self.admit()
        self.q.start_next()
        self.assertIsNone(self.q.child)
        self.assertEqual(len(self.q.jobs()), 2)

    def test_failure_keeps_original_error_and_next_video_can_run(self):
        self.upload('a.mp4', b'fail')
        self.upload('b.mp4', b'good')
        self.admit()
        self.q.start_next()
        self.finish()
        failed = self.q.jobs()[0]
        self.assertEqual(failed['state'], 'failed')
        self.assertIn('original child error', Path(failed['log']).read_text())
        result = read(failed['result'])
        self.assertEqual(result['outcome'], 'system_error')
        self.assertFalse(result['model_available'])
        self.q.start_next()
        self.finish()
        self.assertEqual(self.q.jobs()[1]['state'], 'completed')
        good = read(self.q.jobs()[1]['result'])
        self.assertEqual(good['outcome'], 'completed')
        self.assertTrue(good['model_available'])

    def test_camera_failure_produces_actionable_result_and_status_does_not_launch_worker(self):
        self.upload(content=b'camera'); self.admit(); self.q.start_next(); self.finish()
        job = self.q.jobs()[0]
        result = read(job['result'])
        self.assertEqual(result['outcome'], 'needs_recapture')
        self.assertEqual(result['code'], 'CAMERA_NOT_RECOVERABLE')
        self.assertFalse(result['model_available'])
        self.assertTrue(result['actions'])
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'watch.py'),
                   '--deployment', str(self.profile), '--status']
        run = subprocess.run(command, capture_output=True, encoding='utf-8', timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('person.mp4', run.stdout)
        self.assertIn('相机位姿', run.stdout)
        self.assertEqual(len(self.q.jobs()), 1)

    @unittest.skipIf(os.name == 'nt', 'POSIX signal evidence')
    def test_killed_child_always_has_result_without_inventing_oom(self):
        self.upload(content=b'killed'); self.admit(); self.q.start_next(); self.finish()
        job = self.q.jobs()[0]
        self.assertEqual(job['exit_code'], -signal.SIGKILL)
        result = read(job['result'])
        self.assertEqual(result['code'], 'PROCESS_TERMINATED')
        self.assertFalse(result['model_available'])

    def test_old_terminal_job_gets_feedback_without_retraining(self):
        self.upload(); self.admit()
        job = self.q.jobs()[0]
        job.update(state='failed', error='Insufficient verified fitting observations per real camera.')
        self.q.record(job)
        self.q.recover(); self.q.start_next()
        self.assertIsNone(self.q.child)
        self.assertEqual(read(self.q.jobs()[0]['result'])['code'], 'CAMERA_NOT_RECOVERABLE')

    def test_zero_exit_without_full_training_is_not_success(self):
        self.upload(content=b'short')
        self.admit()
        self.q.start_next()
        self.finish()
        job = self.q.jobs()[0]
        self.assertEqual(job['exit_code'], 0)
        self.assertEqual(job['state'], 'failed')
        self.assertEqual(read(job['result'])['code'], 'INCOMPLETE_RESULT')

    def test_restart_preserves_interrupted_output_without_implicit_retraining(self):
        self.upload()
        self.admit()
        job = self.q.jobs()[0]
        job['state'] = 'running'
        self.q.record(job)
        self.q.recover()
        self.assertEqual(self.q.jobs()[0]['state'], 'interrupted')
        self.assertEqual(read(self.q.jobs()[0]['result'])['outcome'], 'interrupted')
        self.q.start_next()
        self.assertIsNone(self.q.child)

    def test_crash_after_export_recovers_completed_job(self):
        self.upload()
        self.admit()
        self.q.start_next()
        self.finish()
        job = self.q.jobs()[0]
        job['state'] = 'running'
        self.q.record(job)
        self.q.recover()
        self.assertEqual(self.q.jobs()[0]['state'], 'completed')

    def test_worker_lock_prevents_duplicate_and_is_reusable(self):
        path = self.root / 'test.lock'
        with worker_lock(path):
            with self.assertRaisesRegex(RuntimeError, 'already has a worker'):
                with worker_lock(path):
                    self.fail('Duplicate worker admitted')
        with worker_lock(path):
            pass

    def test_modified_original_cannot_change_queued_training_input(self):
        source = self.upload()
        self.admit()
        job = self.q.jobs()[0]
        source.write_bytes(b'replacement')
        self.assertEqual(Path(job['input']).read_bytes(), b'video fixture')

    def test_killed_worker_does_not_leave_a_stale_os_lock(self):
        path = self.root / 'crash.lock'
        code = ('import sys,time; sys.path.insert(0,sys.argv[1]); '
                'from shuolee_portrait.queue119 import worker_lock; '
                'lock=worker_lock(sys.argv[2]); lock.__enter__(); '
                'print("locked",flush=True); time.sleep(60)')
        child = subprocess.Popen([sys.executable, '-c', code,
                str(Path(__file__).resolve().parents[1]/'src'), str(path)], stdout=subprocess.PIPE)
        try:
            self.assertEqual(child.stdout.readline().strip(), b'locked')
            child.kill()
            child.wait(timeout=10)
            with worker_lock(path):
                pass
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            child.stdout.close()


if __name__ == '__main__':
    unittest.main()
