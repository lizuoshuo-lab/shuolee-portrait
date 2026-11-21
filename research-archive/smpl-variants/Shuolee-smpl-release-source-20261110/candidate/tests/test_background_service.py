"""Real detached-process controls on Linux; child outputs are CPU queue fixtures."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait.queue119 import VideoQueue, now, read, save, worker_lock, process_group_running


@unittest.skipUnless(sys.platform == 'linux', 'Linux detached-process controls required')
class BackgroundServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='smpl-x service ')
        self.root = Path(self.temp.name)
        self.app = self.root / 'app'
        self.app.mkdir()
        self.profile = self.root / 'deployment.json'
        save(self.profile, dict(schema='shuolee-deployment119-v1', environment=sys.prefix,
                               assets=str(self.root / 'assets'), runs=str(self.root / 'runs')))
        self.queue = VideoQueue(self.profile)
        self.worker_pids = set()
        self.child = self.root / 'fixture_child.py'
        self.child.write_text('''
import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('video'); p.add_argument('--deployment'); p.add_argument('--output')
a = p.parse_args()
if Path(a.video).read_bytes() == b'hold': time.sleep(60)
root = Path(a.output); (root/'export').mkdir(parents=True)
(root/'export/point_cloud.ply').write_text('CPU orchestration fixture, not reconstruction')
(root/'result.json').write_text(json.dumps(dict(state='completed', steps=18000, format_passed=True)))
''', encoding='utf-8')
        (self.app / 'watch.py').write_text(f'''
import argparse, sys
sys.path.insert(0, {str(PROJECT / 'src')!r})
from shuolee_portrait.queue119 import VideoQueue
p = argparse.ArgumentParser(); p.add_argument('--deployment'); a = p.parse_args()
VideoQueue(a.deployment, settle_seconds=0.1,
           command=[sys.executable, '-s', {str(self.child)!r}]).serve(poll_seconds=0.05)
''', encoding='utf-8')
        self.bootstrap = self.root / 'install.py'
        self.bootstrap.write_text(f'''
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('install_service', {str(PROJECT / 'scripts/install_service.py')!r})
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.PROJECT = Path({str(self.app)!r})
module.systemd_available = lambda: False
module.main(['--deployment', {str(self.profile)!r}])
''', encoding='utf-8')

    def wait_for(self, check, seconds=10):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            value = check()
            if value:
                return value
            time.sleep(0.05)
        self.fail('Process condition timed out')

    def status(self):
        try:
            return read(self.queue.root / 'status.json')
        except (OSError, ValueError):
            return {}

    def install(self, success=True):
        result = subprocess.run([sys.executable, '-s', str(self.bootstrap)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('smpl-x READY', result.stdout)
            self.worker_pids.add(self.status()['pid'])
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn('smpl-x READY', result.stdout)
        return result

    @staticmethod
    def alive(pid):
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
        except FileNotFoundError:
            return False

    def stop_worker(self, pid):
        if self.alive(pid):
            os.kill(pid, signal.SIGTERM)
            self.wait_for(lambda: not self.alive(pid))

    def tearDown(self):
        # Only signal processes carrying this test's unique temporary script paths.
        pids = self.worker_pids | {self.status().get('pid')}
        pids.update(job.get('pid') for job in self.queue.jobs())
        for pid in pids:
            if not isinstance(pid, int):
                continue
            try:
                args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                if os.fsencode(self.app / 'watch.py') in args or os.fsencode(self.child) in args:
                    os.killpg(pid, signal.SIGTERM)
                    self.wait_for(lambda: not self.alive(pid))
            except (FileNotFoundError, ProcessLookupError):
                pass
        self.temp.cleanup()

    def test_terminal_hangup_upload_repeat_deploy_and_restart(self):
        terminal = self.root / 'terminal.py'
        terminal.write_text(self.bootstrap.read_text() +
                            f'\nPath({str(self.root / "terminal-ready")!r}).touch()\n'
                            'import time; time.sleep(60)\n', encoding='utf-8')
        parent = subprocess.Popen([sys.executable, '-s', str(terminal)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            self.wait_for(lambda: (self.root / 'terminal-ready').exists())
            pid = self.status()['pid']
            self.worker_pids.add(pid)
            self.assertEqual(os.getsid(pid), pid)
            self.assertNotEqual(os.getsid(pid), os.getsid(parent.pid))
            self.assertEqual(os.readlink(f'/proc/{pid}/fd/0'), '/dev/null')
            self.assertEqual(os.readlink(f'/proc/{pid}/fd/1'), str(self.queue.root / 'worker.log'))
            os.killpg(parent.pid, signal.SIGHUP)
            output = parent.communicate(timeout=5)[0]
            self.assertIn(b'smpl-x READY', output)
            os.kill(pid, signal.SIGHUP)
            before = (self.queue.root / 'status.json').stat().st_mtime_ns
            self.wait_for(lambda: (self.queue.root / 'status.json').stat().st_mtime_ns > before)
            self.install()
            self.assertEqual(self.status()['pid'], pid)
            (self.queue.inbox / 'one.mp4').write_bytes(b'one')
            (self.queue.inbox / 'two.mp4').write_bytes(b'two')
            (self.queue.inbox / 'partial.mp4.filepart').write_bytes(b'partial')
            self.wait_for(lambda: self.status().get('counts') == {'completed': 2})
            self.stop_worker(pid)
            self.install()
            self.assertNotEqual(self.status()['pid'], pid)
            self.wait_for(lambda: self.status().get('counts') == {'completed': 2})
            time.sleep(0.3)
            self.assertEqual(len(self.queue.jobs()), 2)
            self.assertIsNone(self.status()['active'])
        finally:
            if parent.poll() is None:
                os.killpg(parent.pid, signal.SIGKILL)
                parent.wait(timeout=5)
            parent.stdout.close()

    def test_stale_ready_status_cannot_hide_startup_failure(self):
        save(self.queue.root / 'status.json', dict(state='watching', pid=os.getpid()))
        (self.app / 'watch.py').write_text("raise RuntimeError('fixture startup failure')\n")
        result = self.install(success=False)
        self.assertIn('Background worker exited', result.stderr)
        self.assertIn('fixture startup failure', (self.queue.root / 'worker.log').read_text())

    def test_child_keeps_queue_lock_after_worker_is_killed(self):
        self.install()
        pid = self.status()['pid']
        (self.queue.inbox / 'hold.mp4').write_bytes(b'hold')
        job = self.wait_for(lambda: self.status().get('active'))
        child_pid = job['pid']
        os.kill(pid, signal.SIGKILL)
        self.wait_for(lambda: not self.alive(pid))
        self.assertTrue(self.alive(child_pid))
        with self.assertRaisesRegex(RuntimeError, 'already has a worker'):
            with worker_lock(self.queue.root / 'worker.lock'):
                self.fail('Surviving training child lost the queue lock')
        self.assertIn('already has a worker', self.install(success=False).stderr)
        os.killpg(child_pid, signal.SIGTERM)
        self.wait_for(lambda: not self.alive(child_pid))
        self.install()
        self.assertEqual(self.status()['counts'], {'interrupted': 1})
        self.assertIsNone(self.status()['active'])

    def test_recovery_waits_for_descendants_after_group_leader_exits(self):
        child_file = self.root / 'descendant.pid'
        code = ("import os,time,pathlib; pid=os.fork(); "
                f"pathlib.Path({str(child_file)!r}).write_text(str(pid)) if pid else None; "
                'time.sleep(60) if pid==0 else None')
        leader = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        leader.wait(timeout=5)
        child_pid = int(child_file.read_text())
        job = dict(id='orphan', created=now(), state='running', pid=leader.pid,
                   run=str(self.root / 'runs/orphan'))
        self.queue.record(job)
        try:
            self.assertTrue(process_group_running(leader.pid))
            with self.assertRaisesRegex(RuntimeError, 'Previous training process group'):
                self.queue.recover()
            self.assertEqual(self.queue.jobs()[0]['state'], 'running')
        finally:
            if self.alive(child_pid):
                os.killpg(leader.pid, signal.SIGTERM)
                self.wait_for(lambda: not self.alive(child_pid))
        self.queue.recover()
        self.assertEqual(self.queue.jobs()[0]['state'], 'interrupted')


if __name__ == '__main__':
    unittest.main()
