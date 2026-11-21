"""Focused CPU receipt/commit controls; never execute surface initialization."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import hashlib
import json
import tempfile
import time
import unittest

import prepare_framewise_surface123 as entry


class ReceiptControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_plan_original_digest_bound_and_mutation_rejected(self):
        p=self.root/'plan.json';p.write_text('{"budget": 10}',encoding='utf-8')
        value,digest=entry.read_frozen_plan(p)
        self.assertEqual(value,dict(budget=10));self.assertEqual(digest,entry.sha(p))
        p.write_text('{"budget": 20}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'plan changed'):entry.verify_frozen_plan(p,digest)
        self.assertEqual(value,dict(budget=10))

    def test_plan_changed_during_read_rejected(self):
        p=self.root/'plan.json';p.write_text('{"budget": 10}',encoding='utf-8')
        original=entry.read
        def raced(path):
            value=original(path);p.write_text('{"budget": 20}',encoding='utf-8');return value
        with patch.object(entry,'read',raced),self.assertRaisesRegex(ValueError,'plan changed'):
            entry.read_frozen_plan(p)

    def test_precommit_error_records_failure(self):
        self.assertFalse(entry.record_run_exception(self.root,ValueError('not committed'),time.monotonic()))
        report=entry.read(self.root/'failure.json')
        self.assertFalse(report['artifact_commit_completed']);self.assertTrue(report['partial_outputs_preserved'])
        self.assertFalse((self.root/'post-commit-notification-error.json').exists())

    def test_broken_stdout_after_atomic_commit_does_not_relabel_artifact(self):
        p=self.root/'manifest.json';entry.write_json(p,dict(state='initialized_for_review'))
        original=p.read_bytes()
        self.assertTrue(entry.record_run_exception(self.root,BrokenPipeError('stdout closed'),time.monotonic()))
        self.assertEqual(p.read_bytes(),original);self.assertFalse((self.root/'failure.json').exists())
        report=entry.read(self.root/'post-commit-notification-error.json')
        self.assertTrue(report['artifact_commit_completed']);self.assertFalse(report['partial_outputs_preserved'])

    def test_atomic_writer_failure_never_creates_ready_file(self):
        p=self.root/'manifest.json'
        def fail(stream):
            stream.write(b'partial');raise ValueError('writer failed')
        with self.assertRaisesRegex(ValueError,'writer failed'):entry.atomic_file(p,fail)
        self.assertFalse(p.exists());self.assertEqual(list(self.root.iterdir()),[])

    def test_atomic_writer_refuses_existing_artifact(self):
        p=self.root/'manifest.json';entry.write_json(p,dict(original=True));raw=p.read_bytes()
        with self.assertRaisesRegex(ValueError,'Fresh output'):entry.write_json(p,dict(replacement=True))
        self.assertEqual(p.read_bytes(),raw)

    def test_budget_requires_registered_posix_timer_capabilities(self):
        for name,platform,timer in [('nt','win32',SimpleNamespace()),('posix','linux',SimpleNamespace(SIGALRM=14))]:
            with patch.object(entry.os,'name',name),patch.object(entry.sys,'platform',platform),patch.object(entry,'signal',timer):
                with self.assertRaisesRegex(ValueError,'WSL/Linux POSIX'):entry.require_posix_budget()
        with patch.object(entry.os,'name','posix'),patch.object(entry.sys,'platform','linux'), \
             patch.object(entry,'signal',SimpleNamespace(SIGALRM=14,setitimer=object())):
            entry.require_posix_budget()

    def test_bound_visual_or_geometry_receipt_changes_are_rejected(self):
        p=self.root/'review.json';p.write_text('{"review": true}',encoding='utf-8')
        descriptor=dict(path=p.name,sha256=entry.sha(p));self.assertEqual(entry.checked(self.root,descriptor),p.resolve())
        p.write_text('{"review": false}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'changed input'):entry.checked(self.root,descriptor)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReceiptControls))
    report=dict(state='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,
        source_sha256=entry.sha(entry.__file__),test_sha256=entry.sha(__file__),
        surface_initialization_calls=0,model_forwards=0,GPU_calls=0,
        scope='Pure file/receipt/commit controls; no real initializer, training or visual-quality acceptance.',
        failure_details=[str(test)+'\n'+text for test,text in result.errors+result.failures])
    with open(args.report,'x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
