"""Model bootstrap controls: missing licenses, HTTP transfers and pinned bytes."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_assets as assets


def entry(name, data, **kwargs):
    return dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), **kwargs)


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='smpl-x-model-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'public model fixture' * 100
        self.requests = []
        self.range_mode = 'honor'
        self.body = self.payload
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                request_range = self.headers.get('Range')
                outer.requests.append(request_range)
                offset = int(request_range[6:-1]) if request_range else 0
                partial = bool(request_range) and outer.range_mode != 'ignore'
                if not partial:
                    offset = 0
                self.send_response(206 if partial else 200)
                if partial:
                    start = offset if outer.range_mode == 'honor' else 0
                    self.send_header('Content-Range',
                                     f'bytes {start}-{len(outer.body)-1}/{len(outer.body)}')
                body = outer.body[offset:]
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.server.server_port}/model'
        self.target = self.root / 'downloaded'
        self.row = entry('downloaded', self.payload)

    def test_http_download_matches_pinned_hash_and_size(self):
        assets.download(self.url, self.target, self.row)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.target.with_suffix('.part').exists())

    def test_interrupted_download_resumes_from_verified_final_bytes(self):
        self.target.with_suffix('.part').write_bytes(self.payload[:39])
        assets.download(self.url, self.target, self.row)
        self.assertEqual(self.requests, ['bytes=39-'])
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        self.range_mode = 'ignore'
        self.target.with_suffix('.part').write_bytes(self.payload[:39])
        assets.download(self.url, self.target, self.row)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_wrong_range_is_rejected_without_publishing(self):
        self.range_mode = 'wrong'
        self.target.with_suffix('.part').write_bytes(self.payload[:39])
        with self.assertRaisesRegex(RuntimeError, 'Content-Range'):
            assets.download(self.url, self.target, self.row)
        self.assertFalse(self.target.exists())

    def test_same_size_wrong_hash_does_not_replace_existing_file(self):
        self.body = b'x' * len(self.payload)
        self.target.write_bytes(b'existing original')
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            assets.download(self.url, self.target, self.row)
        self.assertEqual(self.target.read_bytes(), b'existing original')
        self.assertFalse(self.target.with_suffix('.part').exists())

    def test_oversized_response_does_not_publish(self):
        self.body = self.payload + b'excess'
        with self.assertRaisesRegex(RuntimeError, 'exceeds pinned size'):
            assets.download(self.url, self.target, self.row)
        self.assertFalse(self.target.exists())

    @unittest.skipUnless(sys.platform == 'linux', 'Server file locking requires Linux')
    def test_missing_licensed_models_stop_before_network(self):
        manifest = {'files': [entry('licensed/template', b'license'),
                              entry('detector', self.payload, download={'url': self.url})]}
        with self.assertRaisesRegex(RuntimeError, 'Licensed server models'):
            assets.prepare(self.root, [], manifest)
        self.assertEqual(self.requests, [])

    @unittest.skipUnless(sys.platform == 'linux', 'Server file locking requires Linux')
    def test_incomplete_legacy_root_is_assembled_without_modifying_it(self):
        legacy = self.root / 'old-server'
        legacy.mkdir()
        (legacy / 'template').write_bytes(b'license')
        manifest = {'files': [entry('template', b'license'),
                              entry('duplicate/template', b'license', reuse='template'),
                              entry('detector', self.payload, download={'url': self.url})]}
        result = assets.prepare(self.root, [legacy], manifest)
        self.assertEqual(result, self.root / 'models')
        self.assertEqual(list(legacy.iterdir()), [legacy / 'template'])
        self.assertEqual((result / 'duplicate/template').read_bytes(), b'license')
        self.assertEqual((result / 'detector').read_bytes(), self.payload)
        requests = len(self.requests)
        self.assertEqual(assets.prepare(self.root, [legacy, result], manifest), result)
        self.assertEqual(len(self.requests), requests)

    def test_complete_directory_is_preferred_over_first_template_only_candidate(self):
        legacy, complete = self.root / 'legacy', self.root / 'complete'
        for root in (legacy, complete):
            root.mkdir()
            (root / 'template').write_bytes(b'license')
        (complete / 'detector').write_bytes(self.payload)
        manifest = {'files': [entry('template', b'license'),
                              entry('detector', self.payload, download={'url': self.url})]}
        self.assertEqual(assets.prepare(self.root, [legacy, complete], manifest), complete)
        self.assertFalse((self.root / 'models').exists())
        self.assertEqual(self.requests, [])

    @unittest.skipUnless(sys.platform == 'linux', 'Server file locking requires Linux')
    def test_archive_uses_matching_model_and_normalization_and_removes_zip(self):
        normalization = {'type': 'Normalize', 'mean': [1, 2, 3], 'std': [4, 5, 6]}
        normalization_bytes = (json.dumps(normalization, indent=2) + '\n').replace('\n', '\r\n').encode()
        pipeline = {'pipeline': {'tasks': [{'transforms': [normalization]}]}}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('end2end.onnx', self.payload)
            archive.writestr('pipeline.json', json.dumps(pipeline))
            archive.writestr('../outside', b'must not extract')
        self.body = stream.getvalue()
        manifest = {'files': [
            entry('rtmw/normalization.json', normalization_bytes,
                  download={'archive': 'rtmw.zip', 'member': 'pipeline.json', 'normalize': True,
                            'newline': 'crlf'}),
            entry('rtmw/model.onnx', self.payload,
                  download={'archive': 'rtmw.zip', 'member': 'end2end.onnx'})],
            'archives': {'rtmw.zip': entry('rtmw.zip', self.body, url=self.url)}}
        result = assets.prepare(self.root, [], manifest)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual((result / 'rtmw/model.onnx').read_bytes(), self.payload)
        self.assertEqual((result / 'rtmw/normalization.json').read_bytes(), normalization_bytes)
        self.assertFalse((result / '.downloads/rtmw.zip').exists())
        self.assertFalse((self.root / 'outside').exists())

    def test_bad_saved_profile_does_not_hide_existing_candidates(self):
        (self.root / 'queue').mkdir()
        (self.root / 'queue/deployment.json').write_text('incomplete json')
        with patch.object(Path, 'home', return_value=self.root / 'home'):
            result = assets.candidates(self.root, self.root / 'environment')
        self.assertIn(self.root / 'models', result)


if __name__ == '__main__':
    unittest.main()
