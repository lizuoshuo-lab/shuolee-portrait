"""Isolated recorded-camera UI for exact-array native ZIPs. No training entry."""
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
import argparse, hashlib, io, json, mimetypes, os, re, secrets, subprocess, threading, zipfile
from native_interactive122 import Handler as Transport, stamp, sha, save, read, linux, CREATE_NO_WINDOW
from native_interactive122_v2 import request_pair
from native_scene_package122_v5 import load_scene, SCHEMA, REVISION

H = Path(__file__).parent
CORE = {'scene.json', 'cameras.json', 'native-parameters.npz', 'point_cloud.ply'}
MAX_ZIP = 140 * 1024 * 1024


def inspect_zip(data, expected_sha):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_ZIP:
        raise ValueError('Native ZIP exceeds the registered size limit')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if len(names) != len(set(names)) or set(names) != CORE | {'README.txt'}:
            raise ValueError('Exactly four native assets and README.txt required; no paths or duplicates')
        if sum(e.file_size for e in entries) > MAX_ZIP:
            raise ValueError('Expanded native package exceeds limit')
        if any(e.is_dir() or e.flag_bits & 1 or e.compress_type != zipfile.ZIP_STORED or
               e.file_size != e.compress_size or (e.external_attr >> 16) & 0o170000 == 0o120000 for e in entries):
            raise ValueError('Only plain stored files supported')
        files = {e.filename: archive.read(e) for e in entries}
    manifest = json.loads(files['scene.json'])
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Exact-array v5 residual-camera package required; a PLY preview ZIP is different')
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Package not registered for this bounded UI acceptance')
    return files


class State:
    def __init__(self, plan, port):
        self.plan_path = Path(plan).resolve(); self.plan = read(self.plan_path); self.root = self.plan_path.parent
        if port != self.plan['port']: raise ValueError('Registered port required')
        self.port = port; self.token = secrets.token_hex(24); self.lock = threading.Lock()
        self.path = self.root / 'ui-state.json'; previous = read(self.path) if self.path.exists() else {}
        self.jobs = previous.get('jobs', []); self.imported = previous.get('imported'); self.scenes = {}
        if any(j['state'] == 'running' for j in self.jobs):
            raise RuntimeError('Inspect existing running job before restarting')
        if self.imported:
            self.scenes['exact'] = load_scene(self.root / 'imported' / 'scene.json', check_runtime=False)
        self.check_sources(); self.flush()

    def check_sources(self):
        for path, digest in self.plan['immutable_inputs'].items():
            if sha(path) != digest: raise ValueError('Frozen source changed: ' + path)
        if self.imported:
            for name, digest in self.plan['package']['assets'].items():
                if sha(self.root / 'imported' / name) != digest:
                    raise ValueError('Imported native asset changed: ' + name)

    def flush(self):
        save(self.path, dict(stage='R4', updated=stamp(), server_pid=os.getpid(), port=self.port,
            jobs=self.jobs, imported=self.imported, training_started=False, physical_quality_accepted=False))

    def data(self):
        with self.lock:
            packages = {}
            for name, scene in self.scenes.items():
                packages[name] = dict(label='20万点合成控制 · 完整残余畸变相机', scene_sha256=scene.manifest_sha256,
                    scope=scene.manifest['scope'], vertices=scene.manifest['model']['vertices'],
                    cameras=list(scene.cameras.values()), model=scene.manifest['model'],
                    backend=scene.manifest['backend'], renderer=scene.manifest['renderer'],
                    coordinates=scene.manifest['coordinates'], projection=scene.manifest['camera_contract']['projection'])
            return dict(schema='native-residual-ui-v4', csrf=self.token, jobs=list(self.jobs),
                imported=self.imported, packages=packages, controls=self.plan['controls'],
                remaining=len(self.plan['controls'])-len(self.jobs), downloads={
                    'native': self.plan['package']['zip_sha256']})

    def import_zip(self, data):
        with self.lock:
            if any(j['state'] == 'running' for j in self.jobs): raise ValueError('Wait for the current render')
            self.check_sources()
            files = inspect_zip(data, self.plan['package']['zip_sha256'])
            if self.imported: return dict(self.imported, reused=True)
            destination = self.root / 'imported'; destination.mkdir(exist_ok=False)
            for name, value in files.items(): (destination / name).write_bytes(value)
            scene = load_scene(destination / 'scene.json', check_runtime=False)
            if set(scene.cameras) != set(self.plan['camera_ids']) or scene.manifest['camera_contract']['projection'] != 'opencv_brown12_output':
                raise ValueError('Registered recorded-camera set required')
            self.imported = dict(at=stamp(), zip_sha256=hashlib.sha256(data).hexdigest(),
                scene_sha256=scene.manifest_sha256, camera_ids=list(scene.cameras), parameter_source='native-parameters.npz')
            self.scenes['exact'] = scene; self.check_sources(); self.flush()
            return dict(self.imported)

    def start(self, body):
        with self.lock:
            sid, cid = request_pair(body, self.scenes, self.plan['controls'], self.jobs)
            self.check_sources(); scene = self.scenes[sid]
            job = dict(id=f'{len(self.jobs)+1:02d}_{sid}_{cid}', scene_id=sid, camera_id=cid,
                camera=scene.cameras[cid], projection=scene.manifest['camera_contract']['projection'],
                scene_sha256=scene.manifest_sha256, adapter_revision=REVISION, state='running',
                started=stamp(), worker_pid=None, training_started=False)
            self.jobs.append(job); self.flush()
        threading.Thread(target=self.run, args=(job,), daemon=True).start()
        return dict(job)

    def run(self, job):
        output = self.root / job['id']
        try:
            check = subprocess.run(['wsl', '-d', 'Ubuntu-20.04', '--', 'nvidia-smi',
                '--query-compute-apps=pid,process_name', '--format=csv,noheader'], capture_output=True,
                text=True, encoding='utf-8', errors='replace', timeout=15, creationflags=CREATE_NO_WINDOW)
            if check.returncode or check.stdout.strip(): raise RuntimeError('GPU busy or not inspectable')
            command = ['wsl', '-d', 'Ubuntu-20.04', '--exec', 'bash', linux(H / 'native_interactive_worker122_v4.sh'),
                '--scene', linux(self.scenes[job['scene_id']].path), '--camera', job['camera_id'],
                '--output', linux(output)]
            log = self.root / (job['id'] + '.log')
            with log.open('xb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
                with self.lock: job.update(worker_pid=process.pid, command=command, log=log.name); self.flush()
                try: code = process.wait(timeout=self.plan['budget']['per_render_seconds'])
                except subprocess.TimeoutExpired:
                    with self.lock: job.update(needs_attention='Observation timeout: inspect exact PID; no duplicate launch'); self.flush()
                    return
            if code: raise RuntimeError('Native CLI failed with exit ' + str(code))
            match = re.search(r'NATIVE_LINUX_PID=(\d+)', log.read_text(encoding='utf-8', errors='replace'))
            if not match: raise RuntimeError('Missing native Linux PID')
            receipt = read(output / 'render.json')
            if any(receipt[k] != job[k] for k in ['camera_id','projection','scene_sha256','adapter_revision']):
                raise RuntimeError('Rendered camera/version identity mismatch')
            if receipt.get('canonical_parameters') != 'native-parameters.npz' or set(receipt['files']) != {'rgb.png','alpha.png','native-rgb-alpha.npz'}:
                raise RuntimeError('Wrong parameter source or incomplete outputs')
            for name, digest in receipt['files'].items():
                if sha(output / name) != digest: raise RuntimeError('Output SHA mismatch')
            self.check_sources()
            with self.lock:
                job.update(state='complete', finished=stamp(), native_linux_pid=int(match[1]), files=receipt['files'],
                    receipt_sha256=sha(output/'render.json'), native_receipt=receipt, download_prefix='/output/'+job['id']+'/')
                self.flush()
        except Exception as error:
            with self.lock: job.update(state='failed', finished=stamp(), error=str(error)); self.flush()


class Handler(Transport):
    def do_GET(self):
        if not self.allowed_host(): return self.response(403, {'error':'Loopback host only'})
        path = urlsplit(self.path).path; state = self.server.state
        if path == '/': return self.response(200, (H/'native_interactive122_v4.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/download/native.zip':
            try:
                state.check_sources(); file = Path(state.plan['package']['zip'])
                return self.response(200, file.read_bytes(), 'application/zip', 'R4_residual_native.zip')
            except ValueError as error: return self.response(409, {'error':str(error)})
        parts = path.strip('/').split('/')
        if len(parts)==3 and parts[0]=='output':
            job = next((j for j in state.data()['jobs'] if j['id']==parts[1] and j['state']=='complete'), None)
            if job and parts[2] in {'rgb.png','alpha.png','native-rgb-alpha.npz','render.json'}:
                file = state.root / job['id'] / parts[2]
                expected = job['receipt_sha256'] if parts[2]=='render.json' else job['files'][parts[2]]
                if sha(file)!=expected: return self.response(409, {'error':'Saved output SHA changed'})
                return self.response(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'application/octet-stream',
                    file.name if file.suffix in ('.npz','.json') else None)
            return self.response(404,{'error':'Unknown completed output'})
        super().do_GET()

    def do_POST(self):
        state = self.server.state
        if not self.allowed_host() or self.headers.get('Origin')!=f'http://127.0.0.1:{state.port}' or self.headers.get('X-Native-Token')!=state.token:
            return self.response(403,{'error':'Local same-origin token required'})
        if self.path != '/api/import': return super().do_POST()
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= MAX_ZIP: raise ValueError('Invalid import size')
            data = self.rfile.read(length)
            if len(data)!=length: raise ValueError('Incomplete local file transfer')
            return self.response(200,state.import_zip(data))
        except (ValueError,TypeError,zipfile.BadZipFile) as error:
            return self.response(400,{'error':str(error)})


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',required=True); parser.add_argument('--port',type=int,default=8877)
    args=parser.parse_args(); server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler); server.state=State(args.plan,args.port)
    print(f'Exact native package UI http://127.0.0.1:{args.port}/ PID={os.getpid()}',flush=True); server.serve_forever()

if __name__=='__main__': main()
