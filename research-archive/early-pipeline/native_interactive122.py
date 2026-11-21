"""Loopback-only, bounded native recorded-camera rendering UI; never a preview fallback."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timezone
import argparse, hashlib, json, mimetypes, os, re, secrets, subprocess, threading
from urllib.parse import urlsplit

H = Path(__file__).parent
CREATE_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    tmp.replace(path)

def linux(path):
    p = Path(path).resolve().as_posix()
    if len(p) < 4 or p[1:3] != ':/':
        raise ValueError('Windows local path required')
    return '/mnt/' + p[0].lower() + '/' + p[3:]

def request_pair(body, cameras, projections, controls, jobs):
    if not isinstance(body, dict) or set(body) != {'camera_id','projection'}:
        raise ValueError('Only recorded camera_id and projection are accepted; K, pose and path overrides are forbidden')
    c, p = body['camera_id'], body['projection']
    if not isinstance(c, str) or c not in cameras or not isinstance(p, str) or p not in projections:
        raise ValueError('Unknown recorded camera or projection')
    if any(j['state']=='running' for j in jobs):
        raise ValueError('One native render is already running')
    if len(jobs) >= len(controls):
        raise ValueError('Registered render budget is exhausted')
    if body not in controls:
        raise ValueError('Camera/projection is outside registered controls')
    if any((j['camera_id'], j['projection']) == (c,p) for j in jobs):
        raise ValueError('This control was already executed; no duplicate render')
    return c,p

class State:
    def __init__(self, plan_path, port):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read(self.plan_path)
        self.root = self.plan_path.parent
        self.scene_path = Path(self.plan['scene'])
        self.scene = read(self.scene_path)
        self.cameras = {c['id']: c for c in read(self.scene_path.parent/self.scene['cameras']['path'])}
        self.port = port
        self.token = secrets.token_hex(24)
        self.lock = threading.Lock()
        self.path = self.root/'ui-state.json'
        self.jobs = read(self.path)['jobs'] if self.path.exists() else []
        if any(j['state']=='running' for j in self.jobs):
            raise RuntimeError('Existing running receipt requires process inspection; refusing a second server')
        self.check_sources()
        self.flush()

    def check_sources(self):
        for path, expected in self.plan['immutable_inputs'].items():
            if sha(path) != expected:
                raise ValueError('Frozen renderer/package changed: '+path)

    def flush(self):
        save(self.path, dict(stage='R1e', updated=stamp(), server_pid=os.getpid(), port=self.port,
                            jobs=self.jobs, training_started=False, physical_calibration_validated=False))

    def data(self):
        with self.lock:
            return dict(schema='native-recorded-ui-v1', scene_sha256=sha(self.scene_path),
                scope=self.scene['scope'], input_domain=self.scene['input_domain'], cameras=list(self.cameras.values()),
                coordinates=self.scene['coordinates'], renderer=self.scene['renderer'], backend=self.scene['backend'],
                projections=self.scene['projections'], controls=self.plan['controls'], jobs=list(self.jobs),
                remaining=len(self.plan['controls'])-len(self.jobs), csrf=self.token)

    def start(self, body):
        with self.lock:
            c,p = request_pair(body, self.cameras, self.scene['projections'], self.plan['controls'], self.jobs)
            self.check_sources()
            job = dict(id=f'{len(self.jobs)+1:02d}_{p}_{c}', camera_id=c, projection=p, state='running', started=stamp(),
                       camera=self.cameras[c], scene_sha256=sha(self.scene_path), worker_pid=None, training_started=False)
            self.jobs.append(job); self.flush()
        threading.Thread(target=self.run, args=(job,), daemon=True).start()
        return dict(job)

    def run(self, job):
        output = self.root/job['id']
        try:
            # A single loopback server serializes its own jobs. Also reject other GPU compute.
            gpu = subprocess.run(['wsl','-d','Ubuntu-20.04','--','nvidia-smi',
                '--query-compute-apps=pid,process_name','--format=csv,noheader'], capture_output=True,
                text=True, encoding='utf-8', errors='replace', timeout=15, creationflags=CREATE_NO_WINDOW)
            if gpu.returncode or gpu.stdout.strip():
                raise RuntimeError('GPU compute is busy or cannot be inspected; render not launched: '+gpu.stdout.strip())
            command = ['wsl','-d','Ubuntu-20.04','--exec','bash',linux(H/'native_interactive_worker122.sh'),
                       '--scene',linux(self.scene_path),'--camera',job['camera_id'],
                       '--projection',job['projection'],'--output',linux(output)]
            log = self.root/(job['id']+'.log')
            with log.open('xb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
                with self.lock:
                    job.update(worker_pid=process.pid, command=command, log=log.name); self.flush()
                try:
                    code = process.wait(timeout=self.plan['budget']['per_render_seconds'])
                except subprocess.TimeoutExpired:
                    # Do not kill all WSL or start another task. Preserve pending ownership.
                    with self.lock:
                        job.update(state='running', needs_attention='Native render exceeded registered time; inspect exact process before further work'); self.flush()
                    return
            if code != 0:
                raise RuntimeError('Native CLI failed with exit '+str(code)+'; see '+log.name)
            process_record = re.search(r'NATIVE_LINUX_PID=(\d+)', log.read_text(encoding='utf-8',errors='replace'))
            if not process_record:
                raise RuntimeError('Missing native Linux process identity')
            receipt = read(output/'render.json')
            if any(receipt[k] != job[k] for k in ['camera_id','projection','scene_sha256']):
                raise RuntimeError('Native receipt does not match requested recorded camera')
            for file,digest in receipt['files'].items():
                if file not in ['rgb.png','alpha.png','native-rgb-alpha.npz'] or sha(output/file) != digest:
                    raise RuntimeError('Rendered file mismatch')
            self.check_sources()
            with self.lock:
                job.update(state='complete', finished=stamp(), exit_code=code, native_linux_pid=int(process_record[1]), files=receipt['files'],
                           native_receipt=receipt, download_prefix='/output/'+job['id']+'/'); self.flush()
        except Exception as ex:
            with self.lock:
                job.update(state='failed', finished=stamp(), error=str(ex)); self.flush()

class Handler(BaseHTTPRequestHandler):
    def response(self, code, data, content_type='application/json; charset=utf-8', download=None):
        if not isinstance(data, bytes):
            data=json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="'+download+'"')
        self.end_headers(); self.wfile.write(data)

    def allowed_host(self):
        return self.headers.get('Host') == f'127.0.0.1:{self.server.state.port}'

    def do_GET(self):
        if not self.allowed_host():
            return self.response(403, {'error':'Loopback host only'})
        path=urlsplit(self.path).path
        if path=='/':
            return self.response(200, (H/'native_interactive122.html').read_bytes(), 'text/html; charset=utf-8')
        if path=='/api/state':
            return self.response(200, self.server.state.data())
        parts=path.strip('/').split('/')
        if len(parts)==3 and parts[0]=='output':
            job=next((j for j in self.server.state.data()['jobs'] if j['id']==parts[1] and j['state']=='complete'),None)
            name=parts[2]
            if job and name in ['rgb.png','alpha.png','native-rgb-alpha.npz','render.json']:
                data=(self.server.state.root/job['id']/name).read_bytes()
                return self.response(200,data,mimetypes.guess_type(name)[0] or 'application/octet-stream',
                                     name if name.endswith(('.npz','.json')) else None)
        self.response(404, {'error':'Unknown asset'})

    def do_POST(self):
        s=self.server.state
        if not self.allowed_host() or self.headers.get('Origin') != f'http://127.0.0.1:{s.port}' or self.headers.get('X-Native-Token') != s.token:
            return self.response(403, {'error':'Local same-origin token required'})
        if self.path!='/api/render':
            return self.response(404, {'error':'Unknown command'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<2 or length>1024:
                raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(length))
            self.response(202, s.start(body))
        except (ValueError, TypeError) as ex:
            self.response(400, {'error':str(ex)})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8874)
    a=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    server.state=State(a.plan,a.port)
    print(f'Native recorded-camera UI http://127.0.0.1:{a.port}/',flush=True)
    server.serve_forever()

if __name__=='__main__':
    main()
