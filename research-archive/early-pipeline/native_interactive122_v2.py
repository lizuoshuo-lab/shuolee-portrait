"""Versioned local UI for SHA-bound v2 packages and recorded cameras."""
from pathlib import Path
from http.server import ThreadingHTTPServer
import argparse,json,os,re,secrets,subprocess,threading
from urllib.parse import urlsplit
from native_interactive122 import Handler as LegacyTransport,stamp,sha,save,read,linux,CREATE_NO_WINDOW
from native_scene_package122_v2 import load_scene,REVISION

H=Path(__file__).parent

def request_pair(body,scenes,controls,jobs):
    if not isinstance(body,dict) or set(body)!={'scene_id','camera_id'}:
        raise ValueError('Only scene_id and camera_id accepted; projection/K/pose/path overrides forbidden')
    s,c=body['scene_id'],body['camera_id']
    if not isinstance(s,str) or s not in scenes or not isinstance(c,str) or c not in scenes[s].cameras:
        raise ValueError('Unknown registered package or camera')
    if any(j['state']=='running' for j in jobs):raise ValueError('One native render is already running')
    if any(j['state']=='failed' for j in jobs):raise ValueError('Previous render failed; inspect evidence before continuing')
    if body not in controls:raise ValueError('Outside registered controls')
    if any((j['scene_id'],j['camera_id'])==(s,c) for j in jobs):raise ValueError('Already executed; show saved result instead')
    if len(jobs)>=len(controls):raise ValueError('Registered render budget exhausted')
    return s,c

class State:
    def __init__(self,plan_path,port):
        self.plan_path=Path(plan_path).resolve();self.plan=read(self.plan_path);self.root=self.plan_path.parent
        if self.plan['port']!=port:raise ValueError('Registered loopback port required')
        self.port=port;self.token=secrets.token_hex(24);self.lock=threading.Lock();self.path=self.root/'ui-state.json'
        self.jobs=read(self.path)['jobs'] if self.path.exists() else []
        if any(j['state']=='running' for j in self.jobs):raise RuntimeError('Inspect existing running process before restarting service')
        self.check_sources()
        self.scenes={name:load_scene(v['scene'],check_runtime=False) for name,v in self.plan['packages'].items()}
        for scene in self.scenes.values():
            if len(scene.manifest['projections'])!=1 or scene.manifest['backend']['adapter_revision']!=REVISION:
                raise ValueError('One explicit v2 projection per package required')
        self.flush()

    def check_sources(self):
        for path,digest in self.plan['immutable_inputs'].items():
            if sha(path)!=digest:raise ValueError('Frozen package/source changed: '+path)

    def flush(self):
        save(self.path,dict(stage='T2',updated=stamp(),server_pid=os.getpid(),port=self.port,jobs=self.jobs,
            training_started=False,physical_calibration_validated=False,adapter_revision=REVISION))

    def data(self):
        with self.lock:
            return dict(schema='native-recorded-ui-v2',adapter_revision=REVISION,csrf=self.token,jobs=list(self.jobs),controls=self.plan['controls'],
                remaining=len(self.plan['controls'])-len(self.jobs),packages={name:dict(label=self.plan['packages'][name]['label'],
                    scene_sha256=scene.manifest_sha256,scope=scene.manifest['scope'],vertices=scene.manifest['model']['vertices'],
                    input_domain=scene.manifest['input_domain'],cameras=list(scene.cameras.values()),coordinates=scene.manifest['coordinates'],
                    backend=scene.manifest['backend'],renderer=scene.manifest['renderer'],projection=next(iter(scene.manifest['projections'])),
                    projections=scene.manifest['projections'],zip_sha256=self.plan['packages'][name]['zip_sha256']) for name,scene in self.scenes.items()})

    def start(self,body):
        with self.lock:
            sid,cid=request_pair(body,self.scenes,self.plan['controls'],self.jobs);self.check_sources();scene=self.scenes[sid]
            job=dict(id=f'{len(self.jobs)+1:02d}_{sid}_{cid}',scene_id=sid,camera_id=cid,projection=next(iter(scene.manifest['projections'])),
                state='running',started=stamp(),camera=scene.cameras[cid],scene_sha256=scene.manifest_sha256,
                adapter_revision=REVISION,worker_pid=None,training_started=False)
            self.jobs.append(job);self.flush()
        threading.Thread(target=self.run,args=(job,),daemon=True).start()
        return dict(job)

    def run(self,job):
        output=self.root/job['id']
        try:
            gpu=subprocess.run(['wsl','-d','Ubuntu-20.04','--','nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15,creationflags=CREATE_NO_WINDOW)
            if gpu.returncode or gpu.stdout.strip():raise RuntimeError('GPU busy or not inspectable: '+gpu.stdout.strip())
            command=['wsl','-d','Ubuntu-20.04','--exec','bash',linux(H/'native_interactive_worker122_v2.sh'),
                '--scene',linux(self.scenes[job['scene_id']].path),'--camera',job['camera_id'],'--projection',job['projection'],'--output',linux(output)]
            log=self.root/(job['id']+'.log')
            with log.open('xb') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=CREATE_NO_WINDOW)
                with self.lock:job.update(worker_pid=process.pid,command=command,log=log.name);self.flush()
                try:code=process.wait(timeout=self.plan['budget']['per_render_seconds'])
                except subprocess.TimeoutExpired:
                    with self.lock:job.update(needs_attention='Timed observation expired. Inspect this exact process; no duplicate launch.');self.flush()
                    return
            if code:raise RuntimeError('Native v2 CLI failed with exit '+str(code))
            match=re.search(r'NATIVE_LINUX_PID=(\d+)',log.read_text(encoding='utf-8',errors='replace'))
            if not match:raise RuntimeError('Native Linux PID missing')
            receipt=read(output/'render.json')
            if any(receipt[k]!=job[k] for k in ['camera_id','projection','scene_sha256','adapter_revision']):raise RuntimeError('Receipt identity/version mismatch')
            if set(receipt['files'])!={'rgb.png','alpha.png','native-rgb-alpha.npz'}:raise RuntimeError('Incomplete render files')
            for file,digest in receipt['files'].items():
                if sha(output/file)!=digest:raise RuntimeError('Rendered asset SHA mismatch')
            self.check_sources()
            with self.lock:
                job.update(state='complete',finished=stamp(),exit_code=code,native_linux_pid=int(match[1]),files=receipt['files'],
                    native_receipt=receipt,download_prefix='/output/'+job['id']+'/');self.flush()
        except Exception as error:
            with self.lock:job.update(state='failed',finished=stamp(),error=str(error));self.flush()

class Handler(LegacyTransport):
    def do_GET(self):
        if not self.allowed_host():return self.response(403,{'error':'Loopback host only'})
        path=urlsplit(self.path).path
        if path=='/':return self.response(200,(H/'native_interactive122_v2.html').read_bytes(),'text/html; charset=utf-8')
        parts=path.strip('/').split('/')
        if len(parts)==3 and parts[0]=='package' and parts[1] in self.server.state.scenes and parts[2]=='scene.zip':
            s=self.server.state
            try:
                s.check_sources();package=s.plan['packages'][parts[1]];p=Path(package['zip'])
                if sha(p)!=package['zip_sha256']:raise ValueError('Package ZIP changed')
                return self.response(200,p.read_bytes(),'application/zip','T2_'+parts[1]+'_native_v2.zip')
            except ValueError as error:return self.response(409,{'error':str(error)})
        super().do_GET()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--port',type=int,default=8875);a=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);server.state=State(a.plan,a.port)
    print(f'Native v2 package UI http://127.0.0.1:{a.port}/ PID={os.getpid()}',flush=True);server.serve_forever()

if __name__=='__main__':main()
