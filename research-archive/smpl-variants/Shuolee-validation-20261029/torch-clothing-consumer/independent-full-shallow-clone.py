import json,os,subprocess,tempfile,time,signal
from pathlib import Path
r={'kind':'standard full shallow clone, no filter or sparse checkout','expected_commit':'4422cba242fa712c8003bad6d24472f29cc5bd7f','cwd':os.getcwd(),'tls_verification_disabled':False,'persistent_config_modified':False,'tests_run':False}
temp=tempfile.TemporaryDirectory(prefix='independent-full-shallow-')
tmp=Path(temp.name);checkout=tmp/'smpl-x-torch-clothing'
env=dict(os.environ,HOME=str(tmp),HTTPS_PROXY='http://172.19.176.1:65532',HTTP_PROXY='http://172.19.176.1:65532',GIT_TERMINAL_PROMPT='0')
start=time.monotonic()
p=subprocess.Popen(['git','-c','http.version=HTTP/1.1','clone','--depth','1','--branch','smplx/torch-clothing-consumer-123','https://github.com/lizuoshuo-lab/shuolee-portrait.git',str(checkout)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
try:out,err=p.communicate(timeout=180);r['timeout']=False
except subprocess.TimeoutExpired:
 os.killpg(p.pid,signal.SIGTERM);out,err=p.communicate(timeout=10);r['timeout']=True
r.update(returncode=p.returncode,seconds=time.monotonic()-start,stdout=out[-2000:],stderr=err[-2000:],temporary_directory=str(tmp),checkout=str(checkout))
if p.returncode==0:
 r['head']=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
 r['status']=subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True).strip()
 r['source_candidates']=[str(x.relative_to(checkout)) for x in checkout.glob('src/**/free_clothing_torch.py')]
 r['retained_for_parent']=True;temp._finalizer.detach()
else:
 temp.cleanup();r['retained_for_parent']=False
r['temporary_directory_exists']=tmp.exists()
Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/independent-full-shallow-clone.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2),flush=True)

