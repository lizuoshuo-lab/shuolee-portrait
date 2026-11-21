import json,os,re,subprocess,tempfile,time,signal,hashlib
from pathlib import Path
doc=Path('/mnt/d/Projects/Shuolee-torch-clothing-consumer-123/docs/逐帧Torch消费层.md').read_text(encoding='utf-8')
command=next(c for c in re.findall(r'```bash\n(.*?)\n```',doc,re.S) if c.startswith("bash -c 'set -eu;"))
clone_checkout=command.split('; CHECK_PY=',1)[0]+"'"
r={'kind':'exact documented nested bash clone and sparse checkout only','git_version':subprocess.check_output(['git','--version'],text=True).strip(),'proxy_explicit':True,'http_version':'HTTP/1.1','tls_verification_disabled':False,'persistent_config_modified':False,'tests_run':False}
with tempfile.TemporaryDirectory(prefix='independent-doc-clone-') as tmp:
 env=dict(os.environ,HOME=tmp,HTTPS_PROXY='http://172.19.176.1:65532',HTTP_PROXY='http://172.19.176.1:65532',GIT_CONFIG_PARAMETERS="'http.version=HTTP/1.1'",GIT_TERMINAL_PROMPT='0')
 start=time.monotonic()
 p=subprocess.Popen(['bash','-c',clone_checkout],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 try:
  out,err=p.communicate(timeout=100);r['timeout']=False
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM);out,err=p.communicate(timeout=10);r['timeout']=True
 r.update(returncode=p.returncode,seconds=time.monotonic()-start,stdout=out[-2000:],stderr=err[-2000:])
 checkout=Path(tmp)/'smpl-x-torch-clothing'
 if (checkout/'.git').exists():
  r['head']=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
  r['sparse_directories']=subprocess.run(['git','-C',str(checkout),'sparse-checkout','list'],capture_output=True,text=True).stdout.splitlines()
  r['tracked_source_present']=(checkout/'src/smplx_pipeline/free_clothing_torch.py').exists()
  r['source_candidates']=[str(x.relative_to(checkout)) for x in checkout.glob('src/**/free_clothing_torch.py')]
r['temporary_directory_removed']=not Path(tmp).exists()
Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/independent-doc-clone-probe.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2),flush=True)

