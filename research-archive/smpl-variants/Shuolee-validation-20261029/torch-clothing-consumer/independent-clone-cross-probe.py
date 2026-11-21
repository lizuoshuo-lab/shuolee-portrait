import json,os,subprocess,tempfile,time,signal,shlex
from pathlib import Path
repo='https://github.com/lizuoshuo-lab/shuolee-portrait.git';branch='smplx/torch-clothing-consumer-123'
r={'kind':'cross check process-local config and nested shell','tls_verification_disabled':False,'persistent_config_modified':False,'cases':[]}
with tempfile.TemporaryDirectory(prefix='independent-clone-cross-') as tmp:
 for name,shell,configenv in [('direct_envconfig',False,True),('nested_argconfig',True,False),('direct_argconfig',False,False)]:
  env=dict(os.environ,HOME=tmp,HTTPS_PROXY='http://172.19.176.1:65532',HTTP_PROXY='http://172.19.176.1:65532',GIT_TERMINAL_PROMPT='0')
  if configenv:env['GIT_CONFIG_PARAMETERS']="'http.version=HTTP/1.1'"
  dest=Path(tmp)/name
  cmd=['git']+([] if configenv else ['-c','http.version=HTTP/1.1'])+['clone','--depth','1','--filter=blob:none','--no-checkout','--branch',branch,repo,str(dest)]
  if shell:cmd=['bash','-c',"bash -c "+shlex.quote('set -eu; '+shlex.join(cmd))]
  start=time.monotonic()
  p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
  try:out,err=p.communicate(timeout=45);timedout=False
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM);out,err=p.communicate(timeout=10);timedout=True
  row={'name':name,'returncode':p.returncode,'timeout':timedout,'seconds':time.monotonic()-start,'stdout':out[-1500:],'stderr':err[-1500:]}
  if (dest/'.git').exists():row['head']=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
  r['cases'].append(row);print(json.dumps(row),flush=True)
r['temporary_directory_removed']=not Path(tmp).exists()
Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/independent-clone-cross-probe.json').write_text(json.dumps(r,indent=2)+'\n')

