"""Compare non-secret Git transport settings across normal and temporary HOME."""
import json,os,subprocess,tempfile
from urllib.parse import urlsplit
from pathlib import Path

keys=['http.proxy','https.proxy','http.version','http.sslVerify','http.sslCAInfo']
def settings(env):
    result={}
    for key in keys:
        p=subprocess.run(['git','config','--get',key],env=env,capture_output=True,text=True)
        value=p.stdout.strip()
        if 'proxy' in key.lower() and value:
            u=urlsplit(value);value={'scheme':u.scheme,'hostname':u.hostname,'port':u.port,'credentials_present':u.username is not None}
        result[key]=value if p.returncode==0 else None
    return result
r={'normal_home':settings(dict(os.environ))}
with tempfile.TemporaryDirectory(prefix='git-net-config-') as temp:
    r['temporary_home']=settings(dict(os.environ,HOME=temp,GIT_CONFIG_PARAMETERS="'http.version=HTTP/1.1'"))
    for label,env in [('normal',dict(os.environ)),('temporary',dict(os.environ,HOME=temp))]:
        p=subprocess.run(['git','-c','http.version=HTTP/1.1','ls-remote','--heads','https://github.com/lizuoshuo-lab/shuolee-portrait.git','smplx/torch-clothing-consumer-123'],env=env,capture_output=True,text=True,timeout=45)
        r[label+'_ls_remote']={'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
Path('/mnt/d/Projects/Shuolee-validation-20261029/torch-clothing-consumer/network-config-probe.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
