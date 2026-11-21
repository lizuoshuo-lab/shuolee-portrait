from pathlib import Path
from datetime import datetime, timezone
import json
from native_interactive122 import request_pair, read, sha

R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\robust-portrait-122')
U=R/'native-interactive'; p=read(U/'plan.json')
out=U/'request-controls.json'; assert not out.exists()
cameras={x['id']:x for x in read(R/'native-package-loader/fixture-package/cameras.json')}
projections=['pinhole','opencv_fisheye4']; controls=p['controls']; base=controls[0]
checks=[]
assert request_pair(base,cameras,projections,controls,[])==('0','pinhole')
checks.append(dict(case='registered_request_accepted',passed=True))
bad=[('client_K_override',dict(base,K=[[1,0,0],[0,1,0],[0,0,1]]),[]),
 ('client_pose_override',dict(base,w2c=[]),[]),('client_path_override',dict(base,path='../../file'),[]),
 ('unknown_camera',dict(base,camera_id='missing'),[]),('unknown_projection',dict(base,projection='preview'),[]),
 ('unregistered_control',dict(base,camera_id='1'),[]),('running_job',controls[1],[dict(base,state='running')]),
 ('duplicate',base,[dict(base,state='complete')]),('budget',base,[dict(x,state='complete') for x in controls])]
for name,request,jobs in bad:
    try: request_pair(request,cameras,projections,controls,jobs)
    except ValueError as ex: checks.append(dict(case=name,passed=True,reason=str(ex)))
    else: raise AssertionError(name)
for file,digest in p['immutable_inputs'].items():assert sha(file)==digest
out.write_text(json.dumps(dict(stage='R1e_request_boundary',updated=datetime.now(timezone.utc).isoformat(),
    checks=checks,passed=True,GPU_started=False,UI_verified=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(checks=len(checks),GPU_started=False)))
