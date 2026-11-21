from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
H=Path(__file__).parent
R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\robust-portrait-122')
O=R/'native-package-loader';stamp=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
log=(O/'cli-missing-camera.log').read_text(encoding='utf-8')
assert 'ValueError: Recorded camera ID not found: missing' in log and not (O/'must-not-render').exists()
receipt={'updated':stamp,'launcher':'render_native_scene122.sh','scope':'CLI parsing, pinned backend discovery and missing-frame rejection before any GPU rendering',
 'runtime_finding':'Current clean env has no pip-installed gsplat. Local launcher explicitly selects the existing reviewed source checkout; loader verifies source SHA and version.',
 'checks_passed':True,'new_GPU_renders':0,'source_sha256':{n:sha(H/n) for n in ['render_native_scene122.sh','verify_native_cli122.sh','native_scene_package122.py']},
 'evidence':'cli-missing-camera.log','evidence_sha256':sha(O/'cli-missing-camera.log'),'training_started':False}
save(O/'cli-review.json',receipt)
q=read(R/'queue-state.json');q['updated']=stamp
q['last_goal_turn']={'classification':'progress','evidence':['native-package-loader/result.json','native-package-loader/review.json','native-package-loader/viewer-ui-review.json','native-package-loader/ppt-update.json','native-package-loader/cli-review.json'],
 'completed':'Native recorded-camera loader and finite CUDA suite, actual report/preview update, existing PPT update and independent launch boundary verification',
 'remaining':'R1d pixel convention bridge, real camera/material/motion diagnosis, strict physical subpixel admission, fresh per-person reconstruction and unseen-video verification'}
save(R/'queue-state.json',q)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp
ledger['confirmed'].append('R1c local CLI launcher explicitly selects reviewed source gsplat; actual missing-camera CLI rejected before GPU, without new rendering or training.')
ledger['artifacts'].append('native-package-loader/cli-review.json');save(R/'evidence-ledger.json',ledger)
usage='''R1c 本地原生模型包使用说明

当前仅合成控制包通过验收，人物相机仍未准入。不会运行高斯训练。
通过WSL使用现有env与gsplat源码。与00服务器113安装入口分别维护。

示例（会生成一组渲染结果，必须使用不存在的输出目录，勿重复运行验证脚本）：
bash "/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/render_native_scene122.sh" --scene "/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-package-loader/fixture-package/scene.json" --camera 0 --projection pinhole --output /tmp/shuolee-native-camera0

相机必须明确记录像素约定与完整K。已去畸变输入只接受针孔模式。
输出：native-rgb-alpha.npz、rgb.png、alpha.png、render.json。
render.json记录场景SHA与相机ID，强度数组用于同后端回读，不能证明实拍标定准确。
CLI解析及缺失相机拒绝已实际检查，六组合成GPU渲染验证调用相同核心函数。
原生交互GUI、外部SIBR、Spark原生等价和00端本轮运行仍未验证。
'''
(O/'usage.txt').write_text(usage,encoding='utf-8',newline='\n')
state=read(H.parent/'shuolee-coordination/local-research-status.json');state['updated_at']=stamp;state['source_queue_updated']=stamp
state['renderer_readiness']['native_CLI_evidence']=str(O/'cli-review.json')
save(H.parent/'shuolee-coordination/local-research-status.json',state)
print(json.dumps({'goal_turn':'progress','phase':q['phase'],'GPU_task_running':False,'training_started':False}))
