from pathlib import Path
import datetime, hashlib, json
import numpy as np
from PIL import Image

lab=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
out=lab/'outputs/mini-splatting2-computer00-001'
ref=np.asarray(Image.open(out/'delivery/renders/00003.png').convert('RGB').resize((446,792),Image.Resampling.LANCZOS)).astype(float)
checks={}
for version in ['before','after']:
    frame=np.asarray(Image.open(out/('viewer-color-'+version+'.jpg')).convert('RGB').crop((60,92,506,884))).astype(float)
    checks[version]={'mae_8bit':float(abs(ref-frame).mean()),'mean_rgb':frame.mean(axis=(0,1)).tolist()}
checks.update({'official_mean_rgb':ref.mean(axis=(0,1)).tolist(),'reference':'source000024 / test00003 / official CUDA PNG',
    'method':'Browser screenshot viewport center crop [60,92,506,884], official image Lanczos resized to 446x792. Approximate display diagnostic, not a benchmark metric; includes small UI overlay and JPEG screenshot error.',
    'root_cause':'Spark sets encodeLinear=true whenever outputColorSpace is not SRGBColorSpace. The initial LinearSRGBColorSpace forced an unwanted sRGB-to-linear power conversion before display.',
    'fix':'renderer.outputColorSpace=THREE.SRGBColorSpace; no model or training changes',
    'expected':'same source viewpoint and fov; original Gaussian count and degree preserved'})
assert checks['after']['mae_8bit']<checks['before']['mae_8bit']*.2
(out/'viewer-color-validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
model=json.loads((out/'viewer/model.json').read_text(encoding='utf-8'))
delivery={'status':'delivered_interactive_renderer_and_trained_model',
    'local_url':'http://127.0.0.1:8810/',
    'local_directory':str(out/'viewer'),
    'local_shortcut':r'C:\Users\Administrator\Desktop\已训练3DGS_双击查看.lnk',
    'computer00_directory':r'C:\Users\zerozero\Desktop\05_mini_splatting2_viewer\VIEWER',
    'computer00_shortcut':r'C:\Users\zerozero\Desktop\已训练3DGS_双击查看.lnk',
    'model':model['model'],'model_sha256':model['sha256'],'model_bytes':model['modelBytes'],
    'gaussians':149413,'sh_degree':3,'viewer_engine':'Spark2.3.1 / Three.js0.180.0',
    'features':['orbit','pan','zoom','camera path playback','59 labeled camera views','official 8-view comparison','original PLY download','trained-result desktop marker'],
    'checks':{'native_ply_count_and_all_finite':True,'all_32_web_assets_http_ok':True,'local_render_rotate_and_path_pause_verified':True,'computer00_open_and_3d_rotation_verified':True,'color_fix_synced_to_computer00':True},
    'color_validation':checks,'model_unmodified':True,
    'computer00_color_refresh_verified':False,
    'computer00_refresh_note':'Updated viewer.js transferred successfully before ToDesk disconnected; refresh the remote browser to apply. Corrected output verified locally.',
    'qq_progress_feedback':{'recipient':'小润','sent_verified':True,'sent_local':'2026-10-16 17:28 Asia/Shanghai','original_draft_restored':True},
    'dynamic_human_goal_complete':False,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'viewer-delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
p=lab/'manifests/mini-splatting2-computer00.json'
m=json.loads(p.read_text(encoding='utf-8-sig'));m['interactive_viewer']=delivery;m['last_recorded_utc']=delivery['recorded_utc'];p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'before_mae':checks['before']['mae_8bit'],'after_mae':checks['after']['mae_8bit'],'model_sha256':model['sha256']}))
