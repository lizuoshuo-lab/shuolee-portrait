"""Close the first MSV2 input-resolution comparison without promoting its quality."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
from PIL import Image,ImageDraw,ImageFont

OUT=Path('D:/00/记录/2026-11-04_MSV2_人体3倍上采样/demo')
RUN=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-demo-upsample3x-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
summary=[
    '本地训练完成，文件格式与59台配套相机导入通过；重建质量未通过，保留为本次路线对照。',
    '黑底下手臂周围的大块拖影比上一版少，但灰底原生CUDA渲染和独立查看器都显示头顶、手臂及脚边的黑色放射伪影。7个检查视角中，mask外约24.9%—28.0%的像素累计不透明度大于0.1，黑底掩盖了这些多余高斯。',
    '已查看全部59帧全身与脸部对照、12个环绕视角。脸部仍偏软；第6—7、39—41帧的头部方向或表情与输入不同，第27—30帧手持设备在脸前形成黑色拉丝；背面120—240度出现明显黑斑和缺失。',
    '同样渲染到原始476×960后，人物SSIM为0.90813（上一版0.90225），脸部L1为0.05703（上一版0.05773），人物L1为0.02958（上一版0.02731，较差）。指标有升有降，不能认定整体改善。',
    '本次采用官方MSV2的固定黑底合成图颜色损失，未加入人物轮廓不透明度损失。黑色高斯可以拟合黑背景，颜色正确不等于外部透明；这是后续需要单独验证的训练目标问题。此次不更改已训练模型，也不将质量状态标成通过。'
]
save(OUT/'review-findings.json',dict(updated=stamp,training_completed=True,format_passed=True,quality_accepted=False,summary_paragraphs=summary,
    reviewed_frames=[f'frame_{i:06d}' for i in range(1,60)],reviewed_orbit_degrees=list(range(0,360,30)),
    diagnostic=read(OUT/'background-diagnostic/diagnostic.json'),best_model_replaced=False,server_deployed=False))
bundle=OUT/'demo_MSV2_3倍_18000步_点云与相机.zip'
with zipfile.ZipFile(bundle) as z:
    assert z.read('point_cloud.ply')==(OUT/'demo_msv2_3x_18000_yup.ply').read_bytes()
    assert z.read('cameras.json')==(OUT/'cameras_yup.json').read_bytes()
cameras=read(OUT/'cameras_yup.json');assert len(cameras)==59
assert all('rotation' in c and 'w2c' not in c and c['width']==1428 and c['height']==2880 for c in cameras)
save(OUT/'viewer-import-verification.json',dict(updated=stamp,local_viewer='http://127.0.0.1:8870/',entry='Open files',file=bundle.name,
    zip_sha256=sha(bundle),points=95841,cameras=59,gaussian_mode=True,point_mode=True,SH3_mode=True,default_orientation_upright=True,
    gray_background_artifacts=True,initial_camera_ambiguity_fixed='3DGS array now declares only camera-to-world rotation/position; source w2c is in mapping file',
    camera_model_hash_binding='External archive hash verified; plain 3DGS array has no envelope binding hash',remote_00_opened=False,quality_accepted=False))
# Rebuild scientific contact sheets with the correct opacity terminology.
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
rows=read(Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/static-person-upsampling-20261102/demo/original/data/training-inputs.json'))['records']
for item in read(OUT/'background-diagnostic/diagnostic.json')['checks']:
    r=next(r for r in rows if r['frame']==item['frame']);folder=OUT/'background-diagnostic'
    native=Path('//wsl.localhost/Ubuntu-20.04'+r['image'])
    paths=[native,folder/(r['frame']+'_gray.png'),folder/(r['frame']+'_alpha.png')]
    canvas=Image.new('RGB',(900,660),'#eeeeee');d=ImageDraw.Draw(canvas)
    for j,(p,label) in enumerate(zip(paths,['原始输入','MSV2 灰底原生','MSV2 累计不透明度'])):
        with Image.open(p) as im:im=im.convert('RGB');im.thumbnail((300,600),Image.Resampling.LANCZOS)
        d.text((j*300+6,4),label,font=font,fill='black');d.text((j*300+6,28),r['frame'],font=font,fill='black');canvas.paste(im,(j*300+(300-im.width)//2,58))
    canvas.save(folder/(r['frame']+'_compare.png'))
print('MSV2_REVIEW_FINALIZED quality_accepted=false')
