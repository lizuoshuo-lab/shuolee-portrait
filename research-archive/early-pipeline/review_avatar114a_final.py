"""Record actual 114a UI/image review and correct display-only background."""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
PAIR=B/'current/avatar-repository-pair-114'; ROOT=PAIR/'gaussianavatar'; V=B.parent/'viewer'
BUILD=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def win(s):return Path(s.replace('/mnt/c/','C:/',1))

assert read(ROOT/'gaussian-run/result.json')['steps']==18000
assert sha(Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee/avatar_pair_114.py'))=='0c4894e5d1d6c5e64560f558a01dfca736209de3a0dd420cda9f1cdada2db576'
ev=ROOT/'evaluation'; rows=read(ROOT/'data/training-inputs.json')['records']
obs=np.load(ROOT/'data/observations.npz',allow_pickle=False)
display_receipt=ev/'display-background-review.json'
if not display_receipt.exists():
    records=[]
    for i in [0,33,69,99,135,174]:
        row=rows[i];im=Image.open(win(row['image'])).convert('RGB');mask=Image.open(win(row['mask'])).convert('L')
        w=round(im.width*1536/im.height);h=1536
        im=im.resize((w,h),Image.Resampling.LANCZOS);mask=mask.resize((w,h),Image.Resampling.BILINEAR)
        alpha=np.asarray(mask,dtype=np.float32)/255
        rgb=np.asarray(im,dtype=np.float32)/255
        comp=np.rint(np.clip(rgb*alpha[:,:,None]+.25*(1-alpha[:,:,None]),0,1)*255).astype(np.uint8)
        gt=Image.fromarray(comp)
        face=obs['keypoints'][i,23:91]*np.array([w/row['wh'][0],h/row['wh'][1]])
        lo=np.maximum(0,np.floor(face.min(0)-[24,35]).astype(int));hi=np.minimum([w,h],np.ceil(face.max(0)+[24,25]).astype(int))
        yy,xx=np.where(alpha>.5);y0,y1=int(yy.min()),int(yy.max())+1
        boxes={'face':(tuple(np.r_[lo,hi]),(420,460)), 'body':((max(0,int(xx.min())-16),max(0,y0-16),min(w,int(xx.max())+17),min(h,y1+16)),(400,950))}
        for kind,(box,(W,H)) in boxes.items():
            p=ev/f'{kind}-{i+1:06d}.jpg';before=sha(p);panel=Image.open(p).convert('RGB')
            crop=gt.crop(box);crop.thumbnail((W,H),Image.Resampling.LANCZOS)
            panel.paste((35,35,35),(0,30,W,H+30));panel.paste(crop,((W-crop.width)//2,30+(H-crop.height)//2));panel.save(p,quality=95)
            records.append(dict(path=str(p),before_sha256=before,after_sha256=sha(p)))
    save(display_receipt,dict(created=datetime.datetime.now().astimezone().isoformat(),change='Photographic foreground displayed on the same 0.25 gray background as both renders; first column only.',camera_changed=False,crop_changed=False,metrics_changed=False,training_changed=False,files=records))

delivery=read(ROOT/'gaussian-run/viewer-export/delivery.json')
screens=[BUILD/'avatar114a-openfile-gaussian.png',BUILD/'avatar114a-openfile-points.png']
assert all(p.exists() for p in screens)
ui=dict(created=datetime.datetime.now().astimezone().isoformat(),file_opened_through_UI=True,format_verified=True,source_sha256=delivery['sha256'],observed_count=200000,modes=['Gaussian','points'],screenshots=[dict(path=str(p),sha256=sha(p)) for p in screens],observations='Actual Open File loaded the standalone +Y PLY in both modes. Coherent clothed body, correct point count and no additional local limb deformation from import; facial duplication and contour fragments already exist in native renders.',quality_accepted=False,pc00_opened_this_round=False)
save(ROOT/'viewer-ui-review.json',ui)

page=V/'body-smplx.html';html=page.read_text(encoding='utf-8')
old='首批1,000步已实际检查：穿衣全身轮廓已形成，脸部、镜框和衣服边缘仍模糊；不能据此判断重影已解决。继续同一轮预定训练。'
new='已完成18,000步：鼻嘴与眼镜仍有错位和模糊，肩袖及衣物轮廓仍有误差，重影质量未通过。175个训练视角的面部颜色误差为0.0335，113参考为0.0161；这不是独立测试集，也不能仅凭该数值判断几何。标准PLY和+Y版本已核验，本地“打开文件”的高斯与点云两种模式均已实际检查。'
if old in html:html=html.replace(old,new,1)
if 'id="avatar114a-final-review"' not in html:
    extra='<details id="avatar114a-final-review"><summary>GaussianAvatar最终同相机与连续视角对照</summary><p><a href="./?model=whole-body-avatar114-ga">打开114a高斯</a> · <a href="models/whole-body-avatar114-ga/shuolee_114a_s18000_f0100.ply?download=1">下载兼容PLY</a></p><figure><img src="body-avatar114/gaussianavatar/face-000100.jpg" alt="相同比例与背景的原图、113、GaussianAvatar面部"><figcaption>左：实拍前景；中：113；右：GaussianAvatar。保持同一相机、裁剪比例和灰色背景。</figcaption></figure><figure><img src="body-avatar114/gaussianavatar/body-000070.jpg" alt="固定相机全身对照"><figcaption>同一全身相机：观察肩袖、衣摆和腿脚边界。</figcaption></figure><video controls preload="metadata" style="width:100%" src="body-avatar114/gaussianavatar/continuous.mp4"></video><p>同一150帧检查路径；未知视角没有实拍真值。格式核验通过不代表质量验收。</p></details>'
    html=html.replace('<h3>GaussianAvatars_SMPLX</h3>',extra+'<h3>GaussianAvatars_SMPLX</h3>',1)
html=html.replace("(p.state==='training'?'训练中':", "(step>=18000?'训练完成，待质量对照':p.state==='training'?'训练中':",1)
page.write_text(html,encoding='utf-8')
print(json.dumps(dict(ui=str(ROOT/'viewer-ui-review.json'),display_fix=str(display_receipt),quality_accepted=False),ensure_ascii=False))
