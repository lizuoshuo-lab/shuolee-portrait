from pathlib import Path
from datetime import datetime
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

B=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
O=B/'current/background-architecture-109'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def local(s):return Path('C:/'+str(s)[7:]) if str(s).startswith('/mnt/c/') else Path(s)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def project(x,E,K):
    v=E[:3,:3]@x+E[:3,3];p=K@v;return p[:2]/p[2]-.5

assert not (O/'decision.json').exists()
result=read(O/'evaluation.json');details=read(O/'evaluation-tracks.json')
rows=read(B/'current/person-static-shared-face-fresh-098/data/training-inputs.json')['records']
with np.load(B/'current/shared-point-bundle-convergence-101/input.npz') as z:
    data={k:z[k] for k in ['observations','offsets','groups','original_track_ids','K']}
K=data['K'];index={int(j):k for k,j in enumerate(data['original_track_ids'])}
E={arm:np.load(O/(arm+'.npz'))['w2c'] for arm in ['fixed_cameras','background_bundle']}
records={arm:{r['id']:r for r in details[arm]['records']} for arm in E}
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
board=Image.new('RGB',(1200,420),'white');draw=ImageDraw.Draw(board);checks=[]
for col,(raw,g,label) in enumerate([(23,1,'下半脸'),(88,2,'上半脸'),(131,3,'躯干')]):
    f=raw-1;points=[]
    for id,r in records['fixed_cameras'].items():
        if f not in r['frames']:continue
        q=r['frames'].index(f)
        if r['held_group'][q]!=g:continue
        j=index[id];lo,hi=data['offsets'][j:j+2];obs=data['observations'][lo:hi];uv=obs[obs[:,0]==f,1:][0]
        p0=project(np.array(r['xyz']),E['fixed_cameras'][f],K);p1=project(np.array(records['background_bundle'][id]['xyz']),E['background_bundle'][f],K)
        points.append((id,uv,p0,p1))
    assert points, f'No held observations at{raw}:{g}'
    uv=np.array([p[1] for p in points]);center=np.median(uv,axis=0);x0=round(center[0])-200;y0=round(center[1])-145
    im=Image.open(local(rows[f]['image'])).convert('RGB').crop((x0,y0,x0+400,y0+290));d=ImageDraw.Draw(im)
    chosen=np.linspace(0,len(points)-1,min(len(points),8)).round().astype(int)
    for j in chosen:
        id,obs,p0,p1=points[j];u=obs-[x0,y0];a=p0-[x0,y0];b=p1-[x0,y0]
        d.line([tuple(u),tuple(a)],fill=(0,100,255),width=1);d.line([tuple(u),tuple(b)],fill=(255,0,130),width=1)
        d.ellipse((u[0]-3,u[1]-3,u[0]+3,u[1]+3),outline=(0,255,60),width=1)
        for p,color in [(a,(0,100,255)),(b,(255,0,130))]:
            d.line((p[0]-4,p[1],p[0]+4,p[1]),fill=color,width=1);d.line((p[0],p[1]-4,p[0],p[1]+4),fill=color,width=1)
    board.paste(im,(col*400,62));draw.text((col*400+7,4),f'{label}　原视频{raw:06d}',font=font,fill='#173140')
    e0=np.linalg.norm(np.array([p[2]-p[1] for p in points]),axis=1);e1=np.linalg.norm(np.array([p[3]-p[1] for p in points]),axis=1)
    draw.text((col*400+7,32),f'本图留出点{len(points)}个，原图比例裁片',font=small,fill='#173140')
    checks.append(dict(raw_frame=raw,region=label,held_observations=len(points),shown_ids=[int(points[j][0]) for j in chosen],crop=[x0,y0,x0+400,y0+290],median_fixed=float(np.median(e0)),median_background=float(np.median(e1))))
draw.text((7,362),'绿圈：原图观测　蓝十字：原相机投影　紫十字：背景校正后投影',font=font,fill='#173140')
draw.text((7,391),'共享三维点只用轨迹前后段拟合。图示为中段留出观测，不是新的高斯渲染结果。',font=small,fill='#173140')
board.save(O/'projection-review-slide.png')
save(O/'projection-review.json',dict(created=datetime.now().astimezone().isoformat(),selection='Predefined frames23/88/131 and lowerface/upperface/torso; up to8uniformtrackIDs, not ranked by error.',records=checks))
summary='仅由背景修正外参没有改善全身一致性：独立背景中位误差基本不变，躯干明显变差，下半脸未达到提升门槛。保留原074相机，不把这组诊断相机用于高斯训练。'
notes=[
    '109针对107独立审查提出的混杂问题，将相机拟合与人体观测分离。175张真实照片，固定原074相机内参。2684条已分组的背景轨迹中2127条用于拟合，557条整条留出。先检查全部175帧的静止结构区域，排除人物mask及边缘64像素、发光饱和区域、他人鞋尖、椅轮，以及已识别的圆形设备可变轮廓。原始二维对应仍有低纹理与重复纹理不确定性。',
    '两组独立使用相同的本轮全新点初始化。控制组固定相机、仅优化背景三维点；候选组优化外参与背景点。所有人体和557条背景验证轨迹均不进入相机求解。两组Soft-L1尺度2原图像素，最多100次迭代/240秒，均达到收敛。候选外参最大旋转变化1.348度，最大相机中心变化为参考中位深度的1.60%。正式相机文件没有改动。',
    '验收时，每条留出轨迹在两组相机下分别重新三角化和稳健拟合一个XYZ，仅用轨迹前后段，检查中段。背景557条轨迹共5945个留出观测，中位数0.493变0.499像素，p90从2.160到1.861。天花板中位数1.323变1.041，地毯0.358变0.378。背景整体中位数没有达到至少10%改善的门槛。',
    '人体独立观测：下半脸中位数3.512变3.289，p90为5.986变7.181；上半脸1.807变2.755，p90为5.607变6.021；躯干1.461变2.740，p90为4.453变6.510；腿鞋0.927变1.159，p90为3.453变3.810。下半脸20%提升及躯干p90最多增加0.5像素的标准均未通过。单位为完整1514x2694去畸变图像像素。',
    '下半脸的 pooled observation median小幅下降，不代表各物理轨迹普遍改善。53条下脸轨迹配对中位差为+0.235像素，轨迹bootstrap95%区间[-0.164,1.123]。躯干1938条轨迹配对中位差+0.862，区间[0.829,0.901]。近邻纹理轨迹仍有空间相关，区间不能作为外部相机真值。',
    '该结果否定当前这组背景外参校正作为修复方案，不能据此证明原相机完全准确或人体必然在动。背景定位歧义、相机模型误差、人体点对应不一致以及静态近似仍需区分。107指出的高斯有限支撑约束不足仍是已证实的实现缺陷。下一步固定原相机，核查这些材料轨迹在原生高斯中实际贡献点的位置与支撑梯度，不继续扫描相机/面部权重。没有新高斯训练、点云或00部署。',
    '来源：shuolee/current/background-architecture-109中的solver-protocol.json、两个*-result.json、evaluation.json、evaluation-tracks.json，以及projection-review.json登记的原照片裁片。图片为点重投影诊断，不是新的训练效果图。',
]
save(O/'decision.json',dict(created=datetime.now().astimezone().isoformat(),round='109',summary=summary,
    status='background_camera_candidate_rejected',numerical_gate_pass=False,source_cameras_changed=False,new_gaussian_training=False,new_point_cloud=False,latest_gaussian_round='098',quality_accepted=False,
    measurements=result,mechanism='Background-only camera fitting removes direct human residuals from the solve, yet does not transfer improvement to body. It does not identify a unique physical root cause.',
    next_bounded_work='Keep074K/E and inspect finiteGaussian contributor positions/support gradients on already measured material queries. Resolve the specific102initialempty-ray failure without claiming a completed training or changing all-camera coverage.',
    presentation_notes=notes))
print(json.dumps(dict(summary=summary,projection_review=checks),ensure_ascii=False))
