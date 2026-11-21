"""Parse the provided calibration and write a lightweight, evidence-labelled work log."""
from pathlib import Path
import json,hashlib,subprocess,shutil,html
from datetime import datetime,timezone
import cv2,numpy as np

SRC=Path('/mnt/d/00/工作/数据集')
OUT=Path('/mnt/d/00/工作日志/2026-10-27/内参文件解析')
OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
source=SRC/'mono_main_calib.yaml'
fs=cv2.FileStorage(str(source),cv2.FILE_STORAGE_READ)
assert fs.isOpened()
K=fs.getNode('camera_matrix').mat();Kf=fs.getNode('camera_matrix_fov').mat();D=fs.getNode('distortion_coefficients').mat().ravel()
boards={}
for name in ['board_01','board_02','board_03']:
    node=fs.getNode(name)
    boards[name]=dict(rvec=node.at(0).mat().ravel().tolist(),tvec=node.at(1).mat().ravel().tolist(),
        translation_unit='not specified',video_frame_mapping='absent')
parsed=dict(image_width=int(fs.getNode('image_width').real()),image_height=int(fs.getNode('image_height').real()),
    camera_name=fs.getNode('camera_name').string(),camera_model=fs.getNode('camera_model').string(),
    camera_matrix=K.tolist(),camera_matrix_fov=Kf.tolist(),distortion_coefficients=D.tolist(),
    board_poses=boards,reprojection_error=float(fs.getNode('reprojection_error').real()),calib_by_raw_value=3)
fs.release()
videos=[]
for p in sorted(SRC.glob('*.mp4')):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt,color_space,color_transfer:format=duration','-of','json',str(p)],text=True))
    stream=info['streams'][0]
    videos.append(dict(file=p.name,bytes=p.stat().st_size,**stream,duration_seconds=float(info['format']['duration']),
                       calibration_applicability_confirmed=False,current_training_input=p.name.startswith('df776c03')))
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records']
Ki=np.array(rows[0]['K_integer_centers']);Kc=np.array(rows[0]['K'])
assert np.array_equal(Ki[:2,:2],Kc[:2,:2]) and np.allclose(Kc[:2,2]-Ki[:2,2],.5)
crop_h=4000*1520/2704;x0=0.;y0=(3000-crop_h)/2
def mapped(width,height,crop):
    x,y,w,h=crop;sx,sy=width/w,height/h
    A=np.array([[sx,0,sx*(.5-x)-.5],[0,sy,sy*(.5-y)-.5],[0,0,1.]])
    return A,A@K
A,Kcheck=mapped(1600,899,(x0,y0,4000,crop_h));assert np.allclose(Kcheck,Ki,atol=1e-9)
Av,Kv=mapped(2704,1520,(x0,y0,4000,crop_h))
Av9,Kv9=mapped(1080,1920,((4000-3000*9/16)/2,0,3000*9/16,3000))
record=dict(created=datetime.now(timezone.utc).isoformat(),source_path='D:/00/工作/数据集/mono_main_calib.yaml',
    source_bytes=source.stat().st_size,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parsed=parsed,videos=videos,
    current_hypothesis=dict(name='center_crop_then_resize_of_rectified_pinhole_export',physical_calibration_accepted=False,
        factory_D8_applied=False,source_pixel_centers='assumed integer, not vendor-confirmed',
        ideal_landscape_crop_xywh=[x0,y0,4000,crop_h],ideal_crop_is_not_verified_actual_integer_ROI=True,
        encoded_video_K_integer=Kv.tolist(),decoded_training_K_integer=Ki.tolist(),renderer_K_corner=Kc.tolist(),
        decoded_wh=[1600,899],pixel_map=A.tolist(),frames=89,total_video_frames=884,
        extrinsic_source='Fresh OpenCV Essential/PnP + fixed-K bundle adjustment; no manufacturer per-frame extrinsics',
        camera_file_sha256=hashlib.sha256((ROOT/'data/training-inputs.json').read_bytes()).hexdigest(),
        camera_file_wsl=str(ROOT/'data/training-inputs.json'),reference_pose_scale='arbitrary monocular gauge, not verified metric scale'),
    portrait_example_only=dict(K_integer=Kv9.tolist(),warning='Hypothetical full-height centered9:16 crop only; NOT applied to online4; actual crop/rotation/export unknown.'),
    missing=['FISHEYE_8 projection formula/coefficient ordering/units/coordinate convention',
        'Meaning and intended output domain of camera_matrix_fov',
        'Camera serial/lens/mode/firmware applicability per video',
        'Actual crop ROI, resize, rotation, EIS/undistort/digital zoom and per-frame changes',
        'Video per-frame extrinsics/time sync/rolling shutter timing and metric scale',
        'Independent calibration/export validation observations'],
    quality_accepted=False)
(OUT/'02_参数解析与视频清单.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copy2(source,OUT/'03_mono_main_calib_原件.yaml')
assert hashlib.sha256((OUT/'03_mono_main_calib_原件.yaml').read_bytes()).hexdigest()==record['source_sha256']

esc=lambda x:html.escape(str(x))
def mat(x):return '<pre>'+esc('\n'.join('  '.join(f'{v:.12f}' for v in r) for r in x))+'</pre>'
video_rows=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['file'],f"{r['width']}×{r['height']}",r.get('nb_frames','未知'),f"{r['duration_seconds']:.3f}秒",r.get('pix_fmt',''),('当前118' if r['current_training_input'] else '未启动本轮训练'),'尚未确认'])+'</tr>' for r in videos)
board_rows=''.join('<tr><td>'+esc(name)+'</td><td>'+esc(b['rvec'])+'</td><td>'+esc(b['tvec'])+'</td></tr>' for name,b in boards.items())
body=f'''<h1>mono_main_calib.yaml：内参解析与待核实信息</h1>
<p class="date">工作日志 · 2026-10-27 · 适用记录：新人物118。此文件解析与当前实测状态分开列出；后续得到厂家说明时在本份更新。</p>
<div class="lead"><strong>已经得到两组3×3相机矩阵、8个畸变系数、3组标定板位姿和一个标定误差值。</strong>尚未得到能直接用于各段导出视频的完整标定合同，也没有视频逐帧真实外参。当前118使用主矩阵K按居中裁切/缩放假设换算后的固定焦距与主点；8个畸变系数尚未应用，物理标定验收仍未通过。</div>
<h2>1. 文件里实际有什么</h2>
<table><tr><th>字段</th><th>原值</th><th>能说明什么／限制</th></tr>
<tr><td>image_width / image_height</td><td>4000 / 3000，4:3</td><td>文件定义的标定图像尺寸；不是现有MP4编码尺寸。不能只把K原样复制给视频。</td></tr>
<tr><td>camera_name</td><td>MONO_MAIN_MFR</td><td>相机名字；文件没有序列号、拍摄模式或视频文件关联。</td></tr>
<tr><td>camera_model</td><td>FISHEYE_8</td><td>模型名称，文件未附投影公式。仅凭8个数，不能认定是OpenCV rational或OpenCV fisheye模型。</td></tr>
<tr><td>camera_matrix</td><td>fx={K[0,0]:.9f}, fy={K[1,1]:.9f}<br>cx={K[0,2]:.9f}, cy={K[1,2]:.9f}</td><td>焦距和主点单位为此图像域的像素。fx/fy不是毫米；主点接近图像中心。实际投影还依赖镜头模型。</td></tr>
<tr><td>camera_matrix_fov</td><td>fx={Kf[0,0]:.9f}, fy={Kf[1,1]:.9f}<br>主点与上面相同</td><td>第二套矩阵。名称可能与视场设置相关，但厂家含义未确认，不能擅自当作导出去畸变矩阵或直接替换K。</td></tr>
<tr><td>distortion_coefficients</td><td>8×1，见下表</td><td>数值齐全；公式、顺序、适用图像域仍缺。系数数值大或交替正负，本身不能证明错误。</td></tr>
<tr><td>reprojection_error</td><td>{parsed['reprojection_error']:.12f}</td><td>文件报告的重投影误差；通常以像素报告，但文件未定义计算口径。不是新视频的实测误差，也不能证明新视频外参精度。</td></tr>
<tr><td>calib by</td><td>3</td><td>原始元数据值；含义未说明，不据此推断相机数量或标定步骤。</td></tr></table>
<details><summary>两组完整矩阵</summary><p>camera_matrix</p>{mat(K)}<p>camera_matrix_fov</p>{mat(Kf)}</details>
<h3>8个畸变系数：保持原顺序</h3><table><tr><th>文件序号</th><th>原值</th></tr>{''.join(f'<tr><td>{i+1}</td><td>{x:.17g}</td></tr>' for i,x in enumerate(D))}</table>
<p>不把它们自行命名为k1/k2/p1/p2，也不截取前4个传给OpenCV fisheye。下一步需要厂家提供FISHEYE_8从三维射线到像素的公式，或参考实现及对应去畸变输出配置。</p>
<h3>三组board位姿</h3><p>文件注释为“0-top left、1-top right、2-bottom；rvec在tvec前”。它们属于标定板记录，不能填入视频各帧的R/t。平移单位、板坐标系、姿态方向以及编号与采集时刻关系未注明；3分量rvec按OpenCV惯例通常是Rodrigues向量，正式使用仍应核对厂家定义。</p>
<table><tr><th>名称</th><th>rvec原值</th><th>tvec原值</th></tr>{board_rows}</table>
<h2>2. 现在收到哪些视频</h2><p>以下分辨率/帧数/时间来自本机ffprobe读取文件，不由文件名或“4K”描述推断。</p>
<table><tr><th>文件</th><th>编码尺寸</th><th>帧数</th><th>时长</th><th>像素格式</th><th>本轮用途</th><th>与标定匹配</th></tr>{video_rows}</table>
<p>三段横屏约16:9，online4为9:16。4:3→16:9通常需上下裁切，4:3→9:16通常需左右裁切；也可能有先旋转、去畸变或EIS。用户确认“应该居中”，同时表示实际裁边和去畸变不确定，因此此处只登记为假设。online4没有套用118的K。</p>
<h2>3. 当前118到底怎么用了这份内参</h2>
<ol><li>选择白衣人物视频 <code>df776c03-91a4_1792572380952.mp4</code>，原884帧、2704×1520。</li>
<li>当前研究采样每10帧取1帧，共89帧，覆盖整段环绕；整图解码缩放为1600×899。没有声称884帧全部完成外参估计。</li>
<li>在“导出画面已可作为针孔图像”的未核实假设下，用文件主矩阵camera_matrix作为固定K来源。按理想居中裁切4000×{crop_h:.9f}，左上角(0,{y0:.9f})，再缩放到解码尺寸。这个带小数的ROI是计算模型，尚不是相机真实裁边记录。</li>
<li>固定换算后的K，使用OpenCV Essential/RANSAC、PnP和固定K束调整从头求89帧各自外参；没有把所有外参强制相同。当前不使用COLMAP优化后的旧位姿或旧三维点；已有SIFT描述子作为图像特征复用。</li>
<li>89帧原解码图上重新提取mask、全身特征点和GVHMR先验，再从头拟合共享静态人体。没有对图像做二维扭曲来掩盖投影错位。</li></ol>
<h3>像素映射：焦距和主点必须一起换算</h3>
<pre>零基整数像素中心约定：
u_new = sx × (u_old - x0 + 0.5) - 0.5
v_new = sy × (v_old - y0 + 0.5) - 0.5
fx_new = sx × fx_old       fy_new = sy × fy_old
cx_new = sx × (cx_old - x0 + 0.5) - 0.5
cy_new = sy × (cy_old - y0 + 0.5) - 0.5
K_new = A × K_old</pre>
<p>源文件采用哪种像素中心约定仍须厂家确认；上述是本工程当前明确采用的约定。解码高899由比例取整得到，因此sx与sy分别计算，不能用同一个尺度近似两个方向。</p>
<p>理想横屏导出2704×1520的K（整数中心，假设值）：</p>{mat(Kv)}
<p>当前实际输入1600×899的K（检测器整数中心）：</p>{mat(Ki)}
<p>当前渲染器K（角点约定，主点加0.5，投影时再减0.5）：</p>{mat(Kc)}
<p><strong>已验证的是实现中的同一射线映射一致：</strong>89帧解析投影等价误差最大2.274×10⁻¹³像素；高斯训练入口还逐帧核对实际送入CUDA的投影矩阵。这个数值测试不能证明厂家裁切、物理镜头模型或估计外参正确。</p>
<p><strong>未做：</strong>没有把未知8个畸变系数套进其他模型，也没有对当前原始导出图重复去畸变。零额外畸变只是“已矫正导出”的研究假设，不能称已使用完整FISHEYE_8标定。</p>
<h2>4. 已核实与仍待核实</h2>
<table><tr><th>优先级／问题</th><th>现在知道什么</th><th>应拿什么核实</th><th>影响</th></tr>
<tr><td>P0 FISHEYE_8公式</td><td>8个数及顺序已读取</td><td>厂家投影/逆投影参考代码；角度/归一化半径定义；系数顺序</td><td>无法正确应用原镜头畸变</td></tr>
<tr><td>P0 导出图像域</td><td>MP4尺寸已读取</td><td>是否去畸变、输出相机矩阵、鱼眼/线性模式、EIS/锁定视角、变焦设置</td><td>决定K是否能固定、D是否应施加</td></tr>
<tr><td>P0 裁切与缩放</td><td>用户倾向居中，实际裁边未知</td><td>4000×3000→各MP4的实际ROI、旋转、缩放顺序、逐帧变动或导出配置</td><td>影响主点、焦距和边界对齐</td></tr>
<tr><td>P0 标定适用对象</td><td>camera_name是MONO_MAIN_MFR</td><td>同一机身/镜头/分辨率/固件/模式确认，视频与标定一一对应</td><td>避免拿另一模式的K训练</td></tr>
<tr><td>P1 camera_matrix_fov语义</td><td>焦距较主矩阵略小、主点相同</td><td>厂家字段说明、该矩阵对应输出的尺寸及用途</td><td>决定该用哪组K</td></tr>
<tr><td>P1 逐帧真实外参与时钟</td><td>文件只有3组board位姿；现用图像估计R/t</td><td>原始采集包、逐帧相机姿态、坐标方向、时间戳、相机/IMU变换和同步说明（若支持）</td><td>约束真实相机运动；不能把板位姿当视频外参</td></tr>
<tr><td>P1 度量尺度和滚动快门</td><td>单目重建尺度未物理核验</td><td>已知长度、标定板尺寸/平移单位；曝光/逐行读出时序</td><td>尺度、运动扭曲与几何精度</td></tr>
<tr><td>P1 独立图像证据</td><td>背景/身体保护门槛尚未过</td><td>同模式导出标定板帧；原图与导出图对应；真实静止物理点跨视角检查</td><td>区分镜头、对应错误及人物微动</td></tr></table>
<h2>5. 当前误差证据与训练范围</h2>
<p>固定K候选拟合点平均误差约0.385像素，但独立留出观测中：脸部中位误差0.768→0.499像素，身体0.432→0.542、背景共同子集0.309→0.385，可观测轨迹248→245。它没有通过全身/背景保护门槛，不能称已精准标定。</p>
<p>新的同输入人体对照中，自身表面可见性筛选并未更好：同一组脸部观测中位误差4.277→4.298像素，脚部18.435→26.244像素。因此本轮采用参考拟合做一次有早期审查的研究训练：200000初始高斯、空优化器、最多18000步，第1000步实际检查鼻子与腿脚；明显失败就停止。图片、mask、相机与此前旧探针不混用，旧PLY保留原身份。</p>
<p>上述不同表格来自不同观测集合，不能把拟合误差、独立留出误差与人体检测器误差直接比较。人物脚部在沙发/桌边有遮挡，检测点也有歧义；不能把所有错误都归为K或外参，更不能用增加训练步数代替定位。</p>
<h2>6. 同目录文件与复核方式</h2><ul>
<li><a href="02_参数解析与视频清单.json">02_参数解析与视频清单.json</a>：完整数值、视频清单、当前映射和待核实项，可供工程读取。</li>
<li><a href="03_mono_main_calib_原件.yaml">03_mono_main_calib_原件.yaml</a>：原文件逐字节副本，没有改动系数。</li>
<li><a href="04_原件SHA256.txt">04_原件SHA256.txt</a>：原件校验值。</li></ul>
<p>原件：<code>D:\00\工作\数据集\mono_main_calib.yaml</code>；{source.stat().st_size}字节。SHA-256：<code>{record['source_sha256']}</code>。</p>
<p>日志只存小型解析和证据，不复制大模型/环境。WPS负责同步本目录；本机写入完成不代表00已下载或已完成云端核验。</p>'''
css='''body{font:16px/1.75 "Microsoft YaHei",sans-serif;background:#eef2f7;color:#243347;margin:0}main{max-width:1150px;margin:28px auto;padding:32px 44px;background:white;border-radius:12px}h1{font-size:28px}h2{margin-top:36px;border-top:1px solid #d8e2ec;padding-top:18px}h3{font-size:19px}table{border-collapse:collapse;width:100%;font-size:14px;overflow-wrap:anywhere}th,td{border:1px solid #d7e0eb;padding:9px;text-align:left;vertical-align:top}th{background:#edf3fa}.lead{background:#eaf3ff;padding:18px;border-left:5px solid #2872bd}.date{color:#65758b}pre{background:#f3f6fa;padding:16px;overflow:auto;line-height:1.6}code{overflow-wrap:anywhere;font-size:13px}a{color:#125eac}@media print{body{background:white}main{margin:0;padding:12px}tr{break-inside:avoid}}'''
(OUT/'01_内参文件解析与待核实项.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>内参文件解析与待核实项</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
(OUT/'04_原件SHA256.txt').write_text(record['source_sha256']+'  03_mono_main_calib_原件.yaml\n',encoding='utf-8')
shutil.copy2(OUT/'02_参数解析与视频清单.json',REC/'calibration-interpretation.json')
print(json.dumps(dict(saved=str(OUT),source_unchanged=True,videos=videos,K_integer=Ki.tolist()),ensure_ascii=False,indent=2))
