from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, math, re, shutil
from pypdf import PdfReader

H = Path(__file__).parent
T = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
D = R / 'camera-document-review-20261028'
V, S = T/'viewer', Path(r'D:\00\工作\查看器')
L = Path(r'D:\00\工作日志\2026-10-27\内参文件解析')
C = H.parent/'shuolee-coordination'
stamp = datetime.now(timezone.utc).isoformat()

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

assert not D.exists()
D.mkdir()
yaml_path = Path(r'D:\00\工作\数据集\mono_main_calib.yaml')
diagram = Path(r'D:\00\工作\数据集\20261027-114356.jpg')
pdfs = [Path(r'C:\Users\Administrator\Downloads\你怎么知道这两个相机是干什么的，怎么确定的_2026_10_28.pdf'),
        Path(r'C:\Users\Administrator\Downloads\解释一下这个鱼眼相机，这些参数是干什么用的_2026_10_28(1).pdf')]
sources = []
for p in pdfs:
    pdf = PdfReader(str(p))
    sources.append(dict(path=str(p), sha256=sha(p), pages=[dict(page=i+1, text=x.extract_text()) for i,x in enumerate(pdf.pages)],
                        origin='User supplied explanatory PDF; not an official export log. Both pages visually reviewed.',
                        source_authority='Explanation only; rendered footer says AI-generated; conclusions checked individually rather than accepted by origin.'))
    shutil.copy2(p, D/p.name)
save(D/'sources.json', sources)
save(D/'plan.json', dict(updated=stamp, stage='read_only_document_and_projection_algebra',
    inputs={str(p):sha(p) for p in [yaml_path,diagram,*pdfs]},
    action='Check field claims against YAML and calculate only same-ray change between its two numerical matrices after the SAME conditional affine transform. No fitting, images, cameras or renderer modifications.',
    new_user_evidence='Camera team reportedly says the first matrix is used for undistortion followed by resize and crop. Diagram numbers may contain typos; use measured video dimensions, not1532.',
    no_new_candidate=True, no_training=True))
raw = yaml_path.read_text(encoding='utf-8-sig')
def matrix(key):
    segment = re.search(r'^'+re.escape(key)+r':\s*!!opencv-matrix\s*(.*?)(?=^\w|\Z)', raw, re.M|re.S).group(1)
    values = [float(x) for x in re.search(r'data:\s*\[([^]]+)\]', segment, re.S).group(1).split(',')]
    assert len(values)==9
    return [values[0:3], values[3:6], values[6:9]]
K, F = matrix('camera_matrix'), matrix('camera_matrix_fov')
ratio = [F[0][0]/K[0][0], F[1][1]/K[1][1]]
n4path = R/'video-update-20261027/output-intrinsics-audit/result.json'
n4 = read(n4path)
Kout = n4['nominal_contract']['K']
h0 = next(v for v in n4['variants'] if v['id']=='H0')
pixels = h0['face_pixel_samples']['reference_pixels']
cx, cy = Kout[0][2], Kout[1][2]
def difference(p):
    return [(p[0]-cx)*(ratio[0]-1), (p[1]-cy)*(ratio[1]-1)]
face_delta = [difference(p) for p in pixels]
face_norm = [math.hypot(*p) for p in face_delta]
assert K[0][2] == F[0][2] and K[1][2] == F[1][2]
assert difference([cx,cy]) == [0.,0.]
numerics = dict(focal_relative_change_percent=[100*(x-1) for x in ratio],
                camera_matrix=K, camera_matrix_fov=F,
                conditional_output_K=Kout, conditional_output_K_fov=[[Kout[0][0]*ratio[0],0,cx],[0,Kout[1][1]*ratio[1],cy],[0,0,1]],
                face_reference_pixels=pixels, face_same_ray_delta_px=face_delta,
                face_same_ray_displacement_min_px=min(face_norm), face_same_ray_displacement_max_px=max(face_norm),
                meaning='Algebraic sensitivity only. Not measured reprojection errors; not evidence the second matrix was used. No model selection or camera fitting.')
claims = [
    dict(topic='是否鱼眼', verified='YAML明确camera_model=FISHEYE_8。', correction='4000×3000分辨率或_fov名称不能证明鱼眼或设备品牌。'),
    dict(topic='两组矩阵用途', verified='同一camera_name下保存两组3×3矩阵，主点相同、焦距不同。相机方经用户转述：第一组用于去畸变再缩放裁剪。', correction='PDF断言_fov是原始K、第一组是输出K，缺少代码或字段定义。不能由焦距大小推出用途；也不能反向断言。'),
    dict(topic='输出K', verified='应使用实际导出图像对应的投影及K。OpenCV明确区分原始K与输出P/Knew。', correction='“用于去畸变”若未指出传给K还是P/Knew，技术上仍有歧义；仅需相机方导出实际输出K/最终视频K，不必再提供笼统流程。'),
    dict(topic='主点和分辨率', verified='YAML已经直接给出4000×3000；cx/cy为像素主点。', correction='cx/cy不是三维光心，也不必等于图像中心；裁剪后不能用2cx/2cy推断分辨率。'),
    dict(topic='畸变及尺度', verified='文件有8个厂家鱼眼系数。', correction='不能按系数个数直接套OpenCV四系数鱼眼或八系数Brown模型；3×3K本身不保证已校正畸变，也不能单独确定单目重建的绝对尺度。'),
    dict(topic='处理示意图', verified='流程是去畸变后下采样再裁剪；本次图示4096×3072→3605×2704→中心竖幅。用户再次说明数值可能写错。', correction='新版MP4实测1520×2704，按实际尺寸处理，1532不作为运行参数。4000对应K如何用于4096输出、整数crop偏移仍需区分。'),
]
needed = [
    '最省事：每段MP4的最终3×3内参K及对应像素宽高；说明全片固定还是逐帧变化。',
    '如无最终K：实际去畸变输出P/Knew及其输出宽高（不是仅给输入标定K）；下采样后整数宽高；裁剪框(x,y,width,height)，x/y从左上角以0为起点。',
    '附缩放所用函数/配置行，以确认半像素中心约定；若有旋转或翻转，一并记录。已确认无新增防抖，不重复询问。',
    '这些信息用于消除映射歧义，仍须实拍独立重投影验证；没有导出日志也继续现有近似K的有界诊断，不无限等待。',
]
result = dict(updated=stamp, state='read_and_cross_checked', sources=[dict(path=x['path'],sha256=x['sha256']) for x in sources],
              yaml_sha256=sha(yaml_path), diagram_sha256=sha(diagram), claims=claims, numerics=numerics,
              actual_native_size=[1520,2704], diagram_1532_treated_as_unreliable_annotation=True,
              requested_information=needed,
              references=[dict(url='https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html',
                  verified='Official OpenCV docs distinguish sourceK/D and outputP/Knew for undistort; fov_scale changes the new focal. Does not define this manufacturer camera_matrix_fov field.')],
              new_K_promoted=False, new_training=False, assumptions_not_physical_calibration=True)
save(D/'result.json', result)
asset = '122-camera-document-review'
roots = [V/asset,S/asset,L/asset,V/'new-capture-118/calibration-note'/asset,S/'new-capture-118/calibration-note'/asset]
for root in roots:
    root.mkdir(exist_ok=True)
    for p in [D/'result.json',D/'sources.json',*list(D.glob('*.pdf'))]: shutil.copy2(p,root/p.name)
rows = ''.join('<tr><td>'+x['topic']+'</td><td>'+x['verified']+'</td><td>'+x['correction']+'</td></tr>' for x in claims)
section = f'''<!-- ROBUST122_CAMERA_DOCUMENT_BEGIN --><h4 id="robust122-camera-document">2026-10-28：相机方说明、两份PDF与实际输出K</h4>
<p>新增依据：用户转述相机方确认第一组矩阵用于去畸变，之后下采样、裁剪成视频。用户再次说明示意图数字可能写错。<strong>流程说明采用；运行尺寸采用MP4实测1520×2704，不使用图上的1532。</strong>现有N5/N6用的是第一组矩阵推导的条件K，没有改用camera_matrix_fov。没有再次去畸变，也没有修改已完成结果。</p>
<table border="1" cellpadding="7" style="border-collapse:collapse"><tr><th>项目</th><th>已核实或明确来源</th><th>需要纠正或仍未确定</th></tr>{rows}</table>
<p>两份PDF的矩阵数字读取正确，但“4K说明是鱼眼”“较小焦距一定是原始K”“_fov一定对应原始画面”等推断不能当作标定证据。相机方实际处理说明优先于这些解释。官方OpenCV接口把原始K与去畸变输出P/Knew分开；文件名本身无法确定传给了哪个参数。<a href="https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html">官方接口定义</a>。</p>
<p>数值敏感度：两组焦距相差约{abs(100*(ratio[0]-1)):.3f}%。若对同一射线仅替换这组焦距，沿同一缩放裁剪链，9个既有脸部参考像素的位置会改变{min(face_norm):.2f}–{max(face_norm):.2f}px。<strong>这是参数差异的代数影响，不是测得的相机误差，也不能证明应该切换到另一组K。</strong></p>
<p>最省事的补充资料是每段MP4的最终K及对应分辨率。若没有最终K，只需导出一条记录：实际P/Knew、去畸变输出宽高、下采样后整数宽高、crop(x,y,w,h)；附缩放函数/配置即可核对像素中心约定。图示4096×3072与YAML4000×3000之间的K处理、实际整数裁剪偏移是剩余关键。1532按不可靠标注处理，不反复等待数字勘误。</p>
<p>缺少导出记录时继续按明确假设做有界诊断，不把近似K写成物理精确值。仍需分开验证输出映射、背景相机、材料身份和人物微动；新K与人物训练均未获质量准入。<a href="{asset}/result.json">逐项核对与全部数值</a> · <a href="{asset}/sources.json">两份PDF逐页摘录及来源哈希</a>。</p><!-- ROBUST122_CAMERA_DOCUMENT_END -->'''
reports = [V/'body-smplx.html', S/'progress.html', L/'01_内参文件解析与待核实项.html']
for p in reports:
    text = p.read_text(encoding='utf-8-sig')
    assert '<!-- ROBUST122_CAMERA_DOCUMENT_BEGIN -->' not in text and text.count('<!-- ROBUST122_END -->')==1
    p.write_text(text.replace('<!-- ROBUST122_END -->',section+'<!-- ROBUST122_END -->'),encoding='utf-8',newline='\n')
for root in [V/'new-capture-118/calibration-note', S/'new-capture-118/calibration-note']:
    shutil.copy2(reports[-1],root/reports[-1].name)
for root in [V,S]:
    p = root/'latest-result.json'
    v = read(p); before = {k:t for k,t in v.items() if k!='latestDiagnostic'}
    v['latestDiagnostic'].update(summary='已核对相机方第一组K说明与处理示意图；按MP4实测1520×2704处理，1532不作为参数。PDF对两组K用途的推断不作标定证据；仍需确认实际输出K和整数裁剪。119模型保留。', reportUrl='body-smplx.html#robust122-camera-document')
    save(p,v); assert before=={k:t for k,t in read(p).items() if k!='latestDiagnostic'}
next_text = 'Continue bounded camera-mapping diagnosis using first-matrix conditionalK and measured video dimensions. User accepts approximate diagram; do not wait/reask about1532. Camera team firstK report is input evidence, not official raw/rectified field specification. FinalK or outputP/size/resize/crop can settle ambiguity. Do not repeatN5/N6 or switchK silently.'
q = read(R/'queue-state.json'); q.update(updated=stamp,phase='camera_source_documents_reviewed_pending_UI',running=None,next=next_text)
q['completed'].append('Two1-pagePDFs rendered/read and checked against YAML; new camera-team/diagram steering recorded; read-only firstK/fovK sensitivity, no parameter promotion.')
q['last_goal_turn']['remaining'] = 'Actual updated document UI; spatial/time/material residual diagnosis and all-frame admission. No waiting solely on approximate diagram.'
save(R/'queue-state.json',q)
protocol=read(R/'protocol.json');protocol['camera_source_update_20261028']=dict(user_report='First matrix used for undistortion followed by resize/crop; diagram numbers approximate', actual_video_size=[1520,2704], currentK='First matrix remains conditional; its exact outputraster association not proved by PDFs', no_stabilization_added=True, next=next_text);save(R/'protocol.json',protocol)
ledger=read(R/'evidence-ledger.json');ledger['updated']=stamp;ledger['confirmed'].append('Two supplied PDFs individually crosschecked: matrixvalues correct, raw-vs-rectified role assignment unsupported. User relays camera-team firstK workflow and accepts diagram numeric typos; actual1520×2704 used. NumericalK comparison is algebra only, no new fit or training.');ledger['artifacts'].append('camera-document-review-20261028/result.json');save(R/'evidence-ledger.json',ledger)
s=read(C/'local-research-status.json');s.update(updated_at=stamp,source_queue_updated=stamp,next=next_text,camera_document_review=str(D/'result.json'));s['camera_source_update_20261028']=protocol['camera_source_update_20261028'];save(C/'local-research-status.json',s)
plan=T/'shuolee/long-term-plan.md';text=plan.read_text(encoding='utf-8-sig');text=text.replace('<!-- ROBUST122_END -->','\n2026-10-28：相机方经用户转述第一组K用于去畸变，再缩放裁剪；示意图允许数字笔误，以MP4实际1520×2704为准。已审阅两份解释PDF，其原始/输出K用途断言缺少依据。现有第一组条件K继续用于有界诊断；不把另一组焦距差异当作切换依据，不等待1532勘误。若有最终MP4 K或输出P/尺寸/resize/crop日志，可消除剩余坐标合同歧义。\n<!-- ROBUST122_END -->');plan.write_text(text,encoding='utf-8',newline='\n')
save(D/'publication.json',dict(updated=stamp,reports=[str(p) for p in reports],latest119_unchanged=True,UI_verified=False,PPT_content='N5/N6 actualcomparison alreadypublished toexisting52-pagePPT; PDF-onlyclarifications inexistingworklog, no repeated deck.'))
print(json.dumps(dict(result=str(D/'result.json'),face_sensitivity_range=[min(face_norm),max(face_norm)],focal_change_percent=numerics['focal_relative_change_percent'],training=False),ensure_ascii=False))
