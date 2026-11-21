"""Archive reviewed native images using D:/00/记录/规则.md, without refitting."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, re, shutil
import numpy as np
from PIL import Image, ImageDraw
import cv2

H = Path(__file__).parent
T = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = T/'shuolee/current/local-clothed-surface-123'
REF = REC/'smplifyx-reference-control'
O = REF/'conditioned-reference'
RUN = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
DEST = Path('/mnt/d/00/记录')
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p,v: p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

receipt = O/'image-record-export.json'
assert not receipt.exists(), 'Inspect the existing archive receipt instead of exporting twice.'
rules = DEST/'规则.md'
assert rules.is_file() and '只要图片不要文本' in rules.read_text(encoding='utf-8-sig')
plan = read(REC/'grass-preflight/result.json')
assert plan['source_sha256'] == '846558b3a023650b3dccd1fa55da4dc3cfde596ea87de09ab43ab494f5306e14'
source_video = Path('/mnt/d/00/工作/数据集/video5.mp4')
assert sha(source_video) == plan['source_sha256'], 'Dataset identity changed.'
match = re.fullmatch(r'video(\d+)',source_video.stem)
assert match
dataset = f'{int(match.group(1)):03d}'
dest = DEST/dataset
rows = read(RUN/'data/inputs.json')['records']
assert read(O/'result.json')['numeric_execution_passed']

# Reuse the actual reviewed mesh renderer, without executing its fit imports or
# regenerating historical outputs. This is a triangle preview, never a 3DGS image.
renderer = H/'review_smplify_corrected123.py'
tree = ast.parse(renderer.read_text(encoding='utf-8'))
mesh_node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mesh')
scope = {'np':np,'cv2':cv2,'Image':Image}
exec(compile(ast.Module(body=[mesh_node],type_ignores=[]),str(renderer),'exec'),scope)
mesh = scope['mesh']
edges = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
coco = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
files = []

def record(path, source, kind, frame, wh):
    with Image.open(path) as im:
        im.load()
        assert im.size==tuple(wh)
    files.append(dict(path=str(path),source=str(source),kind=kind,
                      source_frame_index=frame,wh=list(wh),sha256=sha(path)))

def copy_image(source, relative, kind, frame):
    target = dest/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(), str(target)
    shutil.copy2(source,target)
    assert sha(source)==sha(target)
    with Image.open(source) as im: wh=im.size
    record(target,source,kind,frame,wh)

def output(im,relative,source,kind,frame):
    target=dest/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(), str(target)
    im.save(target)
    record(target,source,kind,frame,im.size)

def overlay(raw,xy,conf,color):
    im=raw.copy(); draw=ImageDraw.Draw(im)
    for i,p in enumerate(xy):
        if conf[i]>.25:
            draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
    for a,b in edges:
        i,j=coco[a],coco[b]
        if conf[i]>.25 and conf[j]>.25:
            draw.line([tuple(xy[i]),tuple(xy[j])],fill=color,width=2)
    return im

for slot in [0,31,62]:
    row=rows[slot]; frame=row['source_frame_index']
    raw_path=Path(row['image']); mask_path=Path(row['mask'])
    assert sha(raw_path)==row['source_sha256'] and sha(mask_path)==row['mask_sha256']
    name=f'20261029_SMPLifyX数值修复_视频帧{frame:06d}.png'
    for sub in ['mask/原图','特征点/原图','smpl-x/原图']:
        copy_image(raw_path,Path(sub)/name,'native_original',frame)
    copy_image(mask_path,Path('mask/mask')/name,'stored_SAM2_mask',frame)
    raw=Image.open(raw_path).convert('RGB')
    fitpath=O/f'slot{slot:03d}-fit.npz'
    with np.load(fitpath) as z: fit=dict(z)
    conf=fit['confidence'][0]
    output(overlay(raw,fit['targets'][0],conf,(0,255,80)),
           Path('特征点/2d特征点')/name,fitpath,'detector_target_overlay',frame)
    output(overlay(raw,fit['projected'][0],conf,(255,0,220)),
           Path('特征点/空间特征点')/name,fitpath,'projected_SMPLX_points_overlay',frame)
    output(mesh(raw,np.array([0,0,raw.width,raw.height]),fit),
           Path('smpl-x/smpl-x')/name,fitpath,'CPU_triangle_overlay_not_Gaussian',frame)
    for region,label in [('body','全身'),('face','面部'),('left_hand','左手'),('right_hand','右手')]:
        copy_image(O/f'result_slot{slot:03d}_{region}_raw_initial_final_mesh.png',
                   Path('分析')/label/f'20261029_视频帧{frame:06d}_原图_初始投影_最终投影_SMPLX.png',
                   'reviewed_unwarped_four_panel',frame)
copy_image(O/'three_views_raw_mesh.png',
           Path('分析/视角对比/20261029_三视角_原图与SMPLX.png'),
           'reviewed_three_view_montage',None)
assert all(p.suffix.lower() in ('.png','.jpg','.jpeg') for p in dest.rglob('*') if p.is_file())
save(receipt,dict(updated=datetime.now(timezone.utc).isoformat(),
     rule=str(rules),rule_sha256=sha(rules),source_video=str(source_video),
     source_video_sha256=sha(source_video),dataset=dataset,destination=str(dest),
     source_frame_indices=[0,155,310],images=len(files),files=files,
     no_text_on_images=True,only_images_in_dataset_record=True,
     native_image_pixels_and_dimensions_preserved=True,
     overlays_use_saved_points_no_new_fit=True,
     point_display_rule='same existing confidence>.25 display rule; does not establish visibility',
     gaussian_images=0,reason_no_gaussian='This SMPLify-X diagnostic has not trained Gaussians.',
     no_new_images_written_to_work_log=True,previous_source_artifacts_preserved=True,
     WPS_cloud_sync_verified=False))
print(json.dumps({'dataset':dataset,'images':len(files),'destination':str(dest),
                  'gaussian_images':0,'receipt':str(receipt)},ensure_ascii=False))
