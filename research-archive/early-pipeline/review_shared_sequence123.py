"""Native integer crops for the registered shared-identity trial; no fitting."""
from pathlib import Path
import ast,json
import numpy as np
from PIL import Image,ImageDraw
import cv2
import run_smplify_reference123 as r

OUT=r.O/'framewise-canonical-route/shared-three-frame'
RECORD=Path('/mnt/d/00/记录/005')
rows=r.read(r.R/'data/inputs.json')['records'];crops=r.read(r.O/'input-crops.json')
result=r.read(OUT/'result.json');fit=dict(np.load(OUT/'fit.npz'))
assert result['state']=='complete' and fit['betas'].shape==(1,10)
assert not (OUT/'rendered-evidence.json').exists()

# Reuse the already documented diagnostic triangle overlay, not an image warp.
source=Path(__file__).with_name('review_smplify_corrected123.py')
tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mesh')
scope=dict(np=np,cv2=cv2,Image=Image)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
mesh=scope['mesh']

def join(panels):
    assert len({x.size for x in panels})==1
    im=Image.new('RGB',(panels[0].width*len(panels),panels[0].height))
    for j,p in enumerate(panels):im.paste(p,(j*p.width,0))
    return im

def dots(image,xy,b,color,indices):
    draw=ImageDraw.Draw(image)
    for k in indices:
        x,y=xy[k]-b[:2]
        draw.ellipse([x-1,y-1,x+1,y+1],fill=color)

files=[]
for i,cr in enumerate(crops):
    slot=cr['slot'];frame=rows[slot]['source_frame_index'];im=Image.open(rows[slot]['image']).convert('RGB')
    current=dict(vertices_camera=fit['vertices_camera'][i:i+1],faces=fit['faces'],K_integer_centers=fit['K_integer_centers'][i])
    previous=dict(np.load(r.O/f'conditioned-reference/slot{slot:03d}-fit.npz'))
    body=np.array(cr['regions']['body']['crop_integer']);face=np.array(cr['regions']['face']['crop_integer'])
    for stage,b,kind in [('smpl-x',body,'原图_共享体型SMPLX叠加'),
                         ('特征点',body,'原图_2D全部检测点_3D共享体型投影'),
                         ('分析',face,'原图_此前逐帧独立拟合_共享体型逐帧拟合')]:
        raw=im.crop(tuple(b))
        if stage=='smpl-x':panels=[raw,mesh(raw,b,current)]
        elif stage=='特征点':
            det=raw.copy();model=raw.copy();ids=np.flatnonzero(fit['confidence'][i]>.25)
            dots(det,fit['targets'][i],b,(0,255,80),ids);dots(model,fit['projected'][i],b,(255,0,220),ids)
            panels=[raw,det,model]
        else:panels=[raw,mesh(raw,b,previous),mesh(raw,b,current)]
        path=RECORD/stage/f'20261029_共享体型逐帧_视频帧{frame:06d}_从左到右_{kind}.png'
        assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);sheet=join(panels);sheet.save(path)
        assert np.array_equal(np.asarray(sheet)[:,:raw.width],np.asarray(raw))
        files.append(dict(stage=stage,slot=slot,frame=frame,path=str(path),sha256=r.sha(path),
                          crop=b.tolist(),size=list(sheet.size),panels=kind.split('_'),native_pixels=True))
r.save(OUT/'rendered-evidence.json',dict(updated=r.now(),files=files,images=9,no_text_on_images=True,
       new_work_log_images=False,no_resampling=True,no_2D_warp=True,raw_panels_byte_identical=True,
       model_preview='Diagnostic CPU triangle overlay, not Gaussian rendering or native rasterizer acceptance.',
       source_renderer_sha256=r.sha(source),actual_visual_review_pending=True,
       detector_dots='All saved high-relative-confidence targets shown for diagnosis; this does not imply all were used in fitting.',
       new_gaussian=False))
print(json.dumps(dict(images=9,files=[x['path'] for x in files]),ensure_ascii=False))
