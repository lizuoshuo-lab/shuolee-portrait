"""Same-camera, unwarped LR comparison. This script never trains."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from PIL import Image
from plyfile import PlyData
from review_full_sequence123 import read,sha,relocate
from review_shared_vertex_offset123 import join,clip_box,stats
import run_official_gaussians123 as io

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle'
ROOT=O/'shared-vertex-offset-lr1e-5-1000';OUT=ROOT/'comparison'
OLD=O/'shared-vertex-offset-1000/gaussian-run/step-001000'
SNAP=ROOT/'gaussian-run/step-001000'
RECORD=Path('D:/00/记录/005')
BASELINE_NAME='lr1e-6';CANDIDATE_NAME='lr1e-5'
RECORD_PREFIX='20261030_共享顶点偏移学习率对照1000步_左原图_中1e-6_右1e-5'
NOVEL_RECORD='20261030_共享顶点偏移学习率1e-5_1000步_连续未知视角000至035.png'

def compare():
    assert read(SNAP/'native-ply-parity.json')['passed']
    OUT.mkdir(exist_ok=False)
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as f:fit=dict(f)
    assert sha(ROOT/'fit/fitted.npz')==sha(OLD.parent.parent/'fit/fitted.npz')
    rows=read(ROOT/'data/training-inputs.json')['records'];results=[]
    for i in (0,18,37,50,55,74,93):
        row=rows[i];fid=row['frame']
        ims=[Image.open(relocate(row['image'])).convert('RGB'),
             Image.open(OLD/f'{fid}_render.png').convert('RGB'),Image.open(SNAP/f'{fid}_render.png').convert('RGB')]
        assert all(im.size==(2704,1520) for im in ims)
        mask=np.asarray(Image.open(relocate(row['mask'])).convert('L'))>252
        y,x=np.nonzero(mask)
        boxes={'body':clip_box(np.c_[x,y],30,2704,1520),
               'face':clip_box(fit['observed_native_keypoints'][i,67:118,:2],40,2704,1520),
               'feet':clip_box(fit['observed_native_keypoints'][i,19:25,:2],45,2704,1520)}
        arrays=[np.asarray(im,dtype=float)/255 for im in ims];regions={}
        for label,box in boxes.items():
            p=OUT/f'{fid}_{label}.png';join([im.crop(box) for im in ims]).save(p)
            x0,y0,x1,y1=box;valid=mask[y0:y1,x0:x1]
            error=lambda a:float(abs(a[y0:y1,x0:x1]-arrays[0][y0:y1,x0:x1])[valid].mean())
            regions[label]=dict(crop=list(box),baseline_l1=error(arrays[1]),candidate_l1=error(arrays[2]),image=p.name)
            folder=RECORD/('3DGS' if label=='body' else '分析');folder.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,folder/f'{RECORD_PREFIX}_{fid}_{label}.png')
        results.append(dict(frame=fid,regions=regions))
    for label,size in [('body',(440,450)),('face',(440,180)),('feet',(440,180))]:
        panel=Image.new('RGB',(size[0]*2,size[1]*4),(242,242,242))
        for j,p in enumerate(sorted(OUT.glob(f'frame*_{label}.png'))):
            im=Image.open(p);im.thumbnail(size,Image.Resampling.LANCZOS)
            panel.paste(im,((j%2)*size[0],(j//2)*size[1]))
        panel.save(OUT/f'{label}7.png')
    asset=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
    with np.load(asset,allow_pickle=True) as a:
        beta=fit['betas'].reshape(-1);base=a['v_template']+np.einsum('vck,k->vc',a['shapedirs'][...,:len(beta)],beta)
    f=fit['faces'];tri=base[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);geometry=[]
    for name,snapshot in [(BASELINE_NAME,OLD),(CANDIDATE_NAME,SNAP)]:
        with np.load(snapshot/'shared-static-offset.npz',allow_pickle=False) as a:offset=a['static_offset']
        tri=(base+offset)[f];newnormal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        reversals=(normal*newnormal).sum(1)<0
        v=PlyData.read(snapshot/'world.ply')['vertex'].data
        scales=np.exp(np.column_stack([v[f'scale_{j}'] for j in range(3)]))
        geometry.append(dict(name=name,offset_m=stats(np.linalg.norm(offset,axis=1)),
            canonical_shaped_triangle_normal_reversals=int(reversals.sum()),triangles=len(f),
            gaussian_std_max_m=float(scales.max()),gaussians_std_over_3cm=int((scales.max(1)>.03).sum())))
    result=dict(updated=io.now(),steps=1000,comparisons=results,geometry=geometry,
        source_sha256=sha(SNAP/'world.ply'),actual_visual_review=False,quality_accepted=False,
        metric_scope='Saved8bit foreground L1 is not geometric truth; canonical normal reversal is not a complete self-intersection test.',
        fit_camera_inputs_exactly_equal=True)
    io.save(OUT/'comparison.json',result)
    print(json.dumps(dict(geometry=geometry,reference=results[3])))

def novel():
    paths=sorted((ROOT/'export/review').glob('novel_*.png'));assert len(paths)==36
    sheet=Image.new('RGB',(960,1440),(242,242,242))
    for i,p in enumerate(paths):
        im=Image.open(p).convert('RGB');im.thumbnail((160,240),Image.Resampling.LANCZOS)
        sheet.paste(im,((i%6)*160,(i//6)*240))
    sheet.save(OUT/'novel36.png')
    shutil.copy2(OUT/'novel36.png',RECORD/'分析'/NOVEL_RECORD)
    print('NOVEL_CONTACT_READY')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--novel',action='store_true');a=p.parse_args()
    novel() if a.novel else compare()
