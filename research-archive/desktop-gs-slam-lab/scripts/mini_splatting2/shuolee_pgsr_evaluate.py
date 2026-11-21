"""Frozen, native-renderer method comparison. Run each renderer in a fresh process."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[2]
ROOT = LAB/'outputs/mini-splatting2-mip-depth-001'
HOME = ROOT/'shuolee'
RUN = HOME/'current/pgsr-surface'
OUT = RUN/'evaluation'
BASE = ROOT/'native-resolution-cache-full-18000-s0/model'
SOURCE = ROOT/'source-resolution-rectified-002/dataset'
NATIVE = Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')
PGSR = Path('/opt/shuolee/pgsr-runtime')


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def prepare():
    OUT.mkdir(exist_ok=False)
    base={c['img_name']:c for c in read(BASE/'cameras.json')}
    candidate={c['img_name']:c for c in read(RUN/'train/model/cameras.json')}
    assert set(base)==set(candidate) and len(base)==59
    differences={key:max(float(np.max(np.abs(np.asarray(base[n][key])-np.asarray(candidate[n][key]))))
        for n in base) for key in ['width','height','position','rotation','fx','fy']}
    assert max(differences.values())<1e-8,differences
    assignment=next(n for n in ast.parse(Path(__file__).with_name('local_refinement.py').read_text(encoding='utf-8')).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in n.targets))
    boxes=ast.literal_eval(assignment.value)
    dev=[base[n] for n in sorted(base) if int(n)%8==0]
    assert len(dev)==8 and set(int(c['img_name']) for c in dev)==set(boxes)
    regions={}
    for c in dev:
        n=c['img_name'];w,h=c['width'],c['height']
        x0,y0,x1,y1=[round(v*s) for v,s in zip(boxes[int(n)],[w,h,w,h])]
        regions[n]={'person':[x0,y0,x1,y1],'head':[x0,y0,x1,y0+round((y1-y0)/3)]}
    # Only metadata and existing SfM are consulted before freezing evaluation.
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'))
    import read_write_model as rw
    points=rw.read_points3D_binary(str(SOURCE/'sparse/0/points3D.bin'))
    images=rw.read_images_binary(str(SOURCE/'sparse/0/images.bin'))
    byname={Path(i.name).stem:i for i in images.values()}
    anchors=['000001','000007','000014','000021','000028','000035','000042','000049','000057']
    oldhead={r['frame']:np.asarray(r['box'])*2 for r in read(ROOT/'training-head-regions-003/manifest.json')['rows']}
    observations=[]
    for n in anchors:
        c=base[n];im=byname[n];R=np.array(c['rotation']);C=np.array(c['position'])
        motion=np.array(Image.open(ROOT/f'motion-masks-001/{n}.png').convert('L'))
        motion=cv2.resize(motion,(c['width'],c['height']),interpolation=cv2.INTER_NEAREST)
        used=set()
        for k,point_id in enumerate(im.point3D_ids):
            point=points.get(int(point_id))
            if point is None or point.id%5!=0 or point.error>1 or len(set(point.image_ids))<4:continue
            # Duplicated feature observations may carry several point IDs.
            feature=tuple(np.round(im.xys[k],4))
            if feature in used:continue
            used.add(feature)
            xyz=(point.xyz-C)@R;z=float(xyz[2])
            if z<=0:continue
            xy=np.array([c['fx']*xyz[0]/z+c['width']/2,c['fy']*xyz[1]/z+c['height']/2])
            x,y=np.round(xy).astype(int)
            if not (0<=x<c['width'] and 0<=y<c['height']) or motion[y,x]>0:continue
            b=oldhead[n];ishead=bool(b[0]<=xy[0]<b[2] and b[1]<=xy[1]<b[3])
            observations.append(dict(frame=n,point_id=int(point.id),xy=xy.tolist(),z=z,head=ishead))
    assert sum(o['head'] for o in observations)>50
    protocol=dict(owner='shuolee',created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        status='frozen_before_candidate_render',candidate_results_seen=False,
        baseline_ply_sha256=sha(BASE/'point_cloud/iteration_18000/point_cloud.ply'),
        baseline_filter_sha256=sha(BASE/'point_cloud/iteration_18000/filter_3d.pt'),
        source_calibration_sha256={n:sha(SOURCE/'sparse/0'/n) for n in ['images.bin','cameras.bin','points3D.bin']},
        images_sha256={c['img_name']:sha(SOURCE/'images'/(c['img_name']+'.png')) for c in dev},
        camera_max_difference=differences,development_cameras=dev,regions=regions,
        sparse_cameras=[base[n] for n in anchors],sparse_observations=observations,
        sparse_scope='Training/init SfM consistency, not independent geometry accuracy; point_id%5 subset; shared feature coordinates deduplicated.',
        paths=read(HOME/'camera-paths.json'),paths_sha256=sha(HOME/'camera-paths.json'),
        appearance='8 repeatedly used development frames at1514x2694; per-region static masks; head is top third of person box including shoulders/background.',
        lpips='AlexNet LPIPS on head/person rectangles at native pixels; report motion coverage; not a face-specific perceptual metric.',
        depth='PGSR native plane intersection depth versus MiniGS2 alpha-normalized expected Gaussian-center z; alpha>0.95 and coverage reported.',
        novel='Same120cameras:60distance+60elevation,576x1024. No real GT. Two videos plus fixedkeyframes; human review separate.',
        method_comparison=True,single_module_ablation=False,independent_test=False,
        promotion='No automatic promotion. Examine subject identity, surface holes/floaters, scale and temporal stability, and background regressions.')
    dump(OUT/'protocol.json',protocol)
    print(json.dumps(dict(status='protocol_frozen',camera_max_difference=differences,
        sparse_observations=len(observations),head_observations=sum(o['head'] for o in observations)),indent=2))


class Video:
    def __init__(self,path,w,h):
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        assert ffmpeg,'ffmpeg must exist; do not silently save a browser-incompatible video'
        self.process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{w}x{h}','-r','15','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def add(self,im):self.process.stdin.write(np.ascontiguousarray(im,dtype=np.uint8).tobytes())
    def close(self):
        self.process.stdin.close()
        assert self.process.wait()==0


def render_arm(arm):
    assert sys.platform=='linux','Native CUDA renderers run only in the isolated Linux runtime'
    train=read(RUN/'train/manifest.json')
    assert train['status']=='complete','Wait for full training; do not compete for GPU or render rolling checkpoints'
    protocol=read(OUT/'protocol.json')
    folder=OUT/arm;folder.mkdir(exist_ok=False)
    repo=PGSR/'repo' if arm=='pgsr' else NATIVE/'repo-depth-cache'
    deps=PGSR/'deps' if arm=='pgsr' else NATIVE/'deps'
    sys.path[:0]=[str(deps),str(repo),'/opt/gs-slam-lab/mini-splatting2-native-001/deps']
    import torch
    from scene import GaussianModel
    from scene.cameras import Camera
    if arm=='pgsr':
        from gaussian_renderer import render
        ply=RUN/'train/model/point_cloud/iteration_30000/point_cloud.ply'
    else:
        from gaussian_renderer import render_imp
        from native_depth_guidance import expected_depth
        ply=BASE/'point_cloud/iteration_18000/point_cloud.ply'
        assert sha(ply)==protocol['baseline_ply_sha256']
    digest=sha(ply);torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.cuda.set_per_process_memory_fraction(.82)
    pc=GaussianModel(3);pc.load_ply(str(ply))
    if arm=='baseline':
        pc._mip_iteration=18000;pc._mip_cameras=[]
        pc._mip_filter=torch.load(ply.with_name('filter_3d.pt'),map_location='cuda',weights_only=True)
        pc._mip_cache_key=(id(pc._xyz),len(pc.get_xyz),180);pc._mip_refreshes=0
        culled=torch.zeros(len(pc.get_xyz),dtype=torch.bool,device='cuda')
    bg=torch.zeros(3,device='cuda')
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)

    def camera(c):
        R=np.asarray(c['rotation']);C=np.asarray(c['position']);T=-R.T@C
        name=c.get('img_name',c.get('name','path'));w,h=c['width'],c['height']
        fx=2*math.atan(w/(2*c['fx']));fy=2*math.atan(h/(2*c['fy']))
        if arm=='pgsr':
            result=Camera(c.get('id',0),R,T,fx,fy,w,h,str(SOURCE/'images'/(name+'.png')),name,c.get('id',0),
                preload_img=False,data_device='cpu')
        else:
            result=Camera(c.get('id',0),R,T,fx,fy,torch.zeros(3,h,w),None,name,c.get('id',0),data_device='cpu')
        actual=result.world_view_transform.T.cpu().numpy()
        expected=np.eye(4);expected[:3,:3]=R.T;expected[:3,3]=T
        assert np.allclose(actual,expected,atol=1e-6)
        return result

    @torch.no_grad()
    def frame(c,depth=False):
        cam=camera(c)
        if arm=='pgsr':
            package=render(cam,pc,pipe,bg,return_plane=depth,return_depth_normal=False)
            rgb=package['render']
            if depth:
                d=package['plane_depth'].squeeze()
                alpha=render(cam,pc,pipe,bg,override_color=torch.ones_like(pc.get_xyz),
                    return_plane=False,return_depth_normal=False)['render'][0]
        else:
            rgb=render_imp(cam,pc,pipe,bg,culling=culled)['render']
            if depth:d,alpha=expected_depth(cam,pc,culled)
        assert torch.isfinite(rgb).all()
        encoded=rgb.clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
        if not depth:return encoded,None,None
        assert torch.isfinite(d).all() and torch.isfinite(alpha).all()
        return encoded,d.squeeze().cpu().numpy(),alpha.squeeze().cpu().numpy()

    start=time.time();observed=[];depth_rows=[]
    for c in protocol['development_cameras']:
        im,_,_=frame(c);name=c['img_name']
        Image.fromarray(im).save(folder/(name+'.png'))
        observed.append(dict(frame=name,sha256=sha(folder/(name+'.png'))))
        print(arm,'development',name,flush=True)
    for c in protocol['sparse_cameras']:
        name=c['img_name'];_,depth,alpha=frame(c,True)
        obs=[o for o in protocol['sparse_observations'] if o['frame']==name]
        xy=np.asarray([o['xy'] for o in obs],dtype=np.float32)
        ds=cv2.remap(depth,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).ravel()
        al=cv2.remap(alpha,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR).ravel()
        for o,z,a in zip(obs,ds,al):
            depth_rows.append(dict(**o,rendered_depth=float(z),alpha=float(a),
                valid=bool(z>0 and a>.95),relative_error=float(abs(z/o['z']-1))))
        print(arm,'depth',name,flush=True)
    path_rows=[]
    for kind in ['distance','elevation']:
        cameras=[c for c in protocol['paths'] if c['kind']==kind]
        assert len(cameras)==60
        video=Video(folder/(kind+'.mp4'),cameras[0]['width'],cameras[0]['height'])
        try:
            for c in cameras:
                key=c['index'] in [0,15,30,45]
                im,d,a=frame(c,key);video.add(im)
                row=dict(name=c['name'],index=c['index'],kind=kind)
                if key:
                    Image.fromarray(im).save(folder/(c['name']+'.png'))
                    # Shared fixed display units; no per-image contrast remapping.
                    color=cv2.applyColorMap(np.uint8(np.clip((d-1)/9,0,1)*255),cv2.COLORMAP_TURBO)[...,::-1]
                    color[(a<=.95)|(d<=0)]=0
                    Image.fromarray(color).save(folder/(c['name']+'-depth.png'))
                    row.update(alpha_above_095_fraction=float((a>.95).mean()),depth_display_range=[1,10])
                path_rows.append(row)
        finally:video.close()
        print(arm,kind,'60 frames complete',flush=True)
    assert sha(ply)==digest
    for name,digest_input in protocol['source_calibration_sha256'].items():
        assert sha(SOURCE/'sparse/0'/name)==digest_input
    dump(folder/'manifest.json',dict(status='complete',arm=arm,model=str(ply),model_sha256=digest,
        gaussian_count=len(pc.get_xyz),elapsed_seconds=time.time()-start,unchanged_model=True,
        camera_validation=True,observed=observed,path_rows=path_rows,
        depth_semantics='plane intersection' if arm=='pgsr' else 'alpha normalized expected Gaussian center z',
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    dump(folder/'depth-observations.json',depth_rows)
    print(json.dumps(dict(status='complete',arm=arm,elapsed_seconds=time.time()-start)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','baseline','pgsr']);args=p.parse_args()
    if args.mode=='prepare':prepare()
    else:render_arm(args.mode)


if __name__=='__main__':main()

