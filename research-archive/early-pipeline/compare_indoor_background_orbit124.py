"""Same four frozen novel cameras: black-background1000 vs paired-background1000."""
from pathlib import Path
import fcntl,sys,time
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src','/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS
from audit_indoor_directional_support124 import B,D,ROOT,INDICES


def main():
    new=ROOT/'indoor-paired-background124';out=new/'matched-orbit';out.mkdir(exist_ok=False)
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=[('black1000',D/'gaussian-run/step-001000/world.ply'),('paired1000',new/'gaussian-run/step-001000/world.ply')]
    cams=rt.read(B/'delivery/step-018000/novel/cameras.json')['records']
    paths=[Path(__file__),B/'delivery/step-018000/novel/cameras.json']+[p for _,p in cases]
    hashes={str(p):rt.sha(p) for p in paths}
    rt.save(out/'plan.json',dict(updated=rt.now(),source_hashes=hashes,indices=INDICES,budget_seconds=90,
        source_step=1000,candidate_step=1000,reference_frame='frame_000105',source_coordinates='training world; invert prior export rigid basis in cameras',
        camera_transform='E_world=E_portable@diag(1,-1,-1,1)',renderer='Original nativeCUDA,SH0,unaltered Gaussian parameters',
        controls='RGB on black and middle-gray; alpha via white RGB on black. No novel-viewtruth or hidden-leg supervision.',optimizer_updates=0))
    rt.configure(ASSETS)
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');gray=torch.ones(3,device='cuda')*.35
    rows=[];stats=[];started=time.monotonic()
    models=[]
    for label,p in cases:
        model=GaussianModel(0);model.load_ply(str(p));model.active_sh_degree=0;models.append(model)
    with torch.no_grad():
        for idx in INDICES:
            row=dict(cams[idx]);row['w2c']=(np.asarray(row['w2c'])@np.diag([1,-1,-1,1])).tolist()
            cam,_=camera(row,idx);blackpics=[];graypics=[];alphapics=[]
            for (label,_),model in zip(cases,models):
                rgb=render(cam,model,pipe,bg)['render'];alpha=render(cam,model,pipe,bg,override_color=torch.ones((len(model._xyz),3),device='cuda'))['render']
                neutral=render(cam,model,pipe,gray)['render']
                error=float((neutral-(rgb+(1-alpha)*.35)).abs().max());assert error<3e-6
                c=rgb.cpu().numpy().transpose(1,2,0);aa=alpha[0].cpu().numpy();opaque=aa>=.9;dark=opaque&(c.max(2)<.05)
                stats.append(dict(index=idx,name=label,opaque_pixels=int(opaque.sum()),opaque_dark_pixels=int(dark.sum()),opaque_dark_fraction=float(dark.sum()/max(opaque.sum(),1)),gray_compositing_error=error))
                blackpics.append(image(rgb));graypics.append(image(neutral));alphapics.append(image(alpha))
            pieces=blackpics+graypics+alphapics
            panel=Image.new('RGB',(pieces[0].width*len(pieces),pieces[0].height))
            for j,p in enumerate(pieces):panel.paste(p,(j*p.width,0))
            panel.save(out/f'direction-{idx:02d}.png')
            scale=min(1,2100/panel.width);rows.append(panel.resize((round(panel.width*scale),round(panel.height*scale)),Image.Resampling.LANCZOS))
            assert time.monotonic()-started<90
    sheet=Image.new('RGB',(rows[0].width,sum(p.height for p in rows)+44),(22,22,22));ImageDraw.Draw(sheet).text((10,8),'Each pair: fixed black BG1000 / paired random BG1000. Black display | gray display | alpha. Same four cameras, same1000 steps.',fill='white')
    y=44
    for p in rows:sheet.paste(p,(0,y));y+=p.height
    sheet.save(out/'four-matched-directions.png')
    target=Path('/mnt/d/00/记录/008/分析_室内配对背景124_1000_同四方向_黑底_灰底_alpha.png');assert not target.exists();sheet.save(target)
    assert all(rt.sha(p)==h for p,h in hashes.items())
    rt.save(out/'result.json',dict(updated=rt.now(),same_cameras=True,reference_pose_same=True,source_hashes_unchanged=True,
        stats=stats,seconds=time.monotonic()-started,actual_visual_review=False,quality_accepted=False,optimizer_updates=0))
    print('PAIRED_BACKGROUND_MATCHED_ORBITS_COMPLETE',stats,flush=True)


if __name__=='__main__':main()
