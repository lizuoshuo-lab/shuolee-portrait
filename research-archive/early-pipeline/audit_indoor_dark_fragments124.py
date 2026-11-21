"""Separate transparent gaps from dark but opaque pixels in four fixed views."""
import fcntl
from pathlib import Path
import sys,time,traceback,signal
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src','/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS
from audit_indoor_directional_support124 import ROOT,B,D,OUT as SUPPORT,CANDIDATES,INDICES,contribution_summary
from audit_body_anisotropy123 import ply,fields,unit

OUT=ROOT/'indoor-dark-fragments124'


def main():
    assert not OUT.exists();OUT.mkdir()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=[Path(__file__),SUPPORT/'plan.json',SUPPORT/'result.json',SUPPORT/'point-metrics.npz',B/'data/training-inputs.json',B/'delivery/step-018000/novel/cameras.json']
    for label,root in CANDIDATES:
        paths += [root/'delivery/step-018000/point_cloud.ply',root/'gaussian-run/step-018000/world.ply',root/'gaussian-run/step-018000/current-meshes.npz',root/'gaussian-run/step-018000/binding.npz']
        paths += [SUPPORT/f'{label}-orbit-{i:02d}-rgb-support.png' for i in INDICES]
    hashes={str(p):rt.sha(p) for p in paths}
    rt.save(OUT/'plan.json',dict(updated=rt.now(),source_hashes=hashes,indices=INDICES,budget_seconds=90,optimizer_updates=0,
        question='Does dark fragmentation reflect absence of opacity or learned SH0 colors? Are thin Gaussian axes consistently aligned to their bound triangle?',
        fixed_diagnostic_definitions=dict(opaque_alpha_min=.9,dark_max_rgb=.05,gray_max_rgb=.2),
        controls=['Reuse frozen same-camera RGB; rerender replay must match bytes.',
                  'White override gives accumulated alpha. Complement black over white must sum to one.',
                  'Inspect raw SH0, distinguish actual black material from unexplained black clothing regions.',
                  'Quaternion rotations are unit-normalized; thin-axis angles use absolute dot (unoriented normal).'],
        limits=['No novel-view ground truth; dark opaque pixels alone are not errors, e.g. real hair/shorts.',
                'Background contribution is snapshot correlation, not proof of optimization history or exact mismatch cause.',
                'Very thin Gaussian axes can be legitimate surfaces. Bound-triangle normals are not true garment normals.',
                'No filling, deletion, color repair, threshold sweep or new model export. Unknown legs stay unknown.'],quality_accepted=False))
    rt.save(OUT/'progress.json',dict(state='running_readonly',pid=__import__('os').getpid(),updated=rt.now()))
    started=time.monotonic()
    def deadline(*_):raise TimeoutError('Read-only90s budget exceeded')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(90)
    try:
        rt.configure(ASSETS)
        import torch
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
        from shuolee_portrait.training import camera,image
        from utils.general_utils import build_rotation
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
        bg=torch.zeros(3,device='cuda');whitebg=torch.ones(3,device='cuda')
        rows=rt.read(B/'data/training-inputs.json')['records'];cams=rt.read(B/'delivery/step-018000/novel/cameras.json')['records']
        data=dict(np.load(SUPPORT/'point-metrics.npz',allow_pickle=False));reports=[];panels={}
        with torch.no_grad():
            for label,root in CANDIDATES:
                model=GaussianModel(0);model.load_ply(str(root/'delivery/step-018000/point_cloud.ply'));model.active_sh_degree=0
                raw=data[label+'_contribution_by_view_and_region'];fg=raw[:,:,0].sum(0);back=raw[:,:,1].sum(0);unknown=raw[:,:,2].sum(0)
                color=np.maximum(.28209479177387814*fields(ply(root/'gaussian-run/step-018000/world.ply'),['f_dc_0','f_dc_1','f_dc_2'])+.5,0)
                dark=color.max(1)<.05;gray=color.max(1)<.2
                info=dict(points=len(color),dark_points=int(dark.sum()),gray_points=int(gray.sum()),
                    foreground_contribution_from_dark=float(fg[dark].sum()/fg.sum()),
                    background_contribution_from_dark=float(back[dark].sum()/back.sum()),
                    dark_point_contribution_known_background_share=float(back[dark].sum()/max((back[dark]+fg[dark]).sum(),1e-30)),
                    dark_point_contribution_unknown_share=float(unknown[dark].sum()/max(raw[:,dark,:].sum(),1e-30)))
                mesh=dict(np.load(root/'gaussian-run/step-018000/current-meshes.npz',allow_pickle=False))
                bind=np.load(root/'gaussian-run/step-018000/binding.npz',allow_pickle=False)['triangle_ids']
                E=np.asarray(rows[0]['w2c']);v=(mesh['vertices_camera'][0]-E[:3,3])@E[:3,:3]
                tri=v[mesh['faces']][bind];n=unit(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))*np.array([1,-1,-1])
                rotations=build_rotation(model.get_rotation).cpu().numpy();scales=model.get_scaling.cpu().numpy()
                thin=np.argmin(scales,axis=1);axis=rotations[np.arange(len(color)),:,thin]
                assert np.max(np.abs(np.linalg.norm(axis,axis=1)-1))<2e-6
                angles=np.degrees(np.arccos(np.clip(np.abs((axis*n).sum(1)),0,1)))
                positive=fg>0;info['thin_axis_vs_bound_normal_deg']=contribution_summary(angles[positive],fg[positive])
                per=[];panels[label]=[]
                for i in INDICES:
                    cam,_=camera(cams[i],i)
                    rgb=render(cam,model,pipe,bg)['render']
                    alpha=render(cam,model,pipe,bg,override_color=torch.ones((len(color),3),device='cuda'))['render']
                    inverse=render(cam,model,pipe,whitebg,override_color=torch.zeros((len(color),3),device='cuda'))['render']
                    err=float((alpha+inverse-1).abs().max());assert err<2e-5
                    saved=Image.open(SUPPORT/f'{label}-orbit-{i:02d}-rgb-support.png').convert('RGB')
                    source=saved.crop((0,0,saved.width//2,saved.height));actual=image(rgb)
                    byte_error=int(np.abs(np.asarray(source,dtype=np.int16)-np.asarray(actual,dtype=np.int16)).max());assert byte_error==0
                    a=alpha[0].cpu().numpy();c=rgb.cpu().numpy().transpose(1,2,0)
                    opaque=a>=.9;black=opaque & (c.max(2)<.05);graypix=opaque & (c.max(2)<.2)
                    per.append(dict(orbit_index=i,opaque_pixels=int(opaque.sum()),opaque_dark_pixels=int(black.sum()),
                        opaque_dark_fraction=float(black.sum()/max(opaque.sum(),1)),opaque_gray_fraction=float(graypix.sum()/max(opaque.sum(),1)),
                        alpha_complement_max_error=err,rgb_replay_max_byte=byte_error))
                    highlight=np.asarray(actual).copy();highlight[black]=[255,0,255]
                    panel=Image.new('RGB',(actual.width*3,actual.height))
                    panel.paste(actual,(0,0));panel.paste(image(alpha),(actual.width,0));panel.paste(Image.fromarray(highlight),(actual.width*2,0))
                    panel.save(OUT/f'{label}-orbit-{i:02d}-rgb-alpha-dark.png');panels[label].append(panel)
                reports.append(dict(name=label,point_colors=info,views=per))
                del model
        sheetrows=[]
        for j in range(len(INDICES)):
            pics=[panels[n][j] for n,_ in CANDIDATES]
            row=Image.new('RGB',(pics[0].width*2,pics[0].height))
            for k,p in enumerate(pics):row.paste(p,(k*p.width,0))
            scale=min(1,2100/row.width);sheetrows.append(row.resize((round(row.width*scale),round(row.height*scale)),Image.Resampling.LANCZOS))
        page=Image.new('RGB',(sheetrows[0].width,sum(x.height for x in sheetrows)+44),(22,22,22))
        ImageDraw.Draw(page).text((10,8),'Old: RGB / alpha / dark opaque (magenta) | Zoom+clock: RGB / alpha / dark opaque (magenta). Black clothing may be real.',fill='white')
        y=44
        for row in sheetrows:page.paste(row,(0,y));y+=row.height
        page.save(OUT/'four-orbits-rgb-alpha-dark.png')
        target=Path('/mnt/d/00/记录/008/分析_室内放大124_四方向_RGB_透明度_不透明黑色.png');assert not target.exists();page.save(target)
        assert all(rt.sha(p)==h for p,h in hashes.items())
        rt.save(OUT/'result.json',dict(updated=rt.now(),state='readonly_completed',reports=reports,optimizer_updates=0,
            source_hashes_unchanged=True,elapsed_seconds=time.monotonic()-started,actual_visual_review=False,quality_accepted=False))
        rt.save(OUT/'progress.json',dict(state='completed',pid=None,updated=rt.now()))
        print('DARK_FRAGMENT_AUDIT_COMPLETE',reports,flush=True)
    except BaseException as e:
        rt.save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        rt.save(OUT/'progress.json',dict(state='failed_preserved',pid=None,updated=rt.now()));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
