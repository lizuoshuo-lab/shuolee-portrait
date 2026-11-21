"""Four pre-existing evenly spaced orbit cameras; reuse original18k PNGs."""
import fcntl
from pathlib import Path
from types import SimpleNamespace
import sys,time
import numpy as np
from PIL import Image
sys.path[:0]=['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src','/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core']
from shuolee_portrait import runtime as rt
from run_official_gaussians123 import ASSETS


def main():
    root=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    old=root/'indoor-selected-sh0-pipeline124/delivery/step-018000'
    new=root/'indoor-zoom-clock18k124/delivery/step-018000'
    assert rt.read(new/'validation.json')['format_passed']
    assert rt.read(old/'validation.json')['reference_frame']==rt.read(new/'validation.json')['reference_frame']
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=new/'matched-orbit';out.mkdir(exist_ok=False)
    cams=rt.read(old/'novel/cameras.json')['records'];assert len(cams)==36
    indices=(0,9,18,27)
    sources=[Path(__file__),new/'point_cloud.ply',old/'novel/cameras.json']+[old/'novel'/f'novel_{i:03d}.png' for i in indices]
    hashes={str(p):rt.sha(p) for p in sources}
    rt.save(out/'plan.json',dict(updated=rt.now(),source_hashes=hashes,indices=indices,budget_seconds=60,optimizer_updates=0,
        purpose='Same exact four frozen evenly spaced old18k cameras; no selection by new render quality. Original PNGs reused without rerender.',ground_truth_available=False))
    rt.configure(ASSETS)
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    model=GaussianModel(0);model.load_ply(str(new/'point_cloud.ply'));model.active_sh_degree=0
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');pairs=[];started=time.monotonic()
    with torch.no_grad():
        for index in indices:
            cam,_=camera(cams[index],index)
            rgb=render(cam,model,pipe,bg)['render'];assert torch.isfinite(rgb).all()
            current=image(rgb);baseline=Image.open(old/'novel'/f'novel_{index:03d}.png').convert('RGB')
            assert current.size==baseline.size
            current.save(out/f'current-{index:02d}.png')
            pair=Image.new('RGB',(current.width*2,current.height),'black')
            pair.paste(baseline,(0,0));pair.paste(current,(current.width,0));pairs.append(pair)
            assert time.monotonic()-started<60
    sheet=Image.new('RGB',(pairs[0].width*2,pairs[0].height*2),'black')
    for j,p in enumerate(pairs):sheet.paste(p,((j%2)*p.width,(j//2)*p.height))
    sheet.save(out/'four-uniform-directions.png')
    record=Path('/mnt/d/00/记录/008/分析_室内放大时序124_18000_原整图与新放大_同四方向.png')
    assert not record.exists();sheet.save(record)
    assert all(rt.sha(p)==h for p,h in hashes.items())
    rt.save(out/'result.json',dict(updated=rt.now(),indices=indices,same_reference_pose=True,same_camera_for_both=True,
        original_images_reused=True,source_hashes=hashes,optimizer_updates=0,seconds=time.monotonic()-started,
        actual_visual_review=False,quality_accepted=False,ground_truth_available=False))
    print('INDOOR_CLOCK_FOUR_MATCHED_ORBITS_READY',flush=True)


if __name__=='__main__':main()
