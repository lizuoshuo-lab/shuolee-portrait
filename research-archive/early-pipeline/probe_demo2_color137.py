"""One actual HLG-to-sRGB reference preview; no training color choice."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from intake_demo2_candidate137 import SOURCE, OUT
from official_smplifyx_entry123 import sha
from update_leafwoman129_state import atomic, read


def run():
    native=read(OUT/'dense-review/native-source-contract.json');digest=sha(SOURCE)
    assert native['source_sha256']==digest and native['stream']['color_transfer']=='arib-std-b67'
    w,h=native['upright_wh'];assert (w,h)==(1080,1920)
    # Fixed reference display assumption, not a measured scene exposure.
    color=('zscale=matrixin=2020_ncl:transferin=arib-std-b67:primariesin=2020:rangein=limited:transfer=linear:npl=100,'
        'format=gbrpf32le,zscale=primaries=709,tonemap=tonemap=mobius:param=0.3:desat=2:peak=10,'
        'zscale=transfer=iec61966-2-1:primaries=709:matrix=gbr:range=full,format=rgb24')
    ffmpeg=shutil.which('ffmpeg');base=[ffmpeg,'-nostdin','-v','error','-hwaccel','none','-threads','2',
        '-i',str(SOURCE),'-filter_threads','1']
    tail=['-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    data=[subprocess.check_output(base+(['-vf',color] if k else [])+tail,timeout=20) for k in range(2)]
    assert len(data[0])==len(data[1])==w*h*3 and data[0]!=data[1]
    import hashlib
    assert hashlib.sha256(data[0]).hexdigest()==native['frame0_RGB_sha256']
    a,b=[np.frombuffer(x,np.uint8).reshape(h,w,3) for x in data]
    O=OUT/'color-reference';O.mkdir(exist_ok=False)
    page=Image.new('RGB',(720,664),(19,23,28));d=ImageDraw.Draw(page)
    for j,(label,img) in enumerate(zip(('Direct RGB preview (not color-managed)','HLG to sRGB reference preview'),(a,b))):
        x=j*360;d.text((x+5,7),label,fill='white');page.paste(Image.fromarray(img).resize((360,640),Image.Resampling.LANCZOS),(x,24))
    path=O/'frame0-hlg-reference.jpg';page.save(path,quality=94)
    assert sha(SOURCE)==digest
    atomic(O/'result.json',dict(updated=datetime.now(timezone.utc).isoformat(),source_sha256=digest,source_frame_index=0,
        upright_wh=[w,h],first_frame_naive_decode_verified=True,filter=color,assumed_HLG_peak_nits=1000,
        reference_white_nits=100,peak_assumption_not_calibrated=True,
        output_RGB_extrema=[int(b.min()),int(b.max())],output_zero_fraction=float((b==0).mean()),
        output_255_fraction=float((b==255).mean()),visual_review_pending=True,reference_image_sha256=sha(path),
        no_native_raster_resize=True,training_color_admitted=False,training_started=False))
    print('DEMO2_HLG_REFERENCE_PREVIEW',w,h,sha(path))


def upstream_reference():
    """One documented alternate after the first reference visibly clipped."""
    repo=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/EasyVolcap')
    origin=repo/'scripts/preprocess/extract_videos.py'
    color='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv'
    assert color in origin.read_text(encoding='utf-8')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    n=read(OUT/'dense-review/native-source-contract.json');w,h=n['upright_wh'];digest=sha(SOURCE)
    assert digest==n['source_sha256']
    cmd=[shutil.which('ffmpeg'),'-nostdin','-v','error','-hwaccel','none','-threads','2','-i',str(SOURCE),
        '-filter_threads','1','-vf',color+',format=rgb24','-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    data=subprocess.check_output(cmd,timeout=20);assert len(data)==w*h*3
    a=np.frombuffer(data,np.uint8).reshape(h,w,3);old=OUT/'color-reference/frame0-hlg-reference.jpg'
    with Image.open(old) as im:panels=im.convert('RGB')
    page=Image.new('RGB',(1080,664),(19,23,28));page.paste(panels,(0,0));d=ImageDraw.Draw(page)
    d.text((725,7),'EasyVolcap Hable (BT.709 preview)',fill='white')
    page.paste(Image.fromarray(a).resize((360,640),Image.Resampling.LANCZOS),(720,24))
    dest=OUT/'color-reference/frame0-hlg-controls.jpg';page.save(dest,quality=94)
    assert sha(SOURCE)==digest
    atomic(dest.with_suffix('.json'),dict(updated=datetime.now(timezone.utc).isoformat(),source_sha256=digest,
        source_index=0,source_recipe=str(origin),source_recipe_sha256=sha(origin),source_repo_commit=commit,filter=color,
        output_transfer='bt709',upright_wh=[w,h],reference_image_sha256=sha(dest),
        output_255_fraction=float((a==255).mean()),output_zero_fraction=float((a==0).mean()),
        original_reference_255_fraction=read(OUT/'color-reference/result.json')['output_255_fraction'],
        no_training_color_admission=True,geometry_unchanged=True,training_started=False))
    print('DEMO2_EASYVOLCAP_REFERENCE',float((a==255).mean()),sha(dest))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--upstream',action='store_true')
    upstream_reference() if p.parse_args().upstream else run()
