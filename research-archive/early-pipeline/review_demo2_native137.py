"""Actual 3-fps source contact review inputs for demo2; no fitting cache."""
from datetime import datetime, timezone
import json
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from intake_demo2_candidate137 import SOURCE, OUT
from official_smplifyx_entry123 import sha
from update_leafwoman129_state import atomic, read


def run():
    started=time.monotonic()
    prior=read(OUT/'intake.json');digest=sha(SOURCE)
    assert digest==prior['source_sha256'] and prior['display_rotation_degrees']==-90
    ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe')
    q=[ffprobe,'-v','error','-select_streams','v:0','-show_frames','-show_streams',
       '-show_entries','frame=best_effort_timestamp_time:stream=width,height,color_range,color_space,color_transfer,color_primaries:stream_side_data=rotation','-of','json',str(SOURCE)]
    m=json.loads(subprocess.check_output(q,timeout=60));stream,=m['streams'];W,H=1920,1080
    assert (stream['width'],stream['height'])==(W,H) and stream['side_data_list'][0]['rotation']==-90
    assert stream['color_transfer']=='arib-std-b67' and stream['color_space']=='bt2020nc'
    pts=np.asarray([float(f['best_effort_timestamp_time']) for f in m['frames']]);assert len(pts)==869
    targets=pts[0]+np.arange(int(np.floor((pts[-1]-pts[0])*3))+1)/3
    ids=[int(np.argmin(abs(pts-t))) for t in targets];assert len(ids)==len(set(ids))==87
    assert all(pts[r['source_frame_index']]==r['pts_seconds'] for r in prior['samples'])
    common=[ffmpeg,'-nostdin','-v','error','-hwaccel','none','-threads','2']
    outfmt=['-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    auto=subprocess.check_output(common+['-i',str(SOURCE)]+outfmt,timeout=20)
    explicit=subprocess.check_output(common+['-noautorotate','-i',str(SOURCE),'-vf','transpose=clock']+outfmt,timeout=20)
    assert len(auto)==W*H*3 and auto==explicit
    O=OUT/'dense-review';O.mkdir(exist_ok=False)
    rows=[dict(frame=f'frame_{i+1:06d}',source_frame_index=j,time_seconds=float(pts[j]),
               target_seconds=float(targets[i])) for i,j in enumerate(ids)]
    A=[[0,-1,H-1],[1,0,0],[0,0,1]]
    atomic(O/'native-source-contract.json',dict(updated=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest,stream=stream,encoded_wh=[W,H],upright_wh=[H,W],
        rotation_integer_centers_encoded_to_upright=A,actual_frame0_explicit_equals_display=True,
        frame0_RGB_sha256=__import__('hashlib').sha256(auto).hexdigest(),frame_identity_real_pts=True,rows=rows,
        exact_K_available=False,existing_versa_K_reused=False,HLG_display_preview_not_color_managed=True,
        new_training_color_pipeline_required=True,raw_native_cache_written=False,back_or_fitting_admitted=False))
    expression='+'.join(r'eq(n\,%d)'%i for i in ids)
    cmd=common+['-i',str(SOURCE),'-filter_threads','1','-vf','select='+expression,
        '-fps_mode','passthrough','-frames:v',str(len(ids)),'-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    tw,th=162,288;cols=6;pages=[]
    with subprocess.Popen(cmd,stdout=subprocess.PIPE) as child:
        try:
            for i,row in enumerate(rows):
                data=child.stdout.read(W*H*3);assert len(data)==W*H*3
                tile=i%30
                if not tile:
                    page=Image.new('RGB',(tw*cols,(th+22)*5),(19,23,28));draw=ImageDraw.Draw(page)
                im=Image.frombytes('RGB',(H,W),data);im.thumbnail((tw,th),Image.Resampling.LANCZOS)
                x,y=(tile%cols)*tw,(tile//cols)*(th+22)
                page.paste(im,(x,y+22));draw.text((x+4,y+4),f'{i+1:02d}  {row["time_seconds"]:.2f}s',fill='white')
                if tile==29 or i==len(rows)-1:
                    path=O/f'pts-review-{len(pages)+1:02d}.jpg';page.save(path,quality=92,subsampling=0)
                    pages.append(dict(name=path.name,sha256=sha(path)))
                assert time.monotonic()-started<130
            assert child.wait(timeout=10)==0
        finally:
            if child.poll() is None:child.kill();child.wait()
    assert sha(SOURCE)==digest
    atomic(O/'result.json',dict(actual_decode_completed=True,actual_review_pending=True,source_sha256=digest,
        pages=pages,frames=len(rows),seconds=time.monotonic()-started,preprocessing_or_fitting_started=False,
        raw_native_cache_written=False,quality_accepted=False))
    print('DEMO2_3FPS_NATIVE_SOURCE_REVIEW',len(rows),len(pages),round(time.monotonic()-started,2))


if __name__=='__main__':run()
