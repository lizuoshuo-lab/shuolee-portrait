"""Read-only 12-point PTS intake; no raw frame cache or fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from update_leafwoman129_state import ROOT, atomic

SOURCE=Path('D:/00/工作/数据集/demo2.mp4')
OUT=ROOT/'demo2-source-intake137'


def run():
    started=time.monotonic();source_sha=sha(SOURCE)
    probe=[shutil.which('ffprobe'),'-v','error','-select_streams','v:0','-show_streams','-show_frames',
           '-show_entries','frame=best_effort_timestamp_time:stream=width,height,codec_name,pix_fmt,color_space,color_transfer,color_range,r_frame_rate,duration:stream_side_data=rotation',
           '-of','json',str(SOURCE)]
    meta=json.loads(subprocess.check_output(probe,timeout=30));stream,=meta['streams']
    encoded_w,encoded_h=int(stream['width']),int(stream['height']);assert max(encoded_w,encoded_h)<=4096
    rotations=[float(x['rotation']) for x in stream.get('side_data_list',[]) if 'rotation' in x]
    assert len(rotations)<=1 and (not rotations or rotations[0] in (-180.,-90.,0.,90.,180.))
    rotation=rotations[0] if rotations else 0.
    w,h=(encoded_h,encoded_w) if abs(rotation)==90 else (encoded_w,encoded_h)
    pts=np.asarray([float(row['best_effort_timestamp_time']) for row in meta['frames']])
    assert len(pts)>=12 and np.isfinite(pts).all() and (np.diff(pts)>0).all()
    indices=[int(np.argmin(abs(pts-x))) for x in np.linspace(pts[0],pts[-1],12)]
    assert len(set(indices))==12
    OUT.mkdir(exist_ok=False)
    expression='+'.join(r'eq(n\,%d)'%i for i in indices)
    # Apply container display orientation for contact previews only. It has
    # not been applied to fitting coordinates or camera intrinsics.
    command=[shutil.which('ffmpeg'),'-nostdin','-v','error','-hwaccel','none','-threads','2',
             '-i',str(SOURCE),'-filter_threads','1','-vf','select='+expression,'-fps_mode','passthrough',
             '-frames:v','12','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    atomic(OUT/'plan.json',dict(updated=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha,
        selection='12 fixed equally spaced real PTS targets; nearest decoded index. Container display rotation and resizing are preview only.',
        display_rotation_degrees=rotation,first_attempt_exit1_due_to_nonzero_display_rotation=True,
        sample_indices=indices,source_video=str(SOURCE),decode_command=command,quality_accepted=False))
    tw,th=(480,270) if w>=h else (270,480)
    page=Image.new('RGB',(tw*4,(th+25)*3),(19,23,28));draw=ImageDraw.Draw(page);rows=[]
    with subprocess.Popen(command,stdout=subprocess.PIPE) as process:
        try:
            for j,i in enumerate(indices):
                data=process.stdout.read(w*h*3)
                if len(data)!=w*h*3:raise RuntimeError(f'Short frame {i}')
                im=Image.frombytes('RGB',(w,h),data);im.thumbnail((tw,th),Image.Resampling.LANCZOS)
                x,y=(j%4)*tw,(j//4)*(th+25)
                page.paste(im,(x+(tw-im.width)//2,y+25+(th-im.height)//2))
                draw.text((x+8,y+6),f'{i:04d}  {pts[i]:.3f} s',fill='white')
                rows.append(dict(source_frame_index=i,pts_seconds=float(pts[i])))
                assert time.monotonic()-started<90
            assert process.wait(timeout=10)==0
        finally:
            if process.poll() is None:process.kill();process.wait()
    figure=OUT/'native-source-contact.jpg';page.save(figure,quality=92,subsampling=0)
    assert sha(SOURCE)==source_sha
    result=dict(updated=datetime.now(timezone.utc).isoformat(),source_video=str(SOURCE),source_sha256=source_sha,
        stream=stream,decoded_count=len(pts),duration_pts=float(pts[-1]-pts[0]),encoded_wh=[encoded_w,encoded_h],
        display_wh=[w,h],display_rotation_degrees=rotation,samples=rows,
        contact_sha256=sha(figure),seconds=time.monotonic()-started,source_modified=False,raw_frame_cache_written=False,
        independent_person_verified=False,physical_camera_calibration=False,actual_review_pending=True,
        fitting_started=False,training_started=False,quality_accepted=False)
    atomic(OUT/'intake.json',result)
    print('DEMO2_NATIVE_PTS_INTAKE',w,h,len(pts),source_sha)


if __name__=='__main__':run()
