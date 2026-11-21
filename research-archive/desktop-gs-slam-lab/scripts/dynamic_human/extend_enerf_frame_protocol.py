"""One-time explicit extension of the frame-zero diagnostic to a chosen frame."""
from pathlib import Path
p=Path(__file__).resolve().parent
changes={
 'prepare_enerf_geometry.py': [('frame=0,resolution=', 'frame=a.frame,resolution=')],
 'train_enerf_first_frame.py': [('frame=0,author_geometry_used=', "frame=json.loads((out/'geometry-audit.json').read_text())['frame'],author_geometry_used="),('frame=0,native_resolution=', "frame=json.loads((out/'geometry-audit.json').read_text())['frame'],native_resolution=")],
 'evaluate_enerf_first_frame.py': [('a=ap.parse_args();out=', "ap.add_argument('--frame',type=int,default=0);a=ap.parse_args();out="),("r['frame']!=0", "r['frame']!=a.frame"),('frame=0,foreground_psnr_db=', 'frame=a.frame,foreground_psnr_db=')],
}
for name,pairs in changes.items():
    f=p/name;s=f.read_text(encoding='utf8')
    for before,after in pairs:
        assert s.count(before)==1,(name,before);s=s.replace(before,after)
    f.write_text(s,encoding='utf8')
