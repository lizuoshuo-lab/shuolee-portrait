"""Read-only regional audit of already-frozen tracks. Never fit or reject by error."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def native(p):
    if p.startswith('/opt/'):return Path('//wsl.localhost/Ubuntu-20.04'+p)
    if p.startswith('/mnt/'):return Path(p[5].upper()+':/'+p[7:])
    return Path(p)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def summary(values):
    x=np.asarray(values,float)
    return dict(n=len(x),median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None,
        max=float(x.max()) if len(x) else None,lt1px=int(np.count_nonzero(x<1)),ge1px=int(np.count_nonzero(x>=1)),
        ge10px=int(np.count_nonzero(x>=10)),ge100px=int(np.count_nonzero(x>=100)))
def main(subject):
    start=time.monotonic();root=ROOT/subject;out=REC/f'{subject}-regional-support';out.mkdir(exist_ok=False)
    files=[root/'camera/held-evaluation.json',root/'camera/solver-output.npz',root/'data/training-inputs.json']
    plan=dict(created=datetime.now(timezone.utc).isoformat(),subject=subject,kind='read-only frozen-observation regional audit',
        purpose='Check whether global residual statistics conceal person/head errors or weak support; not a camera candidate.',
        regions=['all','person_mask','outside_person_mask','head_box_upper23percent','RTMW_face_ROI_only'],
        definition='Use each native decoded observation without warp. Person from frozen SAM2 mask. Head from top23percent of frozen person box. RTMW face ROI is localization only and may include hats.',
        no_error_filtering=True,no_parameter_update=True,no_feature_extraction=True,no_optimizer=True,
        physical_material_identity_reviewed=False,negative_control_is_not_verified_static=True,
        budget_seconds=120,sources=[dict(path=str(p),sha256=sha(p)) for p in files])
    save(out/'plan.json',plan)
    held=read(files[0]);rows=read(files[2])['records'];features=[];masks=[]
    for i,row in enumerate(rows):
        f=np.load(root/f'camera/features-{i:03d}.npz')
        xy=f['xy'] if 'xy' in f else f['points']
        features.append(xy)
        with Image.open(native(row['mask'])) as im:masks.append(np.asarray(im.convert('L')))
        if time.monotonic()-start>120:raise RuntimeError('Bounded read-only audit exceeded120seconds.')
    groups={k:[] for k in plan['regions']};tracks={k:set() for k in groups};views={k:set() for k in groups};observations=[]
    for tr in held['records']:
        checks=tr['nodes'][1::2]
        assert len(checks)==len(tr['errors']) and [i for i,j in checks]==tr['test_frames']
        for (i,j),error in zip(checks,tr['errors']):
            x,y=map(float,features[i][j]);row=rows[i];w,h=row['wh'];ix,iy=int(round(x)),int(round(y))
            in_mask=0<=ix<w and 0<=iy<h and masks[i][iy,ix]>127
            b=row['initial_box_xyxy'];head=[b[0],b[1],b[2],b[1]+.23*(b[3]-b[1])]
            inside=lambda b: b is not None and b[0]<=x<=b[2] and b[1]<=y<=b[3]
            cats=['all','person_mask' if in_mask else 'outside_person_mask']
            if inside(head):cats.append('head_box_upper23percent')
            if inside(row.get('face_box_xyxy')):cats.append('RTMW_face_ROI_only')
            for cat in cats:groups[cat].append(error);tracks[cat].add(tr['id']);views[cat].add(i)
            observations.append(dict(track=tr['id'],frame_index=i,source_frame_index=row['source_frame_index'],xy=[x,y],
                error_px=error,regions=cats,whole_track_positive_depth=tr['positive_depth']))
    result=dict(created=datetime.now(timezone.utc).isoformat(),subject=subject,elapsed_seconds=time.monotonic()-start,
        groups={cat:dict(**summary(v),tracks=len(tracks[cat]),covered_views=len(views[cat]),view_indices=sorted(views[cat])) for cat,v in groups.items()},
        all_views=len(rows),held_tracks=len(held['records']),nonpositive_depth_tracks=sum(not t['positive_depth'] for t in held['records']),
        excluded_for_error=0,fit_or_holdout_membership_changed=False,formal_calibration_accepted=False,
        caveats=['SAM2 outside is only a region, not verified static scenery.',
                 'RTMW face ROI is not a physical material correspondence and firstwinterhat view detects rear/sidehat.',
                 'These frozen SIFT tracks do not establish physical identity or family independence.',
                 'Large residuals do not uniquely distinguish camera error, bad correspondence and subject motion.'])
    save(out/'observations.json',observations);save(out/'result.json',result)
    assert all(sha(Path(x['path']))==x['sha256'] for x in plan['sources'])
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subject',required=True);a=p.parse_args();main(a.subject)
