"""Append exact same-frame unified 016/006 four-panel JPEGs to 00 records."""

import argparse
import json
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import TemporaryDirectory

from PIL import Image,ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parent))
from archive_bound_panels_016 import require, sha
import launch_unified_cold_016_006 as layout


COMMIT='79b906fff9b62b6833041194fcc2ea3f64a1d464'
TAG='统一79b906f'
def record_jpeg(source,target,crop):
    x0,y0,x1,y1=crop
    width,height=x1-x0,y1-y0
    with Image.open(source) as p:
        im=p.convert('RGB')
    require(im.size==(4*width,height+26),'Four-panel image does not match its native crop.')
    im.crop((0,26,im.width,im.height)).save(target,quality=95,subsampling=0)


def contact_sheet(rows,panels,target):
    """Index all four aligned columns for each selected source frame."""
    width,height=900,325
    sheet=Image.new('RGB',(3*width,8*height),'#161616')
    pairs=sorted(zip(rows,panels,strict=True),key=lambda pair:pair[0]['source_frame_index'])
    require(len(pairs)==24,'The contact sheet requires all 24 selected poses.')
    for i,(_,path) in enumerate(pairs):
        x,y=i%3*width,i//3*height
        with Image.open(path) as source:
            tile=ImageOps.contain(source.convert('RGB'),(width-8,height-8),Image.Resampling.LANCZOS)
        sheet.paste(tile,(x+(width-tile.width)//2,y+(height-tile.height)//2))
    sheet.save(target,quality=95,subsampling=0)


def archive(subject,publish):
    root=layout.root_for(subject)
    run=root/'run'
    dest=Path('/mnt/d/00/记录')/subject
    require(dest.resolve(strict=True)==dest and dest.is_dir(),
            'Choose the existing image-record directory.')
    pins={}

    def pin(path,expected=None):
        path=Path(path).resolve(strict=True)
        actual=sha(path)
        require((expected is None or expected==actual) and pins.get(path,actual)==actual,
                f'Hash mismatch: {path}')
        pins[path]=actual
        return path

    def read(path,expected=None):
        return json.loads(pin(path,expected).read_text(encoding='utf-8-sig'))

    terminal=read(root/'wrapper-terminal.json')
    plan=read(root/'preregistration.json')
    result=read(run/'result.json',terminal['result_sha256'])
    status=read(run/'status.json')
    fit=read(run/'fit/official-fit-result.json',plan['staged_sha256']['fit/official-fit-result.json'])
    selection=read(run/'fit/selection.json',plan['staged_sha256']['fit/selection.json'])
    metadata=read(run/'run.json',plan['staged_sha256']['run.json'])
    require(terminal['state']==status['state']==result['state']=='completed'
            and terminal['candidate_commit']==plan['candidate_commit']==COMMIT
            and terminal['fit_calls']==result['fit_calls']==0
            and terminal['fit_reused'] is result['fit_reused'] is metadata['fit_reused'] is True
            and result['gaussian_from_scratch'] is True and result['from_scratch'] is False
            and result['reused_fit_preregistration_sha256']==pins[root/'preregistration.json']
            and status['format_passed'] is result['format_passed'] is True
            and result['steps']==10000 and result['frames']==24
            and status['sha256']==result['sha256']
            and fit['selection']==selection and fit['new_fitting_started']==(subject=='006'),
            'Unified terminal, reused pose provenance, or standard export failed verification.')
    pin(Path(metadata['video']),plan['source_video_sha256'])
    pin(run/result['ply'],result['sha256'])
    pin(run/result['scene'],result['scene_sha256'])
    pin(run/'fit/fitted.npz',fit['fit_sha256'])
    rows=read(run/'data/training-inputs.json',fit['training_inputs_sha256'])['records']
    replay_path=run/result['bound_replay']
    panel_path=run/result['bound_panels']
    replay=read(replay_path,result['bound_replay_sha256'])
    panels=read(panel_path,result['bound_panels_sha256'])
    bound_path=replay_path.parent.parent/'bound-replay.json'
    bound=read(bound_path,replay['bound_replay_sha256'])
    pin(bound_path.parent/'bound-local.ply',bound['bound_ply_sha256'])
    pin(replay_path.parent/replay['mesh_file'],replay['mesh_sha256'])
    selected=plan['selected_frames']
    ref=plan['reference_frame']
    order=[ref]+[f for f in selected if f!=ref]
    require(len(order)==len(set(order))==24
            and selected==selection['selected_frames']==terminal['selected_frames']
            and order==[r['frame'] for r in rows]==[r['frame'] for r in replay['records']]
            ==[r['frame'] for r in panels['panels']]==[r['frame'] for r in bound['frames']]
            and not set(order)&set(plan['retained_holdouts'])
            and plan['retained_holdouts']==selection['retained_holdouts']==replay['retained_holdouts']
            and result['reference_frame_id']==replay['reference_frame_id']==panels['reference_frame_id']==ref
            and replay['optimizer_updates']==panels['optimizer_updates']==0
            and panels['same_frame_bound_gaussians'] is panels['native_crops'] is True
            and panels['replay_receipt_sha256']==result['bound_replay_sha256'],
            'Native replay, Gaussian, mesh, features, or protected frame identities differ.')
    records=[]
    for slot,(row,splat,panel,bind) in enumerate(zip(rows,replay['records'],panels['panels'],bound['frames'],strict=True)):
        frame=row['frame']
        require(slot==splat['pose_index']==panel['pose_index']==bind['pose_index']
                and row['source_frame_index']==splat['source_frame_index']==panel['source_frame_index']==bind['source_frame_index']
                and row['source_sha256']==splat['source_sha256']==panel['source_sha256']==bind['source_sha256']
                and row['mask_sha256']==splat['mask_sha256']==bind['mask_sha256']
                and row['K']==splat['camera']['K'] and row['w2c']==splat['camera']['w2c']
                and splat['image']==panel['panel']==frame+'.png'
                and panel['gaussian_sha256']==splat['image_sha256'],
                f'Mixed image, geometry or camera on {frame}.')
        pin(row['image'],row['source_sha256'])
        pin(row['mask'],row['mask_sha256'])
        pin(replay_path.parent/splat['image'],splat['image_sha256'])
        source=pin(panel_path.parent/panel['panel'],panel['panel_sha256'])
        records.append((source,dest/f'3DGS_{subject}_{TAG}_原图-高斯-Mesh-特征点_{frame}.jpg',panel['crop_xyxy']))
    report=dict(state='verified',subject=subject,commit=COMMIT,frames=len(records),reference_frame=ref,
                fit_reused=True,cold_video_clock=False,quality_accepted=False,
                selected=[dict(frame=row['frame'],source_frame_index=row['source_frame_index']) for row in rows],
                result_sha256=terminal['result_sha256'],panel_receipt_sha256=result['bound_panels_sha256'],
                images=[])
    if not publish:
        return report

    with TemporaryDirectory(prefix=f'unified-{subject}-record-') as temp:
        staging=Path(temp)
        files=[]
        for source,target,crop in records:
            item=staging/target.name
            record_jpeg(source,item,crop)
            files.append((item,target))
        summary=dest/f'分析_{subject}_{TAG}_入选四合一总览_24帧.jpg'
        contact_sheet(rows,[item for item,_ in files],staging/summary.name)
        files.append((staging/summary.name,summary))
        prepared=[]
        for source,target in files:
            with Image.open(source) as im:
                im.verify()
            digest=sha(source)
            require(not target.is_symlink() and (not target.exists() or sha(target)==digest),
                    f'Existing record differs, preserving {target}.')
            prepared.append((source,target,digest))
        require(all(sha(p)==h for p,h in pins.items()),'A verified source changed during conversion.')
        created=[]
        try:
            for source,target,digest in prepared:
                if not target.exists():
                    with source.open('rb') as original,target.open('xb') as copy:
                        created.append(target)
                        copyfileobj(original,copy)
                require(sha(target)==digest,f'Record hash changed: {target}.')
                report['images'].append(dict(path=str(target),sha256=digest))
        except BaseException:
            for target in created:
                target.unlink(missing_ok=True)
            raise
    report['state']='archived'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    print(json.dumps(archive(args.subject,args.publish),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
