"""After actual visual inspection only, bind six-camera/36-view review to final PLY."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
p=argparse.ArgumentParser()
p.add_argument('--subject',required=True)
p.add_argument('--summary',required=True)
p.add_argument('--novel-summary',required=True)
p.add_argument('--reviewed-all-images',action='store_true',required=True)
a=p.parse_args()
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')/a.subject
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
assert not (REC/f'{a.subject}-quality-review.json').exists(), 'Existing final review; inspect before changing.'
result=read(ROOT/'result.json');v=read(ROOT/'export/validation.json')
assert result['steps']==18000 and result['from_scratch'] and v['format_passed']
digest=sha(ROOT/'export/point_cloud.ply');assert digest==v['sha256']
folder=REC/f'{a.subject}-train-018000-review';novel=REC/f'{a.subject}-novel-final-review'
idx=read(folder/'index.json');ni=read(novel/'index.json')
frames=[x['frame'] for x in idx['images'] if x['region']=='body'];assert len(set(frames))==6
assert len(ni['source_images'])==36 and len(ni['images'])==6
for parent,data in [(folder,idx),(novel,ni)]:
    for entry in data['images']:assert sha(parent/entry['file'])==entry['sha256']
now=datetime.now(timezone.utc).isoformat();ncam=len(read(ROOT/'export/cameras.json')['records'])
save(folder/'actual-review.json',dict(updated=now,actual_visual_review=True,index_sha256=sha(folder/'index.json'),
    steps=18000,summary=a.summary,actually_reviewed_body_frames=frames,
    actually_reviewed_face_regions=[x['frame'] for x in idx['images'] if x['region']=='face'],
    quality_accepted=False,physical_calibration_accepted=False,training_parameters_changed=False,source_sha256=digest))
save(novel/'actual-review.json',dict(updated=now,actual_visual_review=True,index_sha256=sha(novel/'index.json'),
    summary=a.novel_summary,continuous_novel_views_reviewed=36,
    sheets=[dict(path=str(novel/x['file']),sha256=x['sha256']) for x in ni['images']],quality_accepted=False))
quality=dict(updated=now,subject=a.subject,source_sha256=digest,steps=18000,points=v['points'],camera_count=ncam,
    initial_points=200000,actual_same_camera_review=True,same_camera_frames=frames,continuous_novel_views_reviewed=36,
    quality_accepted=False,clothed_center_distribution_observed=True,ghosting_resolved=False,
    subpixel_material_admission_passed=False,summary=a.summary,novel_summary=a.novel_summary,
    physical_calibration_accepted=False,novel_view_ground_truth_available=False,
    source_reviews=[str(folder/'actual-review.json'),str(novel/'actual-review.json')],
    format_passed=v['format_passed'],local_viewer_UI='pending',official_SIBR_GUI_verified=False,
    independent_Spark_GUI_verified=False,company00_verified=False)
save(REC/f'{a.subject}-quality-review.json',quality)
ledger=read(REC/'evidence-ledger.json');ledger['entries'].append(dict(kind=f'{a.subject}_final18k_actual_quality_review',
    date=now,summary=a.summary,novel_summary=a.novel_summary,quality_accepted=False,source_sha256=digest,
    quality_record=str(REC/f'{a.subject}-quality-review.json')))
ledger['updated']=now;save(REC/'evidence-ledger.json',ledger)
print(json.dumps(quality,ensure_ascii=False))
