from pathlib import Path
import json,hashlib
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-static-material')
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=json.loads((O/'visual-review.json').read_text(encoding='utf-8'))
assert review['reviewed_rows']==36 and review['reviewed_contexts']==16
assert not (O/'result-reviewed.json').exists()
save(O/'review-publication-first-error.json',dict(error="TypeError: dict() got multiple values for keyword argument 'state'",scope='Metadata serialization after completed visual-review.json. No detector, geometry or training rerun.',visual_review_sha256=hashlib.sha256((O/'visual-review.json').read_bytes()).hexdigest()))
result=json.loads((O/'run/result.json').read_text());result.update(review_updated=review['updated'],state='complete_reviewed_support_failed',coarse_fixed_fit=3,coarse_fixed_heldout=1,contradicted_review_families=6,independent_face_heldout=0,geometry_admitted=False,quality_accepted=False,summary=review['summary'])
save(O/'result-reviewed.json',result)
save(O/'progress.json',dict(updated=review['updated'],state=result['state'],pid=None,training_started=False,new_camera=False,result='result-reviewed.json'))
print(json.dumps(dict(counts=review['counts'],support=review['coarse_fixed_support'],summary=review['summary']),ensure_ascii=False))
