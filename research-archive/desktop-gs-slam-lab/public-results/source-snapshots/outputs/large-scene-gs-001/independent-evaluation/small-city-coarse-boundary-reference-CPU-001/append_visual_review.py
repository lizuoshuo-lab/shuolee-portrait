from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'report.json').read_text(encoding='utf-8'))
assert not (HERE/'visual-review.json').exists()
checked=[]
for j in (0,2,4,6):
    row=report['views'][j]
    original=row['original_images']['own-coarse-reference/flat']
    path=Path(original['original_raw_pair'])
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==original['original_raw_pair_sha256']
    checked.append(dict(name=row['name'],original_raw_pair=str(path),original_raw_pair_sha256=actual,
        actual_whole_raw_pair_visually_inspected=True,panels='GT left, original predictedRGB right;512x172; no annotations or image edits'))
review=dict(status='actual_four_fixed_coarse_original_raw_pairs_visually_inspected',
    observed_utc=datetime.now(timezone.utc).isoformat(),actual_visually_inspected_raw_pairs=4,
    selection_rule='First of the two frozen geometry002 views in each filename pass group; indices0,2,4,6. Chosen before reading dense scores.',
    original_pairs=checked,source_reference_report_sha256=hashlib.sha256((HERE/'report.json').read_bytes()).hexdigest(),
    findings=['Building and street outlines broadly align in the four inspected pairs.',
        'Vehicle edges, vegetation/branches, lettering and finer road texture are visibly smoother in the coarseSH1 predictions.',
        'Some source target panels contain black regions while raw predicted panels contain scene content there. Raw comparisons alone do not establish mask validity or a seam cause; mainGTm2/predm and symmetric auxiliary metrics remain separate.'],
    limitations=['Only four of the eight fixed reference views were visually inspected; only raw whole pairs, not all masked pairs.',
        'This is the actual coarse reference, not a dense pre/post City comparison, not all scene quality, not proof of seam removal.',
        'No view, mask or preselected ROI was dropped/changed after seeing images; no training or threshold retuning.'],
    CUDA_initialized=False,images_generated_or_edited=False,City_dense_boundary_visual_review_complete=False)
(HERE/'visual-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'actualRawPairs':4,'denseCityComplete':False}))
