"""Compact evidence register referencing actual receipts, not a second report."""
import json
from pathlib import Path
from datetime import datetime,timezone
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/clothed-surface-119')
refs=[]
for pattern in ['*representation-audit.json','*-train-*-review/actual-review.json','*-quality-review.json','*-viewer-ui-review.json','sofa-fov-diagnostic/result.json','phase-result.json']:
    for p in sorted(R.glob(pattern)):refs.append(str(p.relative_to(R)))
value=dict(updated=datetime.now(timezone.utc).isoformat(),evidence=refs,
    established=['Previous local SMPL-bound model penalized center displacement beyond one triangle scale and kept at least one point per bare triangle.',
        '119 explicitly recomputed its official prior and neutral-template fit; raw pixels/cameras/masks/2D observations were fixed controls.',
        'The new model has binding=None and independent world positions; initial Gaussian centers sample fresh clothed silhouette envelopes from112 portrait or89 sofa fixed views.',
        'Actual same1000-step comparison shows stronger skirt/bag support with actual centers, while facial ghosting remains.',
        'Sofa matched18k comparison and actual OpenFile show centers supporting shirt/pants/hair; rear/feet still fail quality.',
        'CPU-only fixed-grid counterfactual: counting out-of-image projections as negatives removes1046 supported voxels, mainly at feet; no corrective training yet.'],
    not_established=['Exact skin/clothing depth, especially concavities and unseen backside.',
        'Unique source of all facial/full-body ghosting.',
        'Physical applicability of supplied K and unknownFISHEYE_8/export mapping.',
        'Company00 deployment or full reconstruction quality acceptance.'])
(R/'evidence-ledger.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('EVIDENCE_REFERENCES',len(refs))
