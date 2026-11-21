"""Record the user's stricter acceptance requirement; do not refit any experiment."""
from global_fixedk121 import *
contract=dict(updated=now(),source='direct_user_message',user_quote='我们这个项目对人脸的误差容忍度要小于一个像素',
 metric='Euclidean2D reprojection distance per independently held-out, physically credible facial material correspondence',
 domain='Original encoded MP4 native pixel resolution; errors at any training downsample are converted back before checking',
 strict_threshold_px=1.,comparison='error_px <1.0 for every evaluated valid facial observation; never median/mean-only',
 coverage='Per-frame and per-view coverage of visible eyes/nose/mouth/face contour must be reported. Too few reliable points, missing views, uncertain identity or unknown measurement precision cannot pass.',
 identity='A single physical material point shares one3D identity across views. Repeated semantic landmark index alone is not evidence.',
 holdout='Whole physical track families kept out of camera fitting. For a prediction test, the tested view is excluded from point triangulation.',
 exclusions='Only documented visibility/occlusion/identity exclusions independently established before consulting residuals. Never drop >=1px points just to pass. Report excluded counts and causes.',
 uncertainty='Subpixel detector coordinates and low optimizer loss do not establish subpixel physical accuracy. Unknown observation/calibration uncertainty means unverified.',
 effect='Replaces earlier permissive face median<=1.5/P90<=3 training gate for current and future delivery. Original preregistered diagnostic thresholds preserved as history; no posthoc refit.',
 guarantees='This is an acceptance target, not a guarantee achievable from current lossy processed videos.',training_started=False)
receipts=[]
for s in ['sofa','portrait']:
    rows=read(ROOT/s/'global-fixedk/heldout-predictions.json')
    xs=[r for r in rows if r['region']=='face_roi' and r['state']=='predicted']
    receipts.append(dict(subject=s,approximate_face_ROI_not_physical_ground_truth=True,valid_prediction_count=len(xs),fraction_strictly_below1=sum(x['error_px']<1 for x in xs)/len(xs),count_ge1=sum(x['error_px']>=1 for x in xs),error_px=stats([x['error_px'] for x in xs]),accepted=False))
save(R/'face-subpixel-acceptance.json',dict(contract=contract,current_diagnostic_reevaluation=receipts))
protocol=read(R/'protocol.json');protocol['current_face_training_admission']=contract;save(R/'protocol.json',protocol)
print(json.dumps(receipts,indent=2))
