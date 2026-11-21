"""Research observation admission independent of matcher confidence.

No geometry, optimization, coordinate correction or point deletion occurs here.
Evidence states must describe the same source/frame/material observation.
This module's controls do not certify a real camera or measurement.
"""
from dataclasses import dataclass, asdict
import math

SCHEMA='shuolee-material-observation-v1'
SPLITS={'fit_background','heldout_background','heldout_human'}

@dataclass(frozen=True)
class Observation:
    source_sha256:str
    frame_id:int
    track_id:int
    frozen_family_id:str
    frozen_split:str
    group:str
    raw_uv:tuple
    width:int
    height:int
    tracker_numeric_validity:bool
    visibility_evidence:str='unknown'
    material_identity_evidence:str='unknown'
    static_evidence:str='unknown'
    measurement_uncertainty_px:float|None=None
    evidence_source:tuple=()
    development_exposure:bool=True
    native_pixel_center_convention:str='integer_centers'

    @property
    def key(self):return (self.source_sha256,self.frame_id,self.track_id)

    def record(self):return {'schema':SCHEMA,**asdict(self)}

def validate_observation(o):
    if len(o.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in o.source_sha256):raise ValueError('Source hash required')
    if o.frozen_split not in SPLITS or not o.frozen_family_id:raise ValueError('Frozen split/family required')
    if o.frame_id<0 or o.track_id<0 or o.width<1 or o.height<1:raise ValueError('Invalid identity or dimensions')
    if o.native_pixel_center_convention!='integer_centers':raise ValueError('Native integer-center observations required; explicit conversion belongs at renderer boundary')
    if len(o.raw_uv)!=2:raise ValueError('Two coordinates required')
    if o.visibility_evidence not in {'unknown','visible','occluded'}:raise ValueError('Visibility state unsupported')
    if o.material_identity_evidence not in {'unknown','confirmed','contradicted'}:raise ValueError('Material state unsupported')
    if o.static_evidence not in {'unknown','bounded_static','moving'}:raise ValueError('Static state unsupported')
    if o.measurement_uncertainty_px is not None and (not math.isfinite(o.measurement_uncertainty_px) or o.measurement_uncertainty_px<0):raise ValueError('Invalid uncertainty')
    if (o.visibility_evidence!='unknown' or o.material_identity_evidence!='unknown' or o.static_evidence!='unknown' or o.measurement_uncertainty_px is not None) and not o.evidence_source:raise ValueError('Claims require evidence provenance')

def point_reasons(o):
    validate_observation(o)
    reasons=[]
    if not o.tracker_numeric_validity:reasons.append('numeric_tracking_not_valid')
    if not all(math.isfinite(float(v)) for v in o.raw_uv):reasons.append('nonfinite_raw_coordinate')
    elif not (0<=o.raw_uv[0]<o.width and 0<=o.raw_uv[1]<o.height):reasons.append('outside_native_image')
    if o.visibility_evidence!='visible':reasons.append('visibility_'+o.visibility_evidence)
    if o.material_identity_evidence!='confirmed':reasons.append('material_'+o.material_identity_evidence)
    if o.measurement_uncertainty_px is None:reasons.append('measurement_uncertainty_unknown')
    return reasons

def camera_fit_permission(o):
    reasons=point_reasons(o)
    if o.frozen_split!='fit_background':reasons.append('frozen_holdout_cannot_fit')
    if o.group!='ground_candidate' and o.group!='static_scene':reasons.append('not_static_scene_candidate')
    if o.static_evidence!='bounded_static':reasons.append('staticness_'+o.static_evidence)
    return dict(allowed=not reasons,reasons=reasons)

def independent_face_admission(o, prediction):
    """Necessary point checks; full coverage has a separate non-vacuous audit.

    Prediction provenance records both camera-fit families and which image
    observations formed the 3D point. Reprojecting a point fit to this target
    is not an independent heldout prediction. Numeric confidence is ignored.
    """
    reasons=point_reasons(o)
    if o.frozen_split!='heldout_human' or o.group!='face_candidate':reasons.append('not_heldout_face')
    if o.development_exposure:reasons.append('method_development_exposed')
    if prediction is None:return dict(accepted=False,reasons=reasons+['no_independent_prediction'],error_px=None)
    if tuple(prediction.get('target_key',()))!=o.key:reasons.append('prediction_identity_mismatch')
    if o.frozen_family_id in prediction.get('camera_fit_families',[]):reasons.append('heldout_family_used_for_camera_fit')
    support={tuple(k) for k in prediction.get('triangulation_observation_keys',[])}
    if o.key in support:reasons.append('target_used_in_triangulation')
    if len(support)<2:reasons.append('insufficient_other_views_for_prediction')
    if any(k[0]!=o.source_sha256 or k[2]!=o.track_id for k in support):reasons.append('mixed_material_or_source_in_triangulation')
    if not prediction.get('frozen_camera_evidence'):reasons.append('camera_evidence_missing')
    uv=prediction.get('uv',[])
    if len(uv)!=2 or not all(math.isfinite(float(v)) for v in uv):
        reasons.append('invalid_prediction');error=None
    else:
        error=math.hypot(uv[0]-o.raw_uv[0],uv[1]-o.raw_uv[1])
        if not math.isfinite(error) or error>=1.0:reasons.append('reprojection_not_strictly_below_1px')
    return dict(accepted=not reasons,reasons=reasons,error_px=error,measurement_uncertainty_px=o.measurement_uncertainty_px,
        conservative_error_plus_uncertainty_px=None if error is None or o.measurement_uncertainty_px is None else error+o.measurement_uncertainty_px)

def audit_family_splits(observations):
    families={}
    for o in observations:
        validate_observation(o);families.setdefault((o.source_sha256,o.frozen_family_id),set()).add(o.frozen_split)
    mixed=[list(k) for k,v in families.items() if len(v)!=1]
    return dict(passed=not mixed,mixed_families=mixed)

def audit_coverage(required_frames,required_regions,registered_frames,point_results):
    """Coverage is necessary, not a replacement for every heldout point passing."""
    required=set(required_frames);regions=set(required_regions)
    have={(r['frame_id'],r['region']) for r in point_results if r['accepted']}
    missing=[(f,r) for f in sorted(required) for r in sorted(regions) if (f,r) not in have]
    failed=[r for r in point_results if not r['accepted']]
    missing_cameras=sorted(required-set(registered_frames))
    return dict(accepted=bool(required and regions and point_results) and not missing and not failed and not missing_cameras,
        missing_camera_frames=missing_cameras,missing_face_region_views=missing,failed_observations=failed,
        accepted_observation_count=sum(r['accepted'] for r in point_results),scope='Necessary coverage and all-observation checks only; body/novel-view acceptance remains separate')
