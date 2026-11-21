"""CPU-only source, fresh-preparation, review and CVO evidence bindings."""
from pathlib import Path
from copy import deepcopy
from native_training_admission122_v2 import read,sha,checked_asset
from native_training_policy122 import digest,validate_policy,complete_graph,cycle_schedule

H=Path(__file__).parent
PIPELINE_FILES=['native_pipeline_binding122.py','prepare_native_person122_v3.py','train_native_person122_v2.py',
    'fitting_output_residual122.py','native_residual_fit122.py','fit_gate_output_residual122.py','native_surface122_v2.py',
    'native_training_admission122_v2.py','material_geometry122_v2.py','native_training122_v5.py','native_training_policy122.py',
    'native_training122_v4.py','native_training122_v2.py','native_training122.py','native_covisibility122.py',
    'native_residual_camera122.py','native_residual_domain122.py','native_portrait_renderer122_v3.py','native_scene_package122_v5.py']

def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def pipeline_sources():return {n:ref(H/n) for n in PIPELINE_FILES}
def verify_sources(record):
    if set(record)!=set(PIPELINE_FILES):raise ValueError('Complete pipeline code provenance required')
    for name,r in record.items():
        checked_asset(H,r)
        if sha(H/name)!=r['sha256']:raise ValueError('Preparation and execution code differ:'+name)

def verify_preparation(manifest,base):
    initial=read(checked_asset(base,manifest['fresh_initialization']))
    if initial.get('schema')!='shuolee-fresh-clothing-initialization-v2-output-residual':raise ValueError('Residual fresh initialization schema required')
    binding=read(checked_asset(base,initial['pipeline_binding']))
    expected=dict(run_id=manifest['run_id'],source_sha256=manifest['source']['asset']['sha256'],
        cameras_sha256=manifest['cameras']['sha256'],frame_plan_sha256=manifest['frame_plan']['sha256'],
        frames_sha256=digest(manifest['frames']),observations_sha256=manifest['observations']['sha256'],
        material_evidence_sha256=manifest['physical_evidence']['sha256'])
    if any(binding.get(k)!=v for k,v in expected.items()):raise ValueError('Preparation source/camera/material/mask binding mismatch')
    parent=read(checked_asset(base,binding['source_manifest']))
    stripped=deepcopy(manifest)
    for k in ['fresh_initialization','surface_review']:stripped.pop(k,None)
    if parent!=stripped:raise ValueError('Only this run initialization and review may be added to frozen training input')
    verify_sources(binding['pipeline_sources'])
    fit=read(checked_asset(base,initial['fresh_fit_receipt']))
    for name in ['new_prior_receipt','new_prior_prediction','original_init','fit_result','fit_protocol']:
        checked_asset(base,fit[name])
    if fit.get('pipeline_sources_sha256')!=digest(binding['pipeline_sources']):raise ValueError('Fresh fit lacks exact pipeline source binding')
    fit_review=read(checked_asset(base,binding['fit_review']))
    if fit_review.get('fitted_sha256')!=initial['fit_arrays']['sha256'] or fit_review.get('actual_same_camera_fit_reviewed') is not True or fit_review.get('allow_surface') is not True:
        raise ValueError('Actual exact-fit review required')
    checked_asset(base,fit_review['review_evidence'])
    gate=read(checked_asset(base,binding['geometry_gate']))
    if not gate.get('coarse_initialization_gate_passed') or gate.get('fit_sha256')!=initial['fit_arrays']['sha256']:
        raise ValueError('Fit geometry gate does not describe this initialization')
    surface=read(checked_asset(base,binding['surface_result']))
    if surface.get('arrays_sha256')!=initial['arrays']['sha256']:raise ValueError('Surface result and initial arrays differ')
    decision=read(checked_asset(base,manifest['surface_review']))
    if decision.get('arrays_sha256')!=initial['arrays']['sha256'] or any(decision.get(k) is not True for k in ['actual_same_camera_surface_reviewed','actual_center_distribution_reviewed','allow_training']):
        raise ValueError('Actual surface and free-center review required')
    checked_asset(base,decision['review_evidence'])
    return dict(pipeline_binding_sha256=initial['pipeline_binding']['sha256'],source_manifest_sha256=binding['source_manifest']['sha256'],
        pipeline_source_sha256=digest(binding['pipeline_sources']),surface_review_sha256=manifest['surface_review']['sha256'])

def sampling_for_input(manifest,base,cameras):
    policy=validate_policy(manifest['training'].get('sampling_policy'))
    spec_ref=manifest['training'].get('covisibility_evidence')
    if policy['kind']=='uniform_single':
        if spec_ref is not None:raise ValueError('Uniform policy cannot silently consume CVO material evidence')
        return policy,None
    spec=read(checked_asset(base,spec_ref))
    expected=dict(schema='shuolee-cvo-fit-evidence-v1',source_sha256=manifest['source']['asset']['sha256'],
        cameras_sha256=manifest['cameras']['sha256'],frame_plan_sha256=manifest['frame_plan']['sha256'])
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('CVO evidence source/frame/camera mismatch')
    evidence={r['sha256'] for r in spec['evidence_assets'] if checked_asset(base,r)}
    frozen=read(checked_asset(base,manifest['observations']))
    frozen=frozen['observations'] if isinstance(frozen,dict) else frozen
    held={o['frozen_family_id'] for o in frozen if o['frozen_split']!='fit_background'}
    mapping={f['frame_id']:c['id'] for f,c in zip(manifest['frames'],cameras)}
    records=deepcopy(spec['observations'])
    for r in records:
        if r['evidence_sha256'] not in evidence:raise ValueError('Unbound CVO physical identity evidence')
        if r['family_id'] in held and r['split']=='fit':raise ValueError('Heldout physical family cannot guide CVO pairing')
        r['frame_id']=mapping[r['frame_id']]
    graph=complete_graph(cameras,records,source_sha256=expected['source_sha256'],min_shared=spec['min_shared'])
    cycle_schedule(cameras,policy,manifest['training']['seed'],graph=graph)
    return policy,graph
