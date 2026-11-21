"""CPU-only index of unmodified original City paired images and fixed patch errors.

Views/tiles are frozen by geometry002, not by dense RGB scores. Dense mode only
runs after the actual 1404-frame independent posterior proof. A separate real
coarse reference exercises the image schema and is never dense completion.
No images are generated, cropped, annotated, rewritten or uploaded here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
EVALS = BASE / 'independent-evaluation'
CANON = EVALS / 'small-city-local-coarse-003-flat-heldout-001'
SELECTION = EVALS / 'small-city-boundary-selection-CPU-002'
REVIEW = EVALS / 'small-city-boundary-selection-002-independent-review-CPU-001/review.json'
CANON_INPUT_SHA = '60ab56f705c6db347b00f821b08a44e49c0d2dca4f0f06e9a978b9c73c8bce25'
REVIEW_SHA = 'ea14f3e75d071c8cd1a355248956876e7e6018989b94be05266fa5fbdbfa902d'
SELECTION_SHA = '6b865fca8a94c90a3b8e69902c6dd67bb93bee0cf85b7d447e472ee98bdfde88'
POSTERIOR = EVALS / 'city-hierarchy-posterior-auditor-CPU-001/audit_city_1404.py'
POSTERIOR_SHA = 'bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133'
ADAPTER = EVALS / 'city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py'
ADAPTER_SHA = 'c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610'
CORE = ROOT / 'scripts/large_scene_gs/evaluate_official_hierarchy.py'
CORE_SHA = '672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-005'
CONFIG_SHA = '55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'
WORKFLOW_SHA = '0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84'
COARSE_SHA = 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31'
MODELS = ('merged-unoptimized', 'merged-post')
SETTINGS = ('leaves', 'tau1', 'tau3', 'tau5', 'tau6', 'tau15')
POSTERIOR_STATUS = 'actual1404_complete_independent_CPU_posterior_audit_passed'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identifier(value):
    require(value.replace('-', '').isalnum() and len(value) <= 110, 'Invalid local output identifier')
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def bound_file(path, expected, protected):
    path = Path(path)
    require(path.is_file() and digest(path) == expected, 'Missing/changed actual bound artifact: '+str(path))
    protected[str(path)] = expected


def dense_completion_binding(evaluation, audit_directory):
    """Read actual posterior and source-result identities; no evaluator import."""
    audit_path = audit_directory/'completion-audit.json'
    require(audit_path.is_file(), 'Actual completed1404 posterior proof missing; no City dense boundary completion')
    posterior = read(audit_path)
    require(posterior['status'] == POSTERIOR_STATUS and posterior['actual_verified_frames'] == 1404
            and posterior['actual_models'] == 2 and posterior['actual_settings_per_model'] == 6
            and posterior['actual_frames_per_setting'] == 117 and posterior['evaluation_run'] == evaluation.name,
            'Posterior proof is not actual complete local1404 protocol')
    require(Path(posterior['source_evaluation_root']).resolve() == evaluation.resolve(), 'Posterior evaluation root differs')
    require(posterior['canonical117_input_npz_sha256'] == CANON_INPUT_SHA
            and posterior['source_adapter_SHA'] == ADAPTER_SHA and posterior['source_core_SHA'] == CORE_SHA
            and posterior['source_config_SHA'] == CONFIG_SHA and posterior['actual_owncoarse_SHA'] == COARSE_SHA
            and posterior['Torch_imported'] is False and posterior['CUDA_called'] is False,
            'Posterior canonical/source/config/owncoarse identity or CPU scope differs')
    protected = {str(audit_path):digest(audit_path)}
    for path, sha in [(POSTERIOR,POSTERIOR_SHA), (audit_directory/'auditor-source.py',POSTERIOR_SHA),
                      (ADAPTER,ADAPTER_SHA), (evaluation/'adapter-source.py',ADAPTER_SHA),
                      (CORE,CORE_SHA), (LOCK/'configuration.json',CONFIG_SHA),
                      (evaluation/'outcome.json',posterior['source_outcome_sha256']),
                      (evaluation/'metrics.json',posterior['source_overall_metrics_sha256']),
                      (evaluation/'protocol.json',posterior['source_protocol_sha256']),
                      (CANON/'protocol.json',posterior['canonical117_protocol_sha256'])]:
        bound_file(path,sha,protected)
    auditor_outcome = read(audit_directory/'outcome.json')
    require(auditor_outcome['status'] == POSTERIOR_STATUS and auditor_outcome['actual1404_completed'] is True
            and auditor_outcome['auditor_source_SHA'] == POSTERIOR_SHA and auditor_outcome['Torch_imported'] is False
            and auditor_outcome['CUDA_called'] is False, 'Actual posterior auditor outcome is not successful1404')
    protected[str(audit_directory/'outcome.json')] = digest(audit_directory/'outcome.json')
    source_outcome = read(evaluation/'outcome.json'); overall = read(evaluation/'metrics.json')
    require(source_outcome['status'] == 'complete117_each_requested_actual_local_pre_post_setting'
            and source_outcome['cuda_used'] is True and source_outcome['driver_sha256'] == ADAPTER_SHA
            and source_outcome['core_evaluator_sha256'] == CORE_SHA, 'Real evaluator source outcome is incomplete/wrong family')
    require(overall['complete_full_leaves_tau1_3_5_6_15_protocol'] is True
            and overall['complete_all_requested_settings'] is True and overall['canonical117_input_scope'] is True
            and overall['source_lock005_configuration_SHA'] == CONFIG_SHA and overall['actual_owncoarse_SHA'] == COARSE_SHA
            and overall['exposure_enabled'] is False, 'Real overall metrics no longer complete actual local1404')
    canonical = read(CANON/'protocol.json'); local = read(evaluation/'protocol.json')
    require(local['status'] == 'actual_completed_own_local_pre_post_model_binding_passed'
            and local['test_names'] == canonical['test_names'] and local['view_count'] == 117
            and local['settings'] == list(SETTINGS) and local['hierarchy_configuration_sha256'] == CONFIG_SHA
            and local['hierarchy_workflow_driver_sha256'] == WORKFLOW_SHA
            and local['canonical117_input_archive_sha256'] == CANON_INPUT_SHA,
            'Actual local evaluation protocol/117 families differ')
    own_inputs = Path(local['independent_input_npz_path'])
    require(own_inputs.resolve() == (evaluation/'independent-inputs/evaluated-inputs.npz').resolve()
            and local['independent_input_npz_sha256'] == CANON_INPUT_SHA, 'Actual evaluator canonical117 archive identity differs')
    bound_file(own_inputs,CANON_INPUT_SHA,protected)
    verified_path = Path(posterior['verified_actual_frames_path'])
    require(verified_path.resolve() == (audit_directory/'verified-actual-frames.json').resolve(), 'Frame ledger outside actual posterior namespace')
    bound_file(verified_path,posterior['verified_actual_frames_sha256'],protected)
    identity_path = Path(posterior['canonical_input_identity_path'])
    require(identity_path.resolve() == (audit_directory/'canonical-input-identity.json').resolve(), 'Canonical identity outside actual posterior namespace')
    bound_file(identity_path,posterior['canonical_input_identity_sha256'],protected)
    identity = read(identity_path)
    require(identity['input_npz_sha256'] == CANON_INPUT_SHA and identity['protocol_sha256'] == posterior['canonical117_protocol_sha256']
            and identity['views'] == 117 and identity['K_C2W_masks_original_files_checked'] is True,
            'Actual canonical117 identity proof differs')
    verified_rows = read(verified_path)
    verified = {(r['model'],r['setting'],r['name']):r for r in verified_rows}
    expected = {(m,s,n) for m in MODELS for s in SETTINGS for n in canonical['test_names']}
    require(len(verified_rows) == len(verified) == 1404 and set(verified) == expected, 'Missing/duplicate/wrong-family posterior frame keys')
    summaries = {(r['model'],r['setting']):r for r in posterior['summaries']}
    require(len(posterior['summaries']) == len(summaries) == 12
            and set(summaries) == {(m,s) for m in MODELS for s in SETTINGS}, 'Posterior setting proofs incomplete')
    for model in MODELS:
        for setting in ('leaves','tau3'):
            folder = evaluation/model/setting; path = folder/'metrics.json'
            bound_file(path,summaries[(model,setting)]['actualmetrics_sha256'],protected)
            metric = read(path); rows = read(folder/'per-image-metrics.json')
            require(metric['status'] == 'complete117_valid_official_continuous_renderer' and metric['model'] == model
                    and metric['setting'] == setting and metric['view_images'] == 117 and metric['per_image'] == rows
                    and [r['name'] for r in rows] == canonical['test_names'], 'Actual selected per-frame metrics detached from posterior setting proof')
            protected[str(folder/'per-image-metrics.json')] = digest(folder/'per-image-metrics.json')
    binding = dict(posterior_completion_path=str(audit_path),posterior_completion_sha256=protected[str(audit_path)],
        posterior_auditor_source_sha256=POSTERIOR_SHA,source_evaluation_root=str(evaluation),
        canonical117_input_npz_sha256=CANON_INPUT_SHA,canonical117_protocol_sha256=posterior['canonical117_protocol_sha256'],
        source_outcome_sha256=posterior['source_outcome_sha256'],source_overall_metrics_sha256=posterior['source_overall_metrics_sha256'],
        source_protocol_sha256=posterior['source_protocol_sha256'],verified_actual_frames_path=str(verified_path),
        verified_actual_frames_sha256=posterior['verified_actual_frames_sha256'],actual_complete1404_proven=True,
        selected_actual_models=list(MODELS),selected_settings=['leaves','tau3'])
    return posterior, verified, protected, binding


def pixel_image(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB', 'Expected original RGB PNG')
        return np.array(image)


def fixed_patch(pred, target, mask, bbox):
    x0,y0,x1,y1 = bbox
    a, g, m = pred[y0:y1,x0:x1], target[y0:y1,x0:x1], mask[y0:y1,x0:x1]
    require(a.size > 0 and g.shape == a.shape and m.shape == a.shape[:2]+(1,), 'Fixed ROI dimensions differ')
    target32 = g.astype(np.float32)/255.
    raw_error = a.astype(np.float64)-g.astype(np.float64)/255.
    main_error = (a*m).astype(np.float64)-((target32*m)*m).astype(np.float64)
    weights = m.astype(np.float64)**2
    valid_denominator = 3*float(weights.sum())
    auxiliary_numerator = float(np.sum(raw_error**2*weights))
    return dict(bbox_xyxy_exclusive=bbox, RGB_components=int(a.size), raw_RGB_MSE=float(np.mean(raw_error**2)),
        official_GTm2_predm_patch_MSE=float(np.mean(main_error**2)), official_patch_denominator=int(a.size),
        symmetric_single_mask_patch_error_numerator=auxiliary_numerator,
        symmetric_single_mask_patch_denominator=valid_denominator,
        symmetric_single_mask_patch_MSE=auxiliary_numerator/valid_denominator if valid_denominator > 0 else None,
        zero_valid_weight_patch=valid_denominator == 0, mask_mean=float(m.mean()), mask_zero_pixels=int((m==0).sum()))


def original_frame(row, folder, j, target, mask, verified, protected):
    filename = f'{j:02d}-{Path(row["name"]).stem}.png'
    target_path = Path(row['target_path'])
    prediction_path = Path(row['prediction_path'])
    raw_pair = Path(row.get('paired_path', folder/'paired'/filename))
    masked_pair = Path(row.get('masked_paired_path', folder/'masked-paired'/filename))
    floating = folder/'float-frames'/(Path(filename).stem+'.npy')
    require(prediction_path.resolve() == (folder/'predicted'/filename).resolve()
            and raw_pair.resolve() == (folder/'paired'/filename).resolve()
            and masked_pair.resolve() == (folder/'masked-paired'/filename).resolve(),
            'Original frame paths outside exact selected family/setting/ordinal')
    require(floating.is_file() and raw_pair.is_file() and masked_pair.is_file(), 'Real original frame products missing')
    hashes = {str(p):digest(p) for p in (target_path,prediction_path,raw_pair,masked_pair,floating)}
    require(hashes[str(target_path)] == row['target_sha256'] and hashes[str(prediction_path)] == row['prediction_sha256'],
            'Existing target/prediction bytes changed')
    if 'paired_sha256' in row:
        require(hashes[str(raw_pair)] == row['paired_sha256'], 'Existing raw pair changed')
    if 'masked_paired_sha256' in row:
        require(hashes[str(masked_pair)] == row['masked_paired_sha256'], 'Existing masked pair changed')
    if verified is not None:
        for key, path in [('float_frame_sha256',floating), ('PNG_sha256',prediction_path),
                          ('raw_pair_sha256',raw_pair), ('masked_pair_sha256',masked_pair)]:
            require(hashes[str(path)] == verified[key], 'Actual posterior frame proof differs: '+key)
    pred = np.load(floating,allow_pickle=False)
    require(pred.dtype == np.float32 and pred.shape == (172,256,3) and np.isfinite(pred).all()
            and np.all((pred>=0)&(pred<=1)), 'Original float32 RGB frame invalid')
    gt8, png, paired8, masked8 = map(pixel_image,(target_path,prediction_path,raw_pair,masked_pair))
    require(np.array_equal(gt8,target), 'Actual target differs from frozen117 RGB')
    require(np.array_equal((pred*255).astype(np.uint8),png), 'Predicted PNG is not original float quantization')
    require(paired8.shape==(172,512,3) and np.array_equal(paired8[:,:256],target)
            and np.array_equal(paired8[:,256:],png), 'Raw original pair panel order differs')
    gt32=target.astype(np.float32)/255.
    masked_gt=((gt32*mask)*mask*255).astype(np.uint8)
    masked_pred=(pred*mask*255).astype(np.uint8)
    require(masked8.shape==(172,512,3) and np.array_equal(masked8[:,:256],masked_gt)
            and np.array_equal(masked8[:,256:],masked_pred), 'Original masked GTm2/predm pair differs')
    protected.update(hashes)
    return pred, dict(original_raw_pair=str(raw_pair),original_raw_pair_sha256=hashes[str(raw_pair)],
        original_masked_pair=str(masked_pair),original_masked_pair_sha256=hashes[str(masked_pair)],
        original_float_RGB=str(floating),original_float_RGB_sha256=hashes[str(floating)],
        original_predicted_RGB=str(prediction_path),original_predicted_RGB_sha256=hashes[str(prediction_path)],
        paired_layout='Original256x172 GT left; original predictedRGB right; total512x172; no image edits.', ordinal=j)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['coarse-reference','dense'])
    parser.add_argument('--run',required=True,type=identifier)
    parser.add_argument('--evaluation-run',type=identifier)
    parser.add_argument('--posterior-audit-run',type=identifier)
    args=parser.parse_args()
    output=EVALS/args.run
    output.mkdir(parents=True,exist_ok=False)
    start=datetime.now(timezone.utc).isoformat()
    write(output/'outcome.json',dict(status='running_CPU_boundary_index',started_utc=start,mode=args.mode,CUDA_initialized=False))
    shutil.copyfile(Path(__file__),output/'source-snapshot.py')
    try:
        posterior=None; completion_binding=None
        if args.mode=='dense':
            require(args.evaluation_run and args.posterior_audit_run,'Dense mode requires actual evaluation and posterior namespaces')
            evaluation=EVALS/args.evaluation_run
            audit_path=EVALS/args.posterior_audit_run/'completion-audit.json'
            posterior,verified,protected,completion_binding=dense_completion_binding(evaluation,audit_path.parent)
            cases=[(m,s,evaluation/m/s) for m in MODELS for s in ('leaves','tau3')]
        else:
            require(not args.evaluation_run and not args.posterior_audit_run,'Coarse reference is an explicitly separate read-only scope')
            protected={}
            cases=[('own-coarse-reference','flat',CANON/'flat')]
            verified={}
        protocol_path=CANON/'protocol.json'
        protocol=read(protocol_path)
        arrays_path=Path(protocol['evaluated_input_npz_path'])
        require(digest(arrays_path)==CANON_INPUT_SHA==protocol['evaluated_input_npz_sha256'],'Frozen117 arrays changed')
        require(digest(REVIEW)==REVIEW_SHA,'Independent geometry002 review changed')
        selection_path=SELECTION/'selection.json'
        selection_report=read(SELECTION/'report.json')
        require(selection_report['status']=='completed_actual_CPU_geometry_only_selection'
                and digest(selection_path)==selection_report['selection_sha256']==SELECTION_SHA,
                'Frozen geometry-only views/tiles changed')
        selection=read(selection_path)
        require(len(selection['selected_names'])==len(set(selection['selected_names']))==8
                and [r['name'] for r in selection['selected_views']]==selection['selected_names'],
                'Require fixed eight geometry-selected views in exact order')
        protected.update({str(p):digest(p) for p in (Path(__file__),protocol_path,arrays_path,REVIEW,selection_path,SELECTION/'report.json')})
        with np.load(arrays_path,allow_pickle=False) as z:
            names,targets,masks,K,c2w=(z[k].copy() for k in ('names','rgb','mask','K','c2w'))
        require(names.tolist()==protocol['test_names'] and len(names)==117,'Original canonical117 identity differs')
        ordinal={n:j for j,n in enumerate(names.tolist())}
        metadata={}
        for model,setting,folder in cases:
            path=CANON/'per-image-metrics.json' if args.mode=='coarse-reference' else folder/'per-image-metrics.json'
            rows=read(path)
            require([r['name'] for r in rows]==names.tolist(),'All117 original per-image order differs')
            protected[str(path)]=digest(path)
            metadata[(model,setting)]={r['name']:r for r in rows}
        records=[]
        for view in selection['selected_views']:
            name=view['name'];j=ordinal[name]
            require(np.array_equal(K[j],np.array(view['K'])) and np.array_equal(c2w[j],np.array(view['C2W'])),
                    'Fixed projected boundary geometry differs from canonical camera')
            predictions={};images={};diagnostics=[]
            for model,setting,folder in cases:
                row=metadata[(model,setting)][name]
                proof=verified[(model,setting,name)] if args.mode=='dense' else None
                if proof is not None:
                    require(proof['model']==model and proof['setting']==setting and proof['name']==name
                            and proof['independent_PSNR_exact'] is True and proof['fresh_topology_coverage'] is True,
                            'Selected frame is not the actual posterior verified dense family')
                pred,paths=original_frame(row,folder,j,targets[j],masks[j],proof,protected)
                label=model+'/'+setting;predictions[label]=pred;images[label]=paths
            for tile in view['dense_geometry_only_tiles']:
                bbox=tile['bbox_xyxy_exclusive']
                values={label:fixed_patch(pred,targets[j],masks[j],bbox) for label,pred in predictions.items()}
                diagnostics.append(dict(geometry_only_tile=tile,RGB_patch_diagnostics=values,
                    interpretation='Preselected sparse-projection proxy patch; not a seam or surface GT mask, not causal ownership/post evidence.'))
            leaf_comparison=None
            if args.mode=='dense':
                a=predictions['merged-unoptimized/leaves'];b=predictions['merged-post/leaves']
                leaf_comparison=dict(actual_RGB_bitwise_equal=bool(np.array_equal(a,b)),
                    actual_max_abs_RGB_difference=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
            records.append(dict(name=name,ordinal=j,geometry_selected_before_dense_results=True,
                sparse_boundary_proxy_count=view['unique_boundary_observed_point_count'],original_images=images,
                fixed_patch_diagnostics=diagnostics,actual_pre_post_leaf_RGB=leaf_comparison))
        require(all(digest(path)==value for path,value in protected.items()),'Original source or image changed during read-only CPU audit')
        require('torch' not in sys.modules,'CPU boundary index must not import torch')
        report=dict(status='completed_actual_coarse_reference_boundary_image_index_only' if args.mode=='coarse-reference'
                     else 'completed_actual_city_geometry_selected_original_boundary_image_index',
            started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),mode=args.mode,
            selected_names=selection['selected_names'],views=records,view_count=8,
            actual_original_frame_sets=8*len(cases),fixed_patch_count=sum(len(r['fixed_patch_diagnostics']) for r in records),
            protected_original_sources=protected,all_original_sources_unchanged=True,CUDA_initialized=False,torch_imported=False,
            images_edited_or_generated=False,actual_visual_inspection_still_required=True,
            source_posterior_complete1404_proof=None if posterior is None else str(audit_path),
            actual_dense_completion_binding=completion_binding,
            City_dense_boundary_results_complete=args.mode=='dense',City_all_seams_eliminated=False,
            limitations=['Coarse reference is one globally trained SH1 scaffold and does not substitute actual City dense pre/post models or the1404 protocol.',
                'Eight views and fixed32px tiles were selected by sparse geometry alone. They are an observation-biased bounded inspection, not all seams or dense visibility.',
                'Masks can retain foreground dynamics. RGB error includes color, geometry, alpha, antialiasing, occlusion and LOD approximations; no causal seam conclusion.',
                'RawMSE, source-order GTm2/predm full-patch MSE and symmetric m2-weighted auxiliary MSE have different denominators. Never silently pool them.',
                'This report verifies original file identity and patch arithmetic. A person/agent must inspect the full paired originals before any visual conclusion.'])
        write(output/'report.json',report)
        write(output/'outcome.json',dict(status=report['status'],exit_code=0,report_sha256=digest(output/'report.json'),
            completed_utc=report['completed_utc'],CUDA_initialized=False,City_dense_boundary_results_complete=args.mode=='dense'))
        print(json.dumps(dict(status=report['status'],original_frame_sets=report['actual_original_frame_sets'],fixed_patches=report['fixed_patch_count'])))
    except Exception as error:
        write(output/'outcome.json',dict(status='failed_preserved_CPU_boundary_index',exit_code=1,mode=args.mode,
            error=str(error),ended_utc=datetime.now(timezone.utc).isoformat(),CUDA_initialized=False))
        raise


if __name__=='__main__':
    main()
