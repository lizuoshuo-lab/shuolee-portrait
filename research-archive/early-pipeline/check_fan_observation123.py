"""Independent, model-free FAN transport/completeness acceptance helper.

This does not modify the registered implementation or old RTMW review. New
loader fixtures are explicitly unreviewed/inactive and grant no fitting access.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
DEFAULT_OUTPUT = ROOT/'smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis/fan-observation'
sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import semantic_face as face
from shuolee_portrait.alignment_inputs import load_alignment_inputs


def intake(folder, archive, key, frames, metadata, *, purpose, original_review=None):
    folder.mkdir()
    # No camera or body data, and no promotion of any prediction to reviewed.
    roster = [dict(frame=r['frame'], wh=r['wh'], source_sha256=r['source_sha256']) for r in frames]
    face.save(folder/'frames.json', dict(records=roster))
    with np.load(archive, allow_pickle=False) as z:
        values = z[key].copy()
    old = {(r['frame'], r['ibug68_index']):r for r in (original_review or [])}
    records = []
    for i, row in enumerate(frames):
        for ibug in range(17,68):
            frozen = old.get((row['frame'],ibug))
            record = dict(frame=row['frame'], anchor_id=f'ibug68:{ibug}', ibug68_index=ibug,
                uv_native=values[i,ibug].tolist(), visibility='unknown',
                split=frozen['split'] if frozen else ('HELD' if ibug%5==0 else 'FIT'),
                physical_family_id=f'unverified_semantic_id:ibug68:{ibug}',
                semantic_identity_verified=False, localization_sigma_px=None, reliability=None,
                visibility_source='unreviewed_transport_check_only', review_evidence=None,
                coordinate_review_status='New prediction has not been anatomically localized or physically measured.')
            if frozen:
                record['immutable_prior_semantic_review'] = dict(visibility=frozen['visibility'],
                    split=frozen['split'], semantic_identity_verified=frozen['semantic_identity_verified'],
                    prior_uv_native=frozen['uv_native'], applied_as_new_prediction_review=False)
            records.append(record)
    face_desc = dict(face.descriptor(archive), convention='ibug68', coordinates_key=key,
                     coordinate_domain=face.PIXELS, metadata=metadata)
    frames_desc = face.descriptor(folder/'frames.json')
    face.save(folder/'unreviewed-transport-inventory.json', dict(
        observations_sha256=face_desc['sha256'], inputs_sha256=frames_desc['sha256'],
        coordinate_domain=face.PIXELS, synthetic_control=purpose=='synthetic_contract_test',
        inventory_role='Read-only transport compatibility only; does not revise frozen semantic review.', records=records))
    manifest = dict(schema='shuolee-alignment-input/1', purpose=purpose, camera_model='rectified_pinhole',
        frames=frames_desc, face=face_desc,
        reviews=dict(face.descriptor(folder/'unreviewed-transport-inventory.json'), format='reviewed_anatomical_v1'))
    path = folder/'alignment-inputs.json'; face.save(path, manifest)
    bundle = load_alignment_inputs(path)
    face.need(np.array_equal(bundle.face_uv, values[:,17:68]), 'Actual loader changed native FAN coordinates.')
    face.need(not bundle.eligible.any() and not bundle.report['fitting_allowed'] and
              bundle.report['active_fit']==bundle.report['active_held']==0, 'Transport fixture unexpectedly activates points.')
    face.need(all(r['localization_sigma_px'] is None and r['reliability'] is None for r in bundle.records),
              'Missing uncertainty/reliability was fabricated.')
    bundle.assert_unchanged()
    return dict(state='exact_native_IBUG68_static51_transport_passed', source_key=key,
        frames=len(frames), exact_17_to_67_values=True, nonidentity_crop_metadata_not_reapplied=True,
        active_fit=0, active_held=0, no_uncertainty_or_reliability_fabricated=True,
        prior_RTMw_coordinates_not_reused_as_FAN_review=True, fitting_allowed=False,
        manifest=face.descriptor(path), loader_source=face.descriptor(PROJECT/'src/shuolee_portrait/alignment_inputs.py'))


def synthetic_check(output):
    face.need(not (output/'native-adapter-synthetic-check.json').exists(), 'Synthetic check already recorded.')
    with tempfile.TemporaryDirectory(prefix='fan-native-adapter-') as temp:
        root = Path(temp)
        frames = [dict(frame=f'synthetic-{i}', wh=[640,480], source_sha256=str(i+1)*64) for i in range(2)]
        metadata = [dict(r, crop_center=[311.7,203.6], crop_scale=.74,
            nonidentity_grid_to_native=[[2.3125,0,237.7],[0,2.3125,129.6],[0,0,1]]) for r in frames]
        face.save(root/'metadata.json', dict(rows=metadata))
        values = np.arange(2*68*2, dtype=np.float64).reshape(2,68,2)*.25+23.375
        np.savez(root/'face.npz', native=values, frame_ids=[r['frame'] for r in frames])
        report = intake(root/'intake', root/'face.npz', 'native', frames,
            dict(face.descriptor(root/'metadata.json'), rows_key='rows'), purpose='synthetic_contract_test')
        manifest_path=root/'intake/alignment-inputs.json'
        manifest=face.read(manifest_path)
        review_path=root/'intake/unreviewed-transport-inventory.json'
        wrong=face.read(review_path); wrong['records'][0]['uv_native'][0] += 1.
        face.save(root/'wrong-review.json',wrong)
        manifest['reviews']=dict(face.descriptor(root/'wrong-review.json'),format='reviewed_anatomical_v1')
        face.save(root/'wrong-manifest.json',manifest)
        try:
            load_alignment_inputs(root/'wrong-manifest.json')
        except ValueError as exc:
            face.need('Reviewed coordinate differs' in str(exc), 'Wrong rejection reason: '+str(exc))
        else:
            raise ValueError('Loader accepted coordinates from a different review target.')
        report.pop('manifest') # Temporary fixture does not masquerade as persistent evidence.
        report.update(synthetic_only=True, mismatched_review_coordinate_rejected=True,
            model_loads=0, network_forwards=0, optimizer_steps=0, gaussian_steps=0,
            helper=face.descriptor(Path(__file__)), time_utc=datetime.now(timezone.utc).isoformat())
        face.save(output/'native-adapter-synthetic-check.json',report)
        print(report)


def actual_check(output):
    result=face.read(output/'result.json')
    face.need(result['state']=='observations_complete_native_review_pending' and
              result['network_forwards_completed']==result['network_forward_attempts']==6,
              'Expected completed single registered six-forward pass.')
    prepared=face.read(output/'prepare.json')
    receipt=face.read(face.verify(prepared['implementation_receipt']))
    manifest=face.read(face.verify(receipt['execution_manifest']))
    for item in receipt['code']+receipt['frozen_sources']+[receipt['frozen_review_snapshot']]:
        face.verify(item)
    archive=face.verify(result['observations'])
    metadata_path=output/'observations.json'
    metadata=face.read(metadata_path)
    review=face.read(manifest['review']['path'])
    with np.load(archive,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    n=len(manifest['frames'])
    face.need(arrays['frame_ids'].tolist()==[r['frame'] for r in manifest['frames']], 'Frame order mismatch.')
    face.need(arrays['source_indices'].tolist()==[r['source_frame_index'] for r in manifest['frames']], 'Source index mismatch.')
    face.need(np.array_equal(arrays['flip_permutation'],face.FLIP_PERMUTATION),'Flip permutation mismatch.')
    for key in ('heatmap_original','heatmap_flip_raw','heatmap_flip_restored','heatmap_sum'):
        face.need(arrays[key].shape==(n,68,64,64) and np.isfinite(arrays[key]).all(), 'Invalid heatmap archive: '+key)
    for branch in ('original','flip_restored','sum'):
        for key in ('continuous','official_integer','points64'):
            value=arrays[branch+'_'+key]
            face.need(value.shape==(n,68,2) and np.isfinite(value).all(), 'Invalid saved branch array.')
        face.need(arrays[branch+'_scores'].shape==(n,68) and np.isfinite(arrays[branch+'_scores']).all(),'Invalid branch scores.')
    sampler_rows=[]
    for i,row in enumerate(manifest['frames']):
        face.need(np.array_equal(face.restore_flip(arrays['heatmap_flip_raw'][i:i+1])[0],arrays['heatmap_flip_restored'][i]),'Flip restoration changed.')
        face.need(np.array_equal(arrays['heatmap_original'][i]+arrays['heatmap_flip_restored'][i],arrays['heatmap_sum'][i]),'Heatmap sum changed.')
        for branch in ('original','flip_restored','sum'):
            native=face.continuous_inverse(arrays[branch+'_points64'][i],row['crop_center'],row['crop_scale'])
            face.need(np.array_equal(native,arrays[branch+'_continuous'][i]),'Continuous inverse was changed.')
            face.need(np.array_equal(np.trunc(native).astype(np.int32),arrays[branch+'_official_integer'][i]),'Official integer parity failed.')
        for branch,key in [('original','heatmap_original'),('flip','heatmap_flip_raw')]:
            raw=np.load(output/f"slot{row['slot']:06d}-{branch}-raw-heatmap.npy",allow_pickle=False)
            face.need(np.array_equal(raw[0],arrays[key][i]),'Independently saved raw branch differs.')
        provenance=metadata['rows'][i]['crop_provenance']
        crop=provenance['actual_integer_crop']; w,h=crop['pre_resize_wh']; ul=crop['upper_left']
        sampler=[[w/256.,0,ul[0]+.5*w/256.-.5],[0,h/256.,ul[1]+.5*h/256.-.5],[0,0,1]]
        face.need(np.allclose(sampler,crop['network_integer_center_to_source_sampling_affine'],atol=1e-12,rtol=0),'Sampling-grid matrix differs.')
        sampler_rows.append(dict(frame=row['frame'],network_integer_center_to_source_sampling_affine=sampler,
            official_ideal_heatmap64_to_native=provenance['ideal_affine']['64']['grid_to_native'],
            hm_to_network_pixel_convention='Not independently physically calibrated; no extra composition or coordinate correction applied.',
            edge_padding_and_resize=crop))
    root=output/'adapter-validation'
    face.need(not root.exists(),'Read-only adapter evidence already exists; inspect rather than overwrite.')
    root.mkdir()
    intakes=[]
    for key in ('original_continuous','flip_restored_continuous','sum_continuous'):
        intakes.append(intake(root/key,archive,key,manifest['frames'],
            dict(face.descriptor(metadata_path),rows_key='rows'),purpose='real',original_review=review['records']))
    face.need(face.sha(manifest['review']['path'])==manifest['review']['sha256'] and
              face.sha(output/'frozen-review-snapshot.json')==manifest['review']['sha256'],'Frozen review changed.')
    face.save(output/'sampling-affine-review.json',dict(rows=sampler_rows,
        ideal_inverse_is_exact_raster_inverse=False,physical_landmark_accuracy_established=False))
    checks=dict(state='saved_output_and_native_adapter_checks_passed',
        time_utc=datetime.now(timezone.utc).isoformat(),frames=n,independent_raw_heatmaps_verified=2*n,
        native_adapter_intake=intakes,all_new_FAN_transport_records_inactive=True,
        prior_semantic_visibility_identity_split_preserved_in_original_snapshot=True,
        no_prior_RTMw_target_replacement=True,source_hashes_unchanged=True,
        actual_runtime_versions=face.read(output/'runtime-environment.json'),
        helper=face.descriptor(Path(__file__)),new_model_loads=0,new_network_forwards=0,
        optimizer_steps=0,gaussian_steps=0,quality_accepted=False,
        still_unverified=['Human/native-image semantic localization review of these new outputs.',
            'Physical landmark accuracy and calibrated uncertainty.',
            'Independent physical calibration of heatmap to network raster centers.',
            'Generalization beyond the registered three frames; 94-frame expansion is not admitted.'])
    face.save(output/'generic-checks.json',checks)
    print(checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('synthetic','check'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    with face.offline_only():
        synthetic_check(args.output) if args.action=='synthetic' else actual_check(args.output)
