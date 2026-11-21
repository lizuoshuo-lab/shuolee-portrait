"""Independently replay native-pixel guards and training inputs from the round-twelve fit."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_smpl_delivery_round11_face as prior
import launch_smpl_delivery_round12 as launch


OUTPUT = prior.OUTPUT.parent.parent / 'smpl-delivery-round12/input-audit'
BASELINE = prior.OUTPUT / 'baseline-face-and-inputs.json'


def verify_face_report(joint, records):
    guard = joint['face_guard']
    prior.require(guard['reference'] == 'official_initializer'
                  and guard['regions'] == prior.FACE_REGIONS
                  and guard['threshold_px'] == prior.FACE_TOLERANCE_PX
                  and guard['confidence_threshold'] == prior.FACE_CONFIDENCE
                  and guard['exact_reprojection_passed'] is True,
                  'Face protection definition or replay status changed.')
    supported = 0
    for frame, record in records.items():
        original = joint['frames'][frame]['face_guard']
        prior.require(set(original) == set(prior.FACE_REGIONS), f'Missing region report for {frame}.')
        supported += any(item['before']['points'] for item in record['regions'].values())
        for region, item in record['regions'].items():
            actual = original[region]
            count = item['before']['points']
            prior.require(actual['supported_points'] == count and actual['status'] == ('pass' if count else 'unobserved'),
                          f'Support or status differs in {frame}/{region}.')
            if not count:
                continue
            prior.require(not actual['failures'] and not record['failures'], f'Face guard failed in {frame}/{region}.')
            for reported, calculated in (('rms_px','rms_px'), ('peak_px','max_px')):
                for phase in ('before','after'):
                    prior.require(abs(actual[phase][reported]-item[phase][calculated]) < prior.SCORE_NUMERICAL_TOLERANCE_PX,
                                  f'Independent projection differs in {frame}/{region}/{phase}/{reported}.')
    prior.require(supported == guard['supported_frame_count'], 'Face support count differs.')


def verify_latent_basis(joint, candidates):
    latent = candidates['pose_embedding'].astype(np.float64).reshape(-1,32)
    residual = latent - np.asarray(joint['shared_latent_final'],dtype=np.float64).reshape(1,32)
    mean_error = float(np.abs(residual.mean(axis=0)).max())
    maximum = float(np.linalg.norm(residual,axis=1).max())
    prior.require(mean_error < 1e-5 and abs(maximum-joint['pose_residual_l2_max']) < 1e-5
                  and maximum <= joint['config']['latent_residual_l2_limit'] + 1e-5,
                  'Output latent rows disagree with shared basis or residual bound.')
    return dict(centered_residual_mean_max=mean_error, centered_residual_l2_max=maximum)


def verify(subject, run):
    contract = prior.read_json(prior.pin(BASELINE, prior.BASELINE_SHA))
    baseline = contract['subjects'][subject]
    baseline_run = Path(baseline['run'])
    fit = prior.read_json(run/'fit/official-fit-result.json')
    manifest, rows, inventory = prior.verify_reuse(subject,run,baseline,fit)
    prior.pin(inventory,launch.R11_SOURCE_SHA[subject])
    plan = prior.read_json(run.parent/'preregistration.json')
    prior.require(plan['schema'] == 'shuolee-round12-constrained-fit-v1'
                  and plan['code_parent_commit'] == launch.CODE_PARENT
                  and plan['baseline_commit'] == launch.BASE, 'Not the registered round-twelve fit.')
    proof = fit['joint_body_refinement']
    joint_path = prior.pin(run/proof['file'],proof['sha256'])
    candidate_path = prior.pin(run/proof['candidate_file'],proof['candidate_sha256'])
    before_path = prior.pin(run/proof['before_selection_file'],proof['before_selection_sha256'])
    prior.pin(before_path,baseline['file_sha256']['fit/selection-before-joint.json'])
    joint = prior.read_json(joint_path)
    prior.require(joint['state'] == 'complete' and joint['native_K_frozen'] and joint['shared_betas_frozen']
                  and joint['per_frame_scale_parameters'] == 0, 'Pose solver changed frozen inputs.')
    before, selection = prior.read_json(before_path), prior.read_json(run/'fit/selection.json')
    holdouts = baseline['retained_holdouts']
    prior.require(selection == fit['selection']
                  and before['retained_holdouts'] == selection['retained_holdouts'] == joint['retained_holdouts']
                  == joint['excluded_holdouts'] == holdouts and set(selection['selected_frames']).isdisjoint(holdouts),
                  'Wrong held-out identities or actual selected frames.')
    before_coverage, after_coverage = map(prior.coverage_verified,(before,selection))
    candidates = prior.read_npz(candidate_path)
    names = prior.fixed_fit_names(candidates,manifest,holdouts)
    prior.require(names == joint['fit_frame_ids'] == baseline['candidate_frames']
                  and set(names) == set(joint['frames']), 'A candidate or observation was omitted.')
    old = prior.read_npz(prior.pin(baseline_run/'fit/joint-candidates.npz',
                                   baseline['file_sha256']['fit/joint-candidates.npz']))
    for field in ('frame_ids','faces','betas','K_integer_centers','observed_native_keypoints',
                  'source_frames','image_sha256','mask_sha256'):
        prior.require(np.array_equal(candidates[field],old[field]),f'Fixed source array differs: {field}.')
    records, parity = prior.frame_comparisons(names,rows,baseline_run,candidates,selection,measure_pose_change=True)
    summary = prior.summarize_faces(records)
    prior.require(not summary['failed_frames'], f'Native facial protection failures: {summary["failed_frames"]}.')
    verify_face_report(joint,records)
    blocks = launch.verify_solver_audit(joint,names)
    exact_body = blocks['exact_body_screen']
    passed = lambda decision: decision['status'] in prior.GATE_PASSED
    original_eligible = [fid for fid in names if passed(before['reasons'][fid])]
    refined_eligible = [fid for fid in names if passed(selection['reasons'][fid])]
    prior.require(exact_body['initial_eligible'] == original_eligible
                  and exact_body['final_eligible'] == refined_eligible
                  and set(original_eligible) <= set(refined_eligible),
                  'Exact body screen differs from the unchanged native selector.')
    prior.require(all(record['pose_parameters_changed'] == blocks['per_frame'][fid]['changed']
                      for fid,record in records.items()),
                  'A saved pose change differs from the accepted local/shared state.')
    training_hashes = prior.verify_final_geometry(run,selection,names,candidates,rows)
    r11 = prior.read_json(prior.pin(prior.OUTPUT/f'{subject}-face-and-coverage.json',{
        '016':'03945608160f67c4d5bfb1bec87be496b135fdbe20acd7ece0482a4a175491ea',
        '006':'e028d586aa4997a5dc73e4f221cd7a1a8b14008ab6a678aa35d4a5e3e27339e0',
    }[subject]))
    for fid, record in records.items():
        for region, item in record['regions'].items():
            r10_region = baseline['frames'][fid]['regions'][region]
            r11_region = r11['frames'][fid]['regions'][region]
            prior.require(item['before'] == r10_region['before'] == r11_region['before'],
                          f'Independent initializer changed in {fid}/{region}.')
            item['r10_after'] = r10_region['after']
            item['r11_after'] = r11_region['after']
    after = set(after_coverage['front']['occupied_source_bins'])
    comparison_coverage = dict(initializer=before_coverage,r10=baseline['coverage'],r11=r11['r11_coverage'])
    coverage_changes = {}
    for label, coverage in comparison_coverage.items():
        previous = set(coverage['front']['occupied_source_bins'])
        coverage_changes[label] = dict(recovered=sorted(after-previous),lost=sorted(previous-after))
    return dict(state='verified',subject=subject,run=str(run),candidate_commit=plan['candidate_commit'],
        source_reuse_verified=True,candidate_frames=len(names),retained_holdouts=holdouts,
        source_sha256=prior.sha(inventory),joint_sha256=prior.sha(joint_path),
        candidate_sha256=prior.sha(candidate_path),training_fit_sha256=training_hashes,
        native_projection_replay_max_absolute_error_px=parity,
        selected_frames=selection['selected_frames'],reference_frame=selection['reference_frame'],
        pose_screen=selection['pose_screen'],coverage_basis=selection['coverage_basis'],
        coverage=dict(comparison_coverage,r12=after_coverage),coverage_changes=coverage_changes,
        face_guard=summary,block_solver=blocks,latent_basis=verify_latent_basis(joint,candidates),
        detector_support=prior.input_detector_support(run,candidates,names),
        changed_pose_parameter_frames=sum(item['pose_parameters_changed'] for item in records.values()),
        changed_camera_vertex_frames=sum(item['camera_vertices_changed'] for item in records.values()),
        measurement_boundary=contract['measurement_boundary'],frames=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=tuple(launch.R11_SOURCE_SHA))
    args = parser.parse_args()
    run = launch.root_for(args.subject)/'run'
    result = verify(args.subject,run)
    destination = OUTPUT/f'{args.subject}-face-and-coverage.json'
    prior.write_json(destination,result)
    print(json.dumps(dict(state='verified',subject=args.subject,path=str(destination),
                          sha256=prior.sha(destination)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
