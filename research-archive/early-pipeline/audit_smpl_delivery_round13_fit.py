"""Independently replay round-thirteen pose guards and compare against R12 and R10."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_smpl_delivery_round12_fit as previous
import launch_smpl_delivery_round13 as launch


prior = previous.prior
BASELINE = previous.BASELINE
OUTPUT = previous.OUTPUT.parent.parent / 'smpl-delivery-round13/input-audit'
R12_AUDIT_SHA = {
    '016': '3b10b5c6f0528f57aa557bcc00afee84c55779daa3003947f6fc2d2059d352ce',
    '006': 'c822731402cf6ba212e0aba8baf19f3c79f73f52b252b47564c7700b039f9b29',
}


def verify(subject, run):
    contract = prior.read_json(prior.pin(BASELINE, prior.BASELINE_SHA))
    baseline = contract['subjects'][subject]
    baseline_run = Path(baseline['run'])
    fit = prior.read_json(run/'fit/official-fit-result.json')
    manifest, rows, inventory = prior.verify_reuse(subject,run,baseline,fit)
    prior.pin(inventory,launch.R12_SOURCE_SHA[subject])
    plan = prior.read_json(run.parent/'preregistration.json')
    prior.require(plan['schema'] == launch.SCHEMA
                  and plan['code_parent_commit'] == launch.CODE_PARENT
                  and plan['baseline_commit'] == launch.BASE, 'Not the registered round-thirteen fit.')
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
    previous.verify_face_report(joint,records)
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
    r12_path = previous.OUTPUT/f'{subject}-face-and-coverage.json'
    r12 = prior.read_json(prior.pin(r12_path,R12_AUDIT_SHA[subject]))
    prior.require(r12['candidate_commit'] == launch.CODE_PARENT
                  and r12['source_sha256'] == launch.R12_SOURCE_SHA[subject]
                  and r12['retained_holdouts'] == holdouts,
                  'Round-twelve solver comparator changed.')
    r12_joint_path = launch.R12_DELIVERY.parent/subject/'evidence/joint/joint-body-refinement.json'
    r12_joint = prior.read_json(prior.pin(r12_joint_path,r12['joint_sha256']))
    for fid, record in records.items():
        for region, item in record['regions'].items():
            r10_region = baseline['frames'][fid]['regions'][region]
            r12_region = r12['frames'][fid]['regions'][region]
            prior.require(item['before'] == r10_region['before'] == r12_region['before'],
                          f'Independent initializer changed in {fid}/{region}.')
            item['r10_after'] = r10_region['after']
            item['r12_after'] = r12_region['after']
    after = set(after_coverage['front']['occupied_source_bins'])
    comparison_coverage = dict(initializer=before_coverage,r10=baseline['coverage'],r12=r12['coverage']['r12'])
    coverage_changes = {}
    for label, coverage in comparison_coverage.items():
        before_bins = set(coverage['front']['occupied_source_bins'])
        coverage_changes[label] = dict(recovered=sorted(after-before_bins),lost=sorted(before_bins-after))
    for region, row in summary['regions'].items():
        for old in ('r10_after','r12_after'):
            values = [frame['regions'][region][old]['rms_px'] for frame in records.values()
                      if frame['regions'][region]['before']['points']]
            row[f'median_{old}_px'] = float(np.median(values)) if values else None
    shared_basis_delta = float(np.linalg.norm(np.asarray(joint['shared_latent_final'],dtype=np.float64)
                                             - np.asarray(r12_joint['shared_latent_final'],dtype=np.float64)))
    return dict(state='verified',subject=subject,run=str(run),candidate_commit=plan['candidate_commit'],
        source_reuse_verified=True,candidate_frames=len(names),retained_holdouts=holdouts,
        source_sha256=prior.sha(inventory),joint_sha256=prior.sha(joint_path),
        candidate_sha256=prior.sha(candidate_path),training_fit_sha256=training_hashes,
        native_projection_replay_max_absolute_error_px=parity,
        selected_frames=selection['selected_frames'],reference_frame=selection['reference_frame'],
        pose_screen=selection['pose_screen'],coverage_basis=selection['coverage_basis'],
        coverage=dict(comparison_coverage,r13=after_coverage),coverage_changes=coverage_changes,
        comparison_selected_frames=dict(r12=r12['selected_frames'],
            added_vs_r12=[fid for fid in selection['selected_frames'] if fid not in r12['selected_frames']],
            dropped_vs_r12=[fid for fid in r12['selected_frames'] if fid not in selection['selected_frames']]),
        comparison_objective=dict(r12_initial=r12_joint['objective_initial'],r12_final=r12_joint['objective_final'],
            r13_initial=joint['objective_initial'],r13_final=joint['objective_final']),
        shared_basis_l2_change_from_r12=shared_basis_delta,
        face_guard=summary,block_solver=blocks,latent_basis=previous.verify_latent_basis(joint,candidates),
        detector_support=prior.input_detector_support(run,candidates,names),
        changed_pose_parameter_frames=sum(item['pose_parameters_changed'] for item in records.values()),
        changed_camera_vertex_frames=sum(item['camera_vertices_changed'] for item in records.values()),
        measurement_boundary=contract['measurement_boundary'],frames=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=tuple(launch.R12_SOURCE_SHA))
    args = parser.parse_args()
    run = launch.root_for(args.subject)/'run'
    result = verify(args.subject,run)
    destination = OUTPUT/f'{args.subject}-face-and-coverage.json'
    prior.write_json(destination,result)
    print(json.dumps(dict(state='verified',subject=args.subject,path=str(destination),
                          sha256=prior.sha(destination)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
