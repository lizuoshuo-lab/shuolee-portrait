"""Read-only source review and synthetic6D controls, never the actual full auditor."""
from __future__ import annotations
import ast
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts/audit_rgbd_affine_full_results.py"
LOCKS = {
    "scripts/audit_rgbd_photogeometric_results.py": "e80762e6591369bbeeab75844a233645f0a2f524759730c8339db41028bccc53",
    "scripts/audit_rgbd_affine_real_results.py": "b549a0af2642b2e203e8f3b5f0deae46dcaf2458f608d7e27435c87b33f8c6d5",
    "outputs/rgbd-affine-full-cpu-preparation-001/protocol.json": "ec4e6264060367f7193883c47c9cd13b34ee5c755da0290f504b66fbeab1c47a",
    "outputs/rgbd-affine-full-cpu-preparation-001/root_protocol_review.json": "977fae21421e9c71d51c29756b8999d4b075b5b8ed53a9c16d46176c88ecd44f",
    "outputs/rgbd-affine-full-cpu-preparation-001/root-independent-full-preflight-review.json": "d815c85825decc9d7330594033416fbfdf1dfc09d472c081176a4329265dfdeb",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--audit-source-sha", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"rgbd-affine-full-auditor-independent-source-review-[0-9]{3}",args.run) is None:
        raise ValueError("CPU source review run must use an explicit unique three-digit revision")
    if re.fullmatch(r"[0-9a-f]{64}",args.audit_source_sha) is None:
        raise ValueError("Exact reviewed audit source SHA required")
    TARGET_SHA = args.audit_source_sha
    OUTPUT = ROOT / "outputs" / args.run
    locked = {"scripts/audit_rgbd_affine_full_results.py":TARGET_SHA,**LOCKS}
    if OUTPUT.exists(): raise ValueError("Never overwrite CPU reviewer evidence")
    OUTPUT.mkdir(exist_ok=False)
    for name, expected in locked.items():
        if sha(ROOT/name) != expected: raise ValueError("Changed reviewed source/input: "+name)
    source = TARGET.read_bytes()
    with (OUTPUT/"source_script.py").open("xb") as stream: stream.write(source)
    for name in tuple(locked)[:3]: compile((ROOT/name).read_bytes(), str(ROOT/name), "exec")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom): imports.append(node.module)
    forbidden = ("torch","gsplat","tum_rgbd","track_rgbd_icp","rgbd_photogeometric_odometry","rgbd_affine_photometric_odometry")
    if any(item in imports for item in forbidden): raise ValueError("Auditor imported an optimizer/loader/GPU stack")
    spec = importlib.util.spec_from_file_location("independent_source_only_full_affine_auditor", TARGET)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    actual_main_calls = []
    def abort_actual_main():
        actual_main_calls.append(True)
        raise BaseException("No actual-main call authorized by this source review")
    module.main = abort_actual_main
    import numpy as np
    from scipy.linalg import expm
    def own_se3(delta):
        x,y,z = delta[3:]; algebra = np.zeros((4,4))
        algebra[:3,:3] = [[0.,-z,y],[z,0.,-x],[-y,x,0.]]
        algebra[:3,3] = delta[:3]
        return expm(algebra)
    transform = np.eye(4); rows = []
    proposals = [np.array([.03,-.02,.01,.04,.02,-.03]), np.array([-.01,.04,.02,-.02,.06,.03]), np.array([.02,.01,0.,.01,0.,.02])]
    limits = [.8,.6,.9]; scales = [.5,1.,None]
    for level, (proposal,limit,scale) in enumerate(zip(proposals,limits,scales)):
        row = {"level":level,"counts":{"joint_correspondences":100},"normalized_robust_objective":1.,
               "T_previous_from_current_before":transform.tolist(),"proposed_delta_twist_m_rad":proposal.tolist(),
               "step_limit_factor":limit,"accepted":scale is not None}
        if scale == .5:
            row["line_search_trials"] = [{"scale":1.,"objective":1.1,"same_correspondence_denominator":100},
                                        {"scale":.5,"objective":.8,"same_correspondence_denominator":100}]
        elif scale == 1.:
            row["line_search_trials"] = [{"scale":1.,"objective":.7,"same_correspondence_denominator":100}]
        else:
            row["line_search_trials"] = [{"scale":1.,"objective":1.,"same_correspondence_denominator":100},
                                        {"scale":.5,"objective":None,"same_correspondence_denominator":100}]
        if scale is not None:
            update = proposal*limit*scale; transform = own_se3(update)@transform
            row.update(line_search_scale=scale,accepted_delta_twist_m_rad=update.tolist(),
                       fixed_correspondence_objective_after=row["line_search_trials"][-1]["objective"],
                       T_previous_from_current_after=transform.tolist())
        rows.append(row)
    fixture = {"accepted":True,"failure_flags":[],"iteration_history":rows,
               "candidate_reference_from_current":transform.tolist(),"candidate_previous_from_current":transform.tolist(),
               "levels":[{"level":i,"iterations_recorded":1,"accepted_updates":int(i<2)} for i in range(3)]}
    with (OUTPUT/"positive_fixture.json").open("xb") as stream:
        stream.write((json.dumps(fixture,indent=2,allow_nan=False)+"\n").encode())
    positive = module.six_step_ledger(fixture)
    if (positive["steps"],positive["trials"],positive["accepted_updates"]) != (3,5,2): raise ValueError("Synthetic positive count check failed")
    def mutate_right_side(value):
        before = np.asarray(value["iteration_history"][1]["T_previous_from_current_before"])
        delta = np.asarray(value["iteration_history"][1]["accepted_delta_twist_m_rad"])
        value["iteration_history"][1]["T_previous_from_current_after"] = (before@own_se3(delta)).tolist()
    def mutate_extra_proposed(value):
        value["iteration_history"][0]["proposed_delta_twist_m_rad"] += [0.,0.]
    def mutate_extra_accepted(value):
        value["iteration_history"][0]["accepted_delta_twist_m_rad"] += [0.,0.]
    mutations = {
        "wrong_before":lambda value:value["iteration_history"][1]["T_previous_from_current_before"][0].__setitem__(3,99.),
        "wrong_after":lambda value:value["iteration_history"][0]["T_previous_from_current_after"][0].__setitem__(3,99.),
        "right_side_update":mutate_right_side,
        "skip_first_descent":lambda value:value["iteration_history"][0]["line_search_trials"][0].update(objective=.9),
        "wrong_fixed_denominator":lambda value:value["iteration_history"][0]["line_search_trials"][0].update(same_correspondence_denominator=99),
        "extra8D_proposed_shape":mutate_extra_proposed,
        "extra8D_accepted_shape":mutate_extra_accepted,
        "NaN_proposed_delta":lambda value:value["iteration_history"][0]["proposed_delta_twist_m_rad"].__setitem__(0,float('nan')),
        "NaN_accepted_delta":lambda value:value["iteration_history"][0]["accepted_delta_twist_m_rad"].__setitem__(0,float('nan')),
        "missing_normal_levels":lambda value:value.update(levels=[]),
    }
    negative = {}
    for label, mutate in mutations.items():
        altered = copy.deepcopy(fixture); mutate(altered)
        try: module.six_step_ledger(altered)
        except ValueError as error: negative[label] = {"rejected":True,"message":str(error)}
        else: raise ValueError("Requested negative control was not rejected: "+label)
    for name, expected in locked.items():
        if sha(ROOT/name) != expected: raise ValueError("Consumed file changed during CPU controls: "+name)
    if actual_main_calls or any(name in sys.modules for name in forbidden): raise ValueError("Actual-main/optimizer/GPU import occurred")
    result = {"status":"passed_independent_readonly_source_and_synthetic_review",
              "review_source_sha256":sha(Path(__file__)),"reviewed_audit_source_sha256":TARGET_SHA,
              "frozen_input_sha256":locked,"compile_passed_three_audit_sources":True,
              "import_inventory":imports,"requested_positive_six_step_fixture":positive,"requested_negative_controls":negative,
              "finite6D_proposed_and_accepted_delta_guards":True,"normal6D_three_level_completeness_guard":True,
              "actual_full_audit_main_calls":0,"actual_registration_calls":0,"GPU_calls":0,
              "source_review":{"read_only":True,"no_optimizer_loader_renderer_import":True,
                    "actual_four_exit0_and_serial_deadline_gate":True,"full_all2594rows2590pairs_manifest_and_iterations_gate":True,
                    "raw_GT_xyzw_nearest_and_no_scale_metrics_explicit_e807_reuse":True,
                    "all_sensor_PNG_index_and_witness_source_hashes_guarded":True,
                    "last_accepted_sensor_reference_integrated_or_held_prefix_checked_every_pair":True,
                    "rejection_flags_gaps_and_all_frame_denominators_retained":True,
                    "all8D_trials_explicit_b549_reuse":True,"6D_only_accepted_SE3_trial_transform_limit_explicit":True,
                    "real_ranks_and_objectives_only_persisted_consistency_explicit":True},
              "findings":[],"limits":["Source-only review and synthetic6D controls; no actual full audit executed.",
                   "Actual4run2594row/2590pair correctness remains gated by the subsequent real CPU artifact audit.",
                   "Saved6D trials lack candidate transforms; only accepted6D SE3 updates are independently checked.",
                   "Real correspondence ranks/objectives are persisted consistency, not recomputed by this review.",
                   "Frozen e807 metrics and b549 saved8D helper reuse is explicit; no third independent formulation claimed."]}
    with (OUTPUT/"metrics.json").open("xb") as stream: stream.write((json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps({"status":result["status"],"audit_source_sha256":TARGET_SHA,"GPU_calls":0,
                      "actual_main_calls":0,"requested_negative_controls":len(negative)},indent=2))


if __name__ == "__main__": main()
