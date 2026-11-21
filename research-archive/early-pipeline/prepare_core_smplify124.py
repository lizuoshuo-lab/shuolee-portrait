"""Apply the existing pre-fit selector to the newly observed core subjects."""
import argparse
from pathlib import Path
import prepare_winterhat_sh0124 as original
from prepare_core_subjects124 import paths
from run_official_gaussians123 import read, save, sha, now


def main(subject):
    video, data, report, number, digest = paths(subject)
    result = read(report/'result.json')
    assert result['state'] == 'observations_complete_review_pending'
    assert result['observations_ready']
    assert sha(data/'data/observations.npz') == result['observations_sha256']
    assert sha(data/'data/inputs.json') == result['observation_manifest_sha256']
    original.ROOT = original.REC/'cross-subject-sh0-124'/(subject+'-input')
    original.DATA = data
    original.VIDEO = video
    original.EXPECTED = digest
    original.prepare()
    plan = read(original.ROOT/'plan.json')
    plan.update(updated=now(), subject=subject, record_id=number,
        user='Continue the named multi-person goal with SMPLify-X temporal constraints.',
        state='native_selection_ready_visibility_review_pending',
        selector_source=str(Path(original.__file__)), selector_sha256=sha(original.__file__),
        input_adapter_source=str(Path(__file__)), input_adapter_sha256=sha(__file__),
        input_roster=result['frames'], sample_fps=3,
        one_12frame_fit=False, new_reference_fit_required=True,
        full_temporal_fit_required=True, no_fitting_started=True,
        no_old_individual_geometry=True, gaussian_admission=False,
        comparison_limit='Same pre-fit selection rule, native observation entry and conditional K transform; new3fps roster. No claim of identical winterhat sequence or end-to-end timing.',
        next='Review all8+4 selected native masks and observations, including truncated or furniture-occluded feet, before fitting. Do not infer invisible geometry from detector confidence.')
    save(original.ROOT/'plan.json', plan)
    print('CORE_SMPLIFY_INPUT_READY', subject, result['frames'], flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('subject',choices=['sofa','indoor'])
    main(p.parse_args().subject)
