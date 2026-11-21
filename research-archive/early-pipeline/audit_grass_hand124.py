"""Read-only check of whether the winterhat wrist conflict explains grass."""
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read, save, sha, now
from smplify_score_contract123 import relative_confidence


def main():
    root = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
    source = Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory/data/observations.npz')
    output = root/'grass/pose-cause-audit/hand-coherence.json'
    assert not output.exists(), 'Preserve the original audit.'
    with np.load(source, allow_pickle=False) as z:
        obs = dict(z)
    confidence, _ = relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    assert len(obs['frame_ids']) == 94
    records = []
    for i, name in enumerate(obs['frame_ids'].astype(str)):
        row = dict(frame=name)
        for side, parent, start in [('right', 10, 112), ('left', 9, 91)]:
            c = confidence[i, start:start+21]
            row[side] = dict(body_wrist_confidence=float(confidence[i, parent]),
                hand_root_confidence=float(c[0]), hand_median_confidence=float(np.median(c)),
                wrist_2d_disagreement_px=float(np.linalg.norm(obs['keypoints'][i, parent]-obs['keypoints'][i, start])),
                body_flip_disagreement_px=float(obs['flip_disagreement'][i, parent]),
                prospective_chain_gate=float(min(confidence[i, parent], c[0])))
        records.append(row)
    selected = [r for r in records if r['frame'] in ('frame_000032', 'frame_000044', 'frame_000045')]
    assert len(selected) == 3
    result = dict(updated=now(), source=str(source), source_sha256=sha(source),
        source_code_sha256=sha(Path(__file__)), frames=94, records=records,
        previously_diagnosed_frames=selected, no_fit_or_observation_changed=True,
        interpretation='The grass right wrist at32/44/45 retains strong body and hand confidence, unlike winterhat96. Duplicate-wrist disagreement is about8 pixels, not106 pixels with a suppressed body wrist. The same missing-parent-confidence mechanism is not supported as an explanation of these grass failures. Monocular depth ambiguity and independent pose branch changes remain possible; no unique cause or improvement is established.',
        quality_accepted=False, new_fitting_started=False)
    save(output, result)
    print('GRASS_HAND_AUDIT_READY', selected, flush=True)


if __name__ == '__main__':
    main()
