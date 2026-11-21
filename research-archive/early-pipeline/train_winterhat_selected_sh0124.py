"""Use the completed eight-view winterhat fit with the unchanged sofa GS handoff.

Only the input roster, reference ID and repeated shared-beta storage differ.
No pose refit, new loss, new optimizer, or density schedule is introduced.
"""
import argparse
import difflib
import inspect
from pathlib import Path
import train_sofa_selected_sh0124 as existing

SUBJECT = 'winterhat-selected-sh0-pipeline124'
OUT = existing.gs.ROOT / SUBJECT
SOURCE = existing.gs.ROOT / 'winterhat-selected-view-pose124'
REFERENCE = 'frame_000071'


def prepare():
    before = inspect.getsource(existing.prepare)
    changes = {
        'assert n == 7 and ids ==': 'assert n == 8 and ids ==',
        "    fit['shared_betas'] = fit['betas'].copy()":
            "    assert np.array_equal(fit['betas'], np.repeat(fit['betas'][:1], n, axis=0))\n"
            "    fit['betas'] = fit['betas'][:1].copy()\n"
            "    fit['shared_betas'] = fit['betas'].copy()",
        "    new = old.replace('assert len(training_slots)==8', 'assert len(training_slots)==7').replace('training_frames=8', 'training_frames=7')":
            '    new = old',
        "seven-frame-interface.diff": "eight-frame-interface.diff",
        "All seven preregistered selected-view fit inputs": "All eight preregistered selected-view fit inputs",
        "label='沙发人物7帧 SH0 全流程试跑'": "label='冬帽人物8帧 SH0 全流程试跑'",
        "record_id='006'": "record_id='001'",
        "Seven preregistered sofa inputs": "Eight preregistered winterhat inputs",
        "Path(__file__), Path(gs.__file__)": "Path(__file__), Path(existing.__file__), Path(gs.__file__)",
    }
    after = before
    for old, new in changes.items():
        assert after.count(old) == 1, old
        after = after.replace(old, new)
    ns = dict(existing.__dict__)
    ns.update(SUBJECT=SUBJECT, OUT=OUT, SOURCE=SOURCE, REFERENCE=REFERENCE,
              existing=existing, __file__=__file__)
    exec(compile(after, __file__ + ':input-adaptation', 'exec'), ns)
    ns['prepare']()
    (OUT / 'subject-input-interface.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='sofa-input-handoff',
        tofile='winterhat-input-handoff')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        existing.gs.train(SUBJECT)
