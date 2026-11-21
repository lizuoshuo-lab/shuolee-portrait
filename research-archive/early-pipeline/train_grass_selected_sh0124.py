"""Frozen seven-view grass fit -> the existing SH0 mesh-bound training.

This reuses the sofa handoff, with an explicit camera-to-world conversion for
the grass input cameras. It never fits a pose or changes the Gaussian solver.
"""
import argparse
import difflib
import inspect
from pathlib import Path
import train_sofa_selected_sh0124 as existing

SUBJECT='grass-selected-sh0-pipeline124'
OUT=existing.gs.ROOT/SUBJECT
SOURCE=existing.gs.ROOT/'grass-screened-seven124'
REFERENCE='frame_000046'

def prepare():
    before=inspect.getsource(existing.prepare)
    changes={
        "    assert np.array_equal(fit['w2c'], np.broadcast_to(np.eye(4), (n, 4, 4)))":
            "    E=fit['w2c']; rotation=E[:,:3,:3]\n"
            "    assert np.allclose(rotation@np.swapaxes(rotation,1,2),np.eye(3),atol=1e-5)\n"
            "    assert np.allclose(np.linalg.det(rotation),1.,atol=1e-5)",
        "    fit['vertices_world'] = fit['vertices_camera'].astype(np.float32)":
            "    fit['vertices_world'] = np.stack([(fit['vertices_camera'][i]-E[i,:3,3])@E[i,:3,:3] for i in range(n)]).astype(np.float32)",
        "label='沙发人物7帧 SH0 全流程试跑'":"label='草地白衣7帧 SH0 全流程试跑'",
        "record_id='006'":"record_id='005'",
        'Seven preregistered sofa inputs':'Seven preregistered grass shared-fit inputs',
        "Path(__file__), Path(gs.__file__)":"Path(__file__), Path(existing.__file__), Path(gs.__file__)",
        "camera_kind='camera_relative_fixedK_body_motion'":"camera_kind='source_fixedK_and_w2c_preserved_not_independently_calibrated'",
    }
    after=before
    for old,new in changes.items():
        assert after.count(old)==1,old
        after=after.replace(old,new)
    ns=dict(existing.__dict__)
    ns.update(SUBJECT=SUBJECT,OUT=OUT,SOURCE=SOURCE,REFERENCE=REFERENCE,existing=existing,__file__=__file__)
    exec(compile(after,__file__+':input-adaptation','exec'),ns)
    ns['prepare']()
    (OUT/'subject-input-interface.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True),after.splitlines(True),fromfile='sofa-input-handoff',tofile='grass-input-handoff')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train']);args=p.parse_args()
    if args.mode=='prepare':prepare()
    else:existing.gs.train(SUBJECT)
