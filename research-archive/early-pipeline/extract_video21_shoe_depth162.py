"""Replay the reviewed shoe tracks and retain their existing 3D measurements."""
from pathlib import Path
import sys

import fit_video21_native148 as portrait
from update_leafwoman129_state import atomic, read, sha


BASE = portrait.OUT
FROZEN = BASE/'shoe-material-local161'
OUT = BASE/'shoe-material-depth162'
SELECTED = {
    'frame_000029': ['image_left_shoe_welt', 'image_right_shoe_welt'],
    'frame_000053': ['near_shoe_eyelet', 'far_shoe_lace'],
}


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old,new)


def main():
    assert not OUT.exists()
    frozen_decision = read(BASE/'fixedk-dense159-reference/review/visual-decision.json')
    source = FROZEN/'source.py'
    assert frozen_decision['material_gallery_actual_review']
    assert frozen_decision['local_shoe_source_sha256'] == sha(source)
    assert frozen_decision['local_shoe_result_sha256'] == sha(FROZEN/'result.json')
    code = source.read_text(encoding='utf-8')
    code = replace(code, "destination=OUT/'shoe-material-local161'",
                   "destination=OUT/'shoe-material-depth162'")
    code = replace(code, "triangulated_from=[a[0],b[0]])",
                   """triangulated_from=[a[0],b[0]],xyz_arbitrary_scale=h[:3].tolist(),
                                  observed_uv={str(i):xy.tolist() for i,xy,_,_ in supported})""")
    exec(compile(code, str(source), 'exec'), {})
    old = read(FROZEN/'result.json')['rows']
    new = read(OUT/'result.json')['rows']
    assert len(old) == len(new) == 14
    for a,b in zip(old,new,strict=True):
        assert a == {k:v for k,v in b.items() if k not in ('xyz_arbitrary_scale','observed_uv')}
    picked = {f:[next(r for r in new if r['reference']==f and r['material']==label)
                 for label in labels] for f,labels in SELECTED.items()}
    assert all(row['held_max_px'] < 1. for rows in picked.values() for row in rows)
    atomic(OUT/'measurements.json',dict(source_result_sha256=sha(FROZEN/'result.json'),
        replay_result_sha256=sha(OUT/'result.json'), reviewed_materials=picked,
        frame_windows_have_separate_points=True, full_pose_observed=False, world_scale_known=False))
    print('VERIFIED_SHOE_DEPTH_EXTRACTED',len(picked),flush=True)


if __name__=='__main__':
    # The saved diagnostic expects the directory containing the project helpers.
    sys.argv[1:] = [str(Path(__file__).resolve().parent)]
    main()
