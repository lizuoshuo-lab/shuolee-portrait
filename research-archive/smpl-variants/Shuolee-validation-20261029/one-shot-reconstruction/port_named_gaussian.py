"""Private interface migration; research inputs are read only."""
import ast
import hashlib
from pathlib import Path

source = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/shuolee_portrait')
target = Path('D:/Projects/Shuolee/src/shuolee_portrait')
training = (source/'training.py').read_text(encoding='utf-8')
exporting = (source/'export.py').read_text(encoding='utf-8')
assert hashlib.sha256((source/'training.py').read_bytes()).hexdigest() == 'ea5520249cc931e9d63e6ec59d62bcd6ac33fff52ca787d4ae0c950f61830d99'
assert hashlib.sha256((source/'export.py').read_bytes()).hexdigest() == '4da488438b5fcfb40f6a028d5bab9952c223a522f3446d9a7ba6344d5ead1520'

def definition(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])

def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

names = ['camera', 'named_pose_layout', 'render_pose', 'reference_ply_comparison',
         'foreground_loss', 'image', 'Views', 'train', 'verify_official_fit']
prefix = training[:training.index('def camera(')]
parts = [definition(training, name) for name in names]
parts[-1] = parts[-1].replace('official exploratory entry', 'official entry')
parts[-1] = replace_once(parts[-1], "    rows_by_frame = {row['frame']: row for row in rows}",
    "    from .official_sequence import checked_image, check_result\n"
    "    rows_by_frame = {row['frame']: row for row in rows}")
parts[-1] = replace_once(parts[-1], "        row = rows_by_frame[item['frame_id']]",
    "        row = rows_by_frame[item['frame_id']]\n"
    "        check_result(result, row)\n"
    "        checked_image(row)")
entry = '''def train_official(root, cfg):
    """Use the existing numerical loop on hash-bound official named fits."""
    return _train_static(Path(root), cfg, execution_kind='official_smplifyx')'''
loop = definition(training, '_train_static')
loop = replace_once(loop, "def _train_static(root, cfg, *, step_limit=None, execution_kind='formal_training', checkpoint_review=None):",
                         "def _train_static(root, cfg, *, execution_kind='formal_training'):")
loop = loop.replace("execution_kind == 'exploratory_official_smplx'", "execution_kind == 'official_smplifyx'")
loop = replace_once(loop, "schedule_steps = cfg['iterations']; steps = schedule_steps if step_limit is None else step_limit",
                         "schedule_steps = cfg['iterations']; steps = schedule_steps")
loop = replace_once(loop, '    stopped_after_review = False\n', '')
loop = replace_once(loop, '''                if checkpoint_review is not None and not checkpoint_review(step, snapshot):
                    stopped_after_review = True
                    steps = step
                    break
''', '')
loop = replace_once(loop, "state='stopped_after_actual_review' if stopped_after_review else 'completed_quality_review_pending'", "state='completed'")
loop = replace_once(loop, "    counts = np.zeros(len(rows), np.int32);", "    if (root/'status.json').is_file():\n"
    "        status = rt.read(root/'status.json')\n"
    "        status.update(stage='training', training_started=True, updated=rt.now())\n"
    "        rt.save(root/'status.json', status)\n"
    "    counts = np.zeros(len(rows), np.int32);")
(target/'training.py').write_text(prefix+'\n\n\n'.join([*parts, entry, loop])+'\n', encoding='utf-8')

# Keep the engineering export's Windows buffer and standard camera exchange
# fixes; port only named-reference selection and its existing transforms.
out = (target/'export.py').read_text(encoding='utf-8')
out = replace_once(out, 'def rotate_ply(', definition(exporting, 'reference_pose_index')+'\n\n\ndef rotate_ply(')
out = replace_once(out, "    out=root/'export';out.mkdir();fit=dict(np.load(root/'fit/fitted.npz'))",
    "    with np.load(root/'fit/fitted.npz',allow_pickle=False) as archive:fit=dict(archive)\n"
    "    reference_slot,reference_frame=reference_pose_index(fit,receipt)\n"
    "    standard_export=cfg.get('gaussian_representation')=='static_clothed_world' or reference_frame is not None\n"
    "    out=root/'export';out.mkdir()")
out = out.replace("fit['root_rotation'][0]", "fit['root_rotation'][reference_slot]")
out = out.replace("fit['world_translation'][0]", "fit['world_translation'][reference_slot]")
out = replace_once(out, "    report['export_frame_mode']=frame_mode", "    report['export_frame_mode']=frame_mode\n"
    "    report['reference_frame_id']=reference_frame\n"
    "    report['reference_pose_index']=reference_slot")
out = out.replace("if cfg.get('gaussian_representation')=='static_clothed_world':", "if standard_export:")
out = replace_once(out, "    standard119=cfg.get('gaussian_representation')=='static_clothed_world'\n", '')
out = out.replace('standard119', 'standard_export')
out = replace_once(out, "camera(rows[i],int(i),768);cb,_=camera(targetrows[i],int(i),768)",
    "camera(rows[i],int(i),768,pose_index=reference_slot);cb,_=camera(targetrows[i],int(i),768,pose_index=reference_slot)")
out = replace_once(out, "cam,_=camera(row,i);", "cam,_=camera(row,i,pose_index=reference_slot);")
out = replace_once(out, "format_passed=True,quality_accepted=False,from_scratch=True,frames=len(rows)",
    "format_passed=True,quality_accepted=False,reference_frame_id=reference_frame,reference_pose_index=reference_slot,from_scratch=True,frames=len(rows)")
(target/'export.py').write_text(out, encoding='utf-8')
print('Ported existing named-frame training and reference export interfaces; no optimization executed.')
