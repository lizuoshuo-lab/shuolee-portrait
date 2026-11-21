"""Archive the exact new route and fixed input contracts without copying assets."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
HELP=Path(__file__).parent
OUT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/clothed-surface-119/source')
files=[ROOT/'src/shuolee_portrait'/s for s in ['clothed_surface.py','clothed_training.py','training.py','fitting.py','prior.py','export.py']]
files += [ROOT/'src/avatar_core/scene'/s for s in ['gaussian_model.py','static_world_gaussian_model.py','smplx_gaussian_model.py']]
files += [HELP/s for s in ['capture119_clothed.py','review_capture119.py','audit_capture119_representation.py','stage_capture119.py','publish_capture119.py','record_capture119_decision.py','record_capture119_ui.py','prepare_capture119_orbits.py']]
files += [HELP/s for s in ['audit_capture119_fov.py','record_capture119_portrait_final.py','record_capture119_sofa_final.py']]
OUT.mkdir(exist_ok=True);records=[]
for source in files:
    digest=hashlib.sha256(source.read_bytes()).hexdigest();target=OUT/source.name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        target=target.with_name(target.stem+'-'+digest[:12]+target.suffix)
    if not target.exists():shutil.copy2(source,target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    records.append(dict(source=str(source),snapshot=str(target),sha256=digest))
(OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('SNAPSHOTTED',len(records))
