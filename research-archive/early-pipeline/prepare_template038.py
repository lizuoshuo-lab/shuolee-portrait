"""Audit licensed templates already downloaded; extract only the missing candidate."""
from pathlib import Path
import zipfile, io, hashlib, json, datetime
import numpy as np

LAB=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
PRIVATE=BASE/'current/novel-view-face/head-structure/private-runtime/models/smplx'
OUT=BASE/'current/smplx-nlf-template-fresh-038'
ARCHIVE=Path('C:/Users/Administrator/Downloads/models_smplx_v1_1.zip')
sha=lambda x:hashlib.sha256(x).hexdigest()
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
OUT.mkdir(exist_ok=True)
assert not (OUT/'template-audit.json').exists()
current=np.load(PRIVATE/'SMPLX_NEUTRAL.npz',allow_pickle=True)
reports={}
with zipfile.ZipFile(ARCHIVE) as z:
    for family in ['NEUTRAL','MALE']:
        blob=z.read('models/smplx/SMPLX_'+family+'.npz')
        model=np.load(io.BytesIO(blob),allow_pickle=True)
        keys=['v_template','shapedirs','posedirs','f','weights','J_regressor','kintree_table','hands_meanl','hands_meanr']
        for key in keys:
            assert current[key].shape==model[key].shape,(family,key)
        assert np.array_equal(current['f'],model['f'])
        assert np.array_equal(current['kintree_table'],model['kintree_table'])
        delta={key:float(np.max(abs(current[key]-model[key]))) for key in keys if key not in ['kintree_table']}
        if family=='NEUTRAL':
            assert max(delta.values())==0
            dest=PRIVATE/'SMPLX_NEUTRAL.npz'
        else:
            dest=PRIVATE/'SMPLX_MALE.npz'
            if dest.exists():assert sha(dest.read_bytes())==sha(blob)
            else:dest.write_bytes(blob)
        reports[family]=dict(path=str(dest),sha256=sha(dest.read_bytes()),archive_member_sha256=sha(blob),
            arrays={k:list(model[k].shape) for k in keys},max_abs_delta_from_current_neutral=delta)
report=dict(created=stamp,source='Existing user-downloaded licensed models_smplx_v1_1.zip',
    templates=reports,topology_and_joint_hierarchy_identical=True,neutral_numeric_arrays_identical=True,
    family_shape_coefficients_interchangeable=False,
    experiment='Both arms start with zero shape coefficients, identical original NLF pose/root evidence, fixed cameras, new optimizer,1200steps. Only full template family differs.',
    identity_inference=False,old_personalized_initialization=False,new_download_bytes=0)
(OUT/'template-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
