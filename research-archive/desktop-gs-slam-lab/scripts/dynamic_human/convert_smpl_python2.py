"""Convert official Python-2 SMPL constant arrays without installing Chumpy.

The restricted loader accepts only the globals inventoried in this exact archive.
Chumpy constants are unwrapped only when their stored x value is an ndarray.
The original model is preserved; tensor values are checked after serialization.
"""
from pathlib import Path
import builtins
import copyreg
import hashlib
import io
import json
import pickle
import shutil
import numpy as np
from scipy import sparse

LAB = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT = LAB / 'outputs/dynamic-human-smpl-010'
TARGET = Path('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl')
WINDOWS = LAB / 'repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl'
RAW = LAB / 'data/licensed/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'


class ChumpyConstant:
    pass


class ModelReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('__builtin__', 'object'): builtins.object,
            ('__builtin__', 'set'): builtins.set,
            ('copy_reg', '_reconstructor'): copyreg._reconstructor,
            ('numpy', 'dtype'): np.dtype,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('scipy.sparse.csc', 'csc_matrix'): sparse.csc_matrix,
            ('chumpy.ch', 'Ch'): ChumpyConstant,
        }
        if (module, name) not in allowed:
            raise ValueError(f'Unexpected pickle global: {module}.{name}')
        return allowed[module, name]


def main():
    receipt = json.loads((OUT / 'template-installation.json').read_text())
    raw = TARGET.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['model_sha256'], 'Only convert the inspected original model'
    model = ModelReader(io.BytesIO(raw), encoding='latin1').load()
    assert isinstance(model, dict)
    unwrapped = {}
    for key, value in model.items():
        if isinstance(value, ChumpyConstant):
            fields = vars(value)
            assert isinstance(fields.get('x'), np.ndarray), (key, {k: type(v).__name__ for k, v in fields.items()})
            unwrapped[key] = dict(stored_fields=sorted(fields), shape=list(fields['x'].shape))
            model[key] = fields['x']
    # v1.1.0 includes 300 shape components despite the legacy filename. Keep
    # all of them; SMPLLayer selects the first 10 for the author's pose inputs.
    expected = {'v_template': (6890, 3), 'shapedirs': (6890, 3, 300),
        'posedirs': (6890, 3, 207), 'weights': (6890, 24), 'J_regressor': (24, 6890),
        'f': (13776, 3), 'kintree_table': (2, 24)}
    arrays = {}
    for key, value in model.items():
        if sparse.issparse(value):
            a = value.toarray()
        elif isinstance(value, np.ndarray):
            a = value
        else:
            assert isinstance(value, (str, int, float, list, tuple, dict)), (key, type(value))
            continue
        assert a.dtype.kind in 'biufc' and np.isfinite(a).all(), (key, a.dtype)
        if key in expected:
            assert a.shape == expected[key], (key, a.shape)
        arrays[key] = dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())
    assert set(expected).issubset(arrays)
    np.testing.assert_allclose(model['weights'].sum(1), 1, atol=1e-7)
    assert model['f'].min() == 0 and model['f'].max() == 6889
    data = pickle.dumps(model, protocol=4)
    # This is our own generated plain NumPy/SciPy data, not a new external pickle.
    restored = pickle.loads(data)
    for key, entry in arrays.items():
        a = restored[key].toarray() if sparse.issparse(restored[key]) else restored[key]
        assert hashlib.sha256(a.tobytes()).hexdigest() == entry['sha256']
    digest = hashlib.sha256(data).hexdigest()
    original_local = TARGET.with_name('SMPL_NEUTRAL.python2-original.pkl')
    assert not original_local.exists()
    shutil.copy2(TARGET, original_local)
    TARGET.write_bytes(data)
    WINDOWS.write_bytes(data)
    assert hashlib.sha256(WINDOWS.read_bytes()).hexdigest() == digest
    report = dict(status='converted_without_chumpy_all_numeric_values_preserved',
        original_sha256=receipt['model_sha256'], normalized_sha256=digest,
        normalized_bytes=len(data), unwrapped_constants=unwrapped, arrays=arrays,
        original_windows=str(RAW), original_linux=str(original_local),
        runtime_model=str(TARGET), windows_model=str(WINDOWS),
        transformation='Python-2 latin1 byte decoding; unwrap constant Chumpy.x arrays; serialize plain NumPy/SciPy protocol4. No template/shape/pose/topology values changed.',
        publication='Licensed original and derived models remain local.')
    (OUT / 'template-conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
