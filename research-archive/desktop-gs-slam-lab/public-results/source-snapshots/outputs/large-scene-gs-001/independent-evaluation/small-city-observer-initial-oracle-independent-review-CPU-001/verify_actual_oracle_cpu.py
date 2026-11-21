"""Independent readonly actualPlyData→rawuint32 initialization oracle check."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from plyfile import PlyData
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001';OUT=Path(__file__).parent
ORACLE=BASE/'official-training-observation-plans/small-city-initial-suspect-oracle-001/expected-raw.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def main():
    require(not (OUT/'review.json').exists(),'Preserveactualoracle review')
    oracle=json.loads(ORACLE.read_text(encoding='utf-8'));path=Path(oracle['actual_coarse_PLY']);initial_sha=sha(ORACLE)
    require(sha(path)==oracle['actual_coarse_PLY_sha256']=='abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31','Actualcoarse modelSHA differs')
    require(oracle['status']=='CPU_exact_actual_coarse_raw6group_to_initialized_suspect_oracle_passed' and oracle['original_coarse_PLY_row_zero_based']==1778538 and oracle['suspect_selected_prefix_row_zero_based']==1377084,'Actualoracle identity failed')
    data=PlyData.read(path,mmap='r')['vertex'].data;v=data[1778538]
    def a(fields):return np.ascontiguousarray([v[f] for f in fields],dtype=np.float32)
    rest=np.zeros((15,3),np.float32)
    # Direct basis/channel indexing, independent of upstream reshape transpose.
    for basis in range(3):
        for channel in range(3):rest[basis,channel]=v[f'f_rest_{channel*3+basis}']
    groups=dict(xyz=a(('x','y','z')),f_dc=a(('f_dc_0','f_dc_1','f_dc_2'))[None,:],f_rest=rest,
        opacity=a(('opacity',)),scaling=a(('scale_0','scale_1','scale_2')),rotation=a(('rot_0','rot_1','rot_2','rot_3')))
    records=[]
    for name,value in groups.items():
        expected=oracle['groups'][name];bits=[f'{int(x):08x}' for x in value.view(np.uint32).reshape(-1)]
        require(expected['dtype']=='float32' and expected['shape']==list(value.shape) and expected['uint32_hex']==bits,'Actual6group rawbits/shape mismatch:'+name)
        require(expected['raw_group_bytes_sha256']==hashlib.sha256(value.tobytes()).hexdigest(),'Actual6group bytesSHA mismatch:'+name)
        require(np.array_equal(value,np.asarray(expected['values'],np.float32)) and np.isfinite(value).all(),'Actual6group values mismatch:'+name)
        records.append(dict(group=name,shape=list(value.shape),actual_raw_uint32_bitexact=True,raw_group_bytes_sha256=expected['raw_group_bytes_sha256']))
    chunk=BASE/'official-inputs/small-city-training-002/chunks/1_1';center=np.loadtxt(chunk/'center.txt',dtype=np.float32);extent=np.loadtxt(chunk/'extent.txt',dtype=np.float32)
    distance=np.maximum(np.abs(data['x']-center[0]),np.abs(data['y']-center[1]));selected=(distance>np.float32(.5)*extent[0])&(distance<np.float32(1.5)*extent[0]);selected[:100000]=True;indices=np.flatnonzero(selected)
    require(len(indices)==oracle['selected_scaffold_count']==1835543 and indices[1377084]==1778538,'Actualprefix mapping differs')
    require(sha(ORACLE)==initial_sha and sha(path)==oracle['actual_coarse_PLY_sha256'] and 'torch' not in sys.modules,'Readonlysource modified/Torch imported')
    result=dict(status='passed_independent_actual_coarse_raw6groups_uint32_and_SH3_padding_oracle_CPU',oracle_path=str(ORACLE),oracle_sha256=initial_sha,actual_coarse_sha256=oracle['actual_coarse_PLY_sha256'],originalrow=1778538,selectedrow=1377084,selected_scaffold_count=len(indices),groups=records,
        SH1_basis_channel_indexing_independently_checked=True,higher_SH12basis_rows_positive_zero_uint32=bool(np.all(rest[3:].view(np.uint32)==0)),
        actual_runtime_initialized_G_model_or_GPU_not_checked=True,source_and_model_modified=False,Torch_imported=False,CUDA_called=False)
    write(OUT/'review.json',result);print(json.dumps(dict(status=result['status'],oracle_sha256=initial_sha,review_sha256=sha(OUT/'review.json'))))
if __name__=='__main__':main()
