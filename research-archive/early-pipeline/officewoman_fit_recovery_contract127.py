"""Read-only, stdlib-only contract for one specific127 CPU metadata recovery.

This never repairs files or treats reconstructed metadata as fitting quality.
The original process exit1 and original failure remain canonical evidence.
"""
import hashlib
import json
import os
from pathlib import Path
import re

FROZEN_FIT_HELPER_SHA256='0dd3be080d32705bc4f6e8315eb0cacaeea66a4599e478676409abf7156bcac3'
RECEIPT_NAME='postprocess-recovery127.json'
ERROR='zero-size array to reduction operation maximum which has no identity'
RUNNER='<same-official-neutral-shared-officewoman127-runner>'


def _path(path):
    path=Path(path)
    return Path('\\\\?\\'+str(path.resolve())) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path


def _sha(path):
    return hashlib.sha256(_path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(_path(path).read_text(encoding='utf-8-sig'))


def verify_recovery(out,frozen_helper_path,required=False):
    """Return a verified receipt summary or None when no receipt exists.

    A present but incomplete/different recovery raises rather than falling back
    to success. No arrays, optimization, rendering or process launch is done.
    """
    out=Path(out);path=out/RECEIPT_NAME
    if not _path(path).exists():
        if required:
            raise ValueError('Original127fit exit was not zero; verified CPU metadata-recovery receipt is required.')
        return None
    r=_read(path);exit_doc=_read(out/'process-exit.json');failure=_read(out/'failure.json')
    assert r['cpu_exit_code']==0 and r['original_process_exit_code']==exit_doc['exit_code']==1
    for key in ('source_hashes_verified','all5_stage_log_records_verified','all24_per_frame_arrays_verified','no_optimizer_updates','no_GPU'):
        assert r[key] is True,key
    pairs=(('original_failure_sha256','failure.json'),('original_process_exit_sha256','process-exit.json'),
        ('fit_log_sha256','fit.log'),('executed_runner_sha256','executed-runner.py.txt'),('result_sha256','result.json'))
    for key,name in pairs:
        assert r[key]==_sha(out/name),(key,name)
    assert r['frozen_fit_helper_sha256']==_sha(frozen_helper_path)==FROZEN_FIT_HELPER_SHA256
    assert failure['error']=='ValueError' and failure['message']==ERROR
    assert f'File "{RUNNER}", line 41, in run' in failure['traceback']
    lines=_path(out/'executed-runner.py.txt').read_text(encoding='utf-8-sig').splitlines()
    assert "metric[name+'_supported_max_px']=float(error[ix][w>0].max())" in lines[40]
    log=_path(out/'fit.log').read_text(encoding='utf-8-sig')
    stages=re.findall(r'Stage (\d{3}) done after ([0-9.]+) seconds',log)
    assert [n for n,_ in stages]==['000','001','002','003','004']
    assert re.search(r'Body fitting Orientation 0 done after [0-9.]+ seconds',log)
    selected=_sha(out/'selected-fit.npz')
    assert r['selected_fit_sha256_before']==r['selected_fit_sha256_after']==selected
    result=_read(out/'result.json');plan=_read(out/'plan.json')
    assert result['fit_sha256']==selected and result['metadata_reconstructed'] is True
    assert result['original_in_memory_receipt_recovered'] is False
    assert result['quality_accepted'] is False and result['gaussian_admitted'] is False
    frames=[row['frame'] for row in plan['rows']]
    assert len(frames)==len(set(frames))==24
    files=r['per_frame_files']
    assert len(files)==24 and [row['frame'] for row in files]==frames
    for row in files:
        assert row['frame'] in frames and set(row)=={'frame','sha256'}
        assert row['sha256']==_sha(out/'frames'/row['frame']/'winner-camera-space.npz')
    return dict(verified=True,receipt_name=RECEIPT_NAME,receipt_sha256=_sha(path),
        original_process_exit_code=1,cpu_postprocess_exit_code=0,selected_fit_sha256=selected,
        original_failure_retained=True,optimizer_updates=0,GPU_started=False,
        metadata_reconstructed=True,actual_visual_review=False,quality_accepted=False)


def review_completion(out,frozen_helper_path):
    exit_code=_read(Path(out)/'process-exit.json')['exit_code']
    if exit_code==0:
        return dict(original_process_exit_code=0,metadata_reconstructed=False,quality_accepted=False)
    if exit_code!=1:
        raise ValueError('Unsupported original127fit exit code.')
    return verify_recovery(out,frozen_helper_path,required=True)
