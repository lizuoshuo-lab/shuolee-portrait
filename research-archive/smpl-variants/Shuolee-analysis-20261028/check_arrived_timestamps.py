"""Distinguish input timing from null-output muxer time-base warnings."""
from pathlib import Path
import json,subprocess
import numpy as np
base=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
path=base/'arrived-videos-validation.json';report=json.loads(path.read_text())
for row in report['videos']:
    if not row['decode_errors']:continue
    p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries',
        'frame=best_effort_timestamp_time','-of','json',row['path']],capture_output=True,text=True,check=True)
    frames=json.loads(p.stdout)['frames'];times=np.array([float(f['best_effort_timestamp_time']) for f in frames])
    delta=np.diff(times)
    row['input_timestamp_check']=dict(decoded_frames=len(frames),nonincreasing_steps=int((delta<=0).sum()),
        min_interval_seconds=float(delta.min()),max_interval_seconds=float(delta.max()),
        strictly_increasing=bool((delta>0).all()),ffprobe_errors=p.stderr)
    row['null_muxer_warning_is_input_nonmonotonicity']=not bool((delta>0).all())
    print(json.dumps(dict(name=row['name'],input_timestamp_check=row['input_timestamp_check'])),flush=True)
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
