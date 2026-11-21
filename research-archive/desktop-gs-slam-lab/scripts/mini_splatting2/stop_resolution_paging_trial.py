"""Stop only the verified full-resolution trial suffering WDDM paging."""
import json
import os
from pathlib import Path
import signal
import time
ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
audit=json.loads((ROOT/'native-resolution-evaluator-correction-001/training-runtime.json').read_text())
pid=audit['processes'][0]['pid'];path=Path('/proc')/str(pid)
cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert cmd.startswith('/opt/gs-slam-lab/dynamic-human/.venv/bin/python ')
assert '/repo-depthguide/msv2/train_depth_guided.py' in cmd and 'native-resolution-full-18000-s0/model' in cmd
receipt=dict(status='interruption_requested',created=time.time(),linux_training_pid=pid,
    reason='Full-resolution training uses ~4GiB Windows shared GPU memory and slows to ~2iterations/s; revise bounded prior cache and mathematically equivalent backward scheduling.',
    evidence='Windows GPU Process Memory counters for vmwp PID29808: shared4056MiB, dedicated15203MiB. Separate ASR PID18064 untouched.',
    preserve='All completed models, partial trial logs and original trial directory stay intact; no restart into existing output directory.',
    change_scope='Interrupt this training child only; its orchestrators exit on the failed child.')
(ROOT/'native-resolution-paging-interruption.json').write_text(json.dumps(receipt,indent=2))
os.kill(pid,signal.SIGINT);print(json.dumps(receipt))
