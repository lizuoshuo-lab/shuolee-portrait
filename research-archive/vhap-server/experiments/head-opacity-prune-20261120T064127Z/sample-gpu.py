from pathlib import Path
from datetime import datetime, timezone
import subprocess,time,csv,re,sys
roots=list(map(Path,sys.argv[1:]))
with open("gpu-samples.csv","x",buffering=1) as f:
 writer=csv.writer(f); writer.writerow(["utc","run","step_floor","pid","gpu_memory_mib"])
 while not (roots[-1]/"joint/metrics.json").is_file():
  table=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True)
  for line in table.splitlines():
   pid,mb=map(str.strip,line.split(","))
   try: words=Path("/proc",pid,"cmdline").read_bytes().split(b"\0")
   except (FileNotFoundError,PermissionError): continue
   for label,root in zip(["A","B"],roots):
    if str(root/"joint").encode() not in words: continue
    log=root/"logs/gaussian_training-1.log"
    if not log.exists(): continue
    with log.open("rb") as stream:
     stream.seek(max(0,log.stat().st_size-6000)); tail=stream.read()
    matches=re.findall(rb'step (\d+)/10000',tail)
    writer.writerow([datetime.now(timezone.utc).isoformat(),label,int(matches[-1]) if matches else 0,pid,mb])
  time.sleep(5)
