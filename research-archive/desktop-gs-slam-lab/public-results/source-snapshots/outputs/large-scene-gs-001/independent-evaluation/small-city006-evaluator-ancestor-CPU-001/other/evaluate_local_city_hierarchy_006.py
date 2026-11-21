import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index("--root")+1]);(root/"ready.json").write_text(json.dumps({"pid":os.getpid(),"argv":sys.argv,"CPU_only":True}),encoding="utf-8")
deadline=time.monotonic()+15
while not (root/"stop.flag").exists() and time.monotonic()<deadline:time.sleep(.05)
