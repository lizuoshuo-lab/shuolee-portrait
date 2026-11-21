import json,os,sys,time
from pathlib import Path
stop=Path(sys.argv[sys.argv.index("--stop-file")+1]);ready=Path(sys.argv[sys.argv.index("--ready-file")+1])
ready.write_text(json.dumps({"pid":os.getpid(),"argv":sys.argv,"CPU_role_fixture":True,"Torch_imported":False,"CUDA_called":False}),encoding="utf-8")
end=time.monotonic()+20
while not stop.exists() and time.monotonic()<end:time.sleep(.05)
