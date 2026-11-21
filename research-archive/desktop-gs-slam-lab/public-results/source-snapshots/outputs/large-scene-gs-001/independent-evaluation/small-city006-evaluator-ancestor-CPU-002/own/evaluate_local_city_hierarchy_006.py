import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index("--control-root")+1]);adapter=Path(sys.argv[sys.argv.index("--adapter")+1])
if "--chain-parent" in sys.argv:
    args=[sys.executable,str(Path(__file__).resolve()),"--gpu","--chain-child","--control-root",str(root),"--adapter",str(adapter)]
    p=subprocess.Popen(args,creationflags=subprocess.CREATE_NO_WINDOW)
    (root/"parent-ready.json").write_text(json.dumps({"pid":os.getpid(),"child_popen_pid":p.pid,"argv":sys.argv,"CPU_only":True}),encoding="utf-8")
    p.wait(timeout=25)
else:
    s=importlib.util.spec_from_file_location("CPU_exact_ancestor_adapter",adapter);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    # Only source-path metadata is replaced for this CPU control. The actual
    # production no_live_trainers/helper bodies and exact Windows query run.
    m.__file__=str(Path(__file__).resolve())
    (root/"child-ready.json").write_text(json.dumps({"pid":os.getpid(),"parent_pid":os.getppid(),"argv":sys.argv,"Torch_imported":"torch" in sys.modules,"CUDA_called":False}),encoding="utf-8")
    deadline=time.monotonic()+15
    while not (root/"go.flag").exists() and time.monotonic()<deadline:time.sleep(.05)
    rows=[]
    for case in ("self-exact-ancestor","other-independent-evaluator"):
        d=root/case;d.mkdir()
        try:m.no_live_trainers(d,case);row={"case":case,"status":"allowed"}
        except Exception as ex:row={"case":case,"status":"rejected","error":str(ex)}
        rows.append(row)
        (root/(case+"-returned.flag")).write_text("actual CPU call returned",encoding="utf-8")
        if case=="self-exact-ancestor":
            deadline=time.monotonic()+15
            while not (root/"other-ready.flag").exists() and time.monotonic()<deadline:time.sleep(.05)
    (root/"child-results.json").write_text(json.dumps({"results":rows,"Torch_imported":"torch" in sys.modules,"CUDA_called":False}),encoding="utf-8")
