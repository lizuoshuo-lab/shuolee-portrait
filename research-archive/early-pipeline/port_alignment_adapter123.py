"""Port the reviewed candidate without changing the frozen runtime or old fitter."""
from pathlib import Path
import hashlib, json
H=Path(__file__).parent
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
S=P/'src/shuolee_portrait'
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=R/'generic-2d3d-research/adapter-a0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def change(text, old, new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new)
before={str(p.relative_to(P)):sha(p) for p in (P/'src').rglob('*.py') if p.name!='alignment_inputs.py'}
assert not (O/'before-source-hashes.json').exists()
(O/'before-source-hashes.json').write_text(json.dumps(before,indent=2)+'\n',encoding='utf-8')

gate=(H/'local_surface123_alignment_gate.py').read_text(encoding='utf-8')
start=gate.index('def input_path(');end=gate.index('\n\ndef evaluate(',start)
gate=gate[:start]+'''def input_path(root, value):
    """Portable manifest paths; no workstation-specific alias guessing."""
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path
'''+gate[end:]
(S/'reviewed_admission.py').write_text(gate,encoding='utf-8',newline='\n')

core=(H/'smplx_direct_correspondence123.py').read_text(encoding='utf-8')
core=core[:core.index('\ndef load_reviewed_face(')]
core=change(core,"issubset({'FIT', 'HELD'})","issubset({'FIT', 'HELD', 'UNASSIGNED'})")
core=change(core,"        tensor = lambda x:","        if np.any(np.asarray(eligible) & (np.asarray(split) == 'UNASSIGNED')):\n            raise ValueError('Unassigned observations cannot be eligible.')\n        tensor = lambda x:")
(S/'smplx_correspondence.py').write_text(core,encoding='utf-8',newline='\n')

source=R/'direct-smplx-bindings/candidate_fitting.py'
assert sha(source)=='b3b325b2d91addc74cada8ee85ba8fa37b6f6d829fddd7910f76245f908f523c'
code=source.read_text(encoding='utf-8')
code=change(code,'from smplx_direct_correspondence123 import load_reviewed_face, face_world\n','')
code=change(code,'torch=c.runtime()\n','')
code=change(code,'def main(root,steps):\n', '''def _fit_prepared(root,steps,bundle):
    # Called only by fit_backend.run after the hash-bound CPU preflight and
    # unchanged physical camera admission. No dataset-specific dispatch.
    from .smplx_correspondence import face_world, project_integer_centres
    bundle.assert_unchanged()
    torch=c.runtime()
''')
code=change(code,"rows=c.read(root/'data/inputs.json')['records'];n=len(rows)","rows=bundle.rows;n=len(rows)")
code=change(code,"obs=load(root/'data/observations.npz');pred=load(root/'inference/prediction-postprocessed.npz')","obs=bundle.body;pred=load(root/'inference/prediction-postprocessed.npz')")
code=change(code,"E=np.array([r['w2c'] for r in rows]);K=np.array([r['K'] for r in rows]);times=np.array([r['time_seconds'] for r in rows])","E=np.stack([v[1] for v in bundle.cameras]);Ki=np.stack([v[0] for v in bundle.cameras]);K=np.stack([v[2] for v in bundle.cameras]);times=np.array([r['time_seconds'] for r in rows])")
code=change(code,"direct_face,binding_review=load_reviewed_face(root,rows,obs,model,device='cuda')","direct_face,binding_review=bundle.make_face_objective(model,device='cuda')")
code=change(code,"target=g(obs['keypoints']);eg,kg=g(E),g(K);dt=g(np.diff(times));R0g,t0g=g(R0),g(t0)","target=g(obs['keypoints']);eg,kg,kig=g(E),g(K),g(Ki);dt=g(np.diff(times));R0g,t0g=g(R0),g(t0)")
code=change(code,"        cam=x@eg[:,:3,:3].transpose(1,2)+eg[:,:3,3][:,None];q=cam@kg.transpose(1,2)\n        return q[...,:2]/q[...,2:].clamp_min(.01)-.5,cam", "        uv,_=project_integer_centres(x,eg,kig)\n        cam=x@eg[:,:3,:3].transpose(1,2)+eg[:,:3,3][:,None]\n        return uv,cam")
code=change(code,"            clip=torch.stack((2*k[0,0]/w*v[:,0]+(2*k[0,2]/w-1)*z,-2*k[1,1]/himg*v[:,1]+(1-2*k[1,2]/himg)*z,1.00002*z-.0200002,z),-1)","            from .smplx_correspondence import camera_clip_coordinates\n            clip=camera_clip_coordinates(v,k,w,himg)")
code=change(code,"observations_sha256=c.sha(root/'data/observations.npz'),input_sha256=c.sha(root/'data/inputs.json'),","observations_sha256=bundle.manifest['wholebody']['sha256'],input_sha256=bundle.manifest['frames']['sha256'],\n        alignment_manifest_sha256=bundle.report['manifest_sha256'],face_observations_sha256=bundle.manifest['face']['sha256'],")
code=change(code,"    c.save(out/'history.json',history)","    bundle.assert_unchanged()\n    c.save(out/'history.json',history)")
(S/'fitting_direct.py').write_text(code,encoding='utf-8',newline='\n')
print(json.dumps({'new_modules':[str(S/f) for f in ['reviewed_admission.py','smplx_correspondence.py','fitting_direct.py']], 'old_fitter_unchanged':sha(S/'fitting.py')==before[str(Path('src/shuolee_portrait/fitting.py'))]}))
