from pathlib import Path
import ast,hashlib,json,datetime
L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
G=L/'repos/GaussianAvatars_SMPLX-shuolee'
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
old=G/'audit_named_point_bundle_099.py';new=G/'audit_named_point_bundle_101.py'
assert not new.exists() and not (B/'current/shared-point-bundle-convergence-101').exists()
t=old.read_text(encoding='utf-8')
t=t.replace('import pycolmap',"sys.path.insert(0,'/opt/gs-slam-lab/shuolee-bundle-audit/pyceres-2.3')\nimport pyceres\nimport pycolmap")
t=t.replace("O=c.BASE/'current/shared-point-ray-consistency-099/joint-camera-points'","O=c.BASE/'current/shared-point-bundle-convergence-101'")
t=t.replace("budget='Native default iteration limit; 240 wall seconds per arm. A failed arm is recorded and no budget increase is automatic.',", """budget='Unchanged100iteration maximum and240wallseconds per arm. No automatic extension.',
      solver_termination=dict(function_tolerance=1e-6,gradient_tolerance=1e-10,parameter_tolerance=1e-8,max_num_iterations=100),
      evidence='100runtime audit found all three native COLMAP stopping tolerances equal0. Ceres option binding requires explicitly imported pyceres; use isolated2.3 without changing training packages.',
      new_variable_relative_to_099='Enable explicit standard Ceres stopping tolerances only. Same points, observations, cameras, losses, native100iterationbudget and acceptance thresholds. Rebuild every point from raw2D observations.',
      reference_099='Existing results are comparison only; no reused optimized cameras/points.',""")
needle="    c.save(O/(arm+'-start.json'),dict(created=c.stamp(),options=opt.summary(),scene_points=nscene,named_points=len(allx)-nscene,fit_observations=int(fit.sum()),pid=os.getpid()))"
replacement="""    opt.solver_options.function_tolerance=1e-6
    opt.solver_options.gradient_tolerance=1e-10
    opt.solver_options.parameter_tolerance=1e-8
    assert opt.solver_options.max_num_iterations==100
    opt.solver_options.minimizer_progress_to_stdout=True
    c.save(O/(arm+'-start.json'),dict(created=c.stamp(),options=opt.summary(),solver_options=opt.solver_options.summary(),scene_points=nscene,named_points=len(allx)-nscene,fit_observations=int(fit.sum()),pid=os.getpid()))"""
assert needle in t;t=t.replace(needle,replacement)
ast.parse(t);new.write_text(t,encoding='utf-8',newline='\n')
receipt=dict(created=datetime.datetime.now().astimezone().isoformat(),source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
 candidate=str(new),candidate_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),same100iterationbudget=True,
 changed='Explicit Ceres termination tolerances; fresh3arm replay; native periteration logging for checking actual stop.',
 source_cameras_unchanged=True,gaussian_training_started=False)
(Path(__file__).parent/'shared-bundle-101-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
