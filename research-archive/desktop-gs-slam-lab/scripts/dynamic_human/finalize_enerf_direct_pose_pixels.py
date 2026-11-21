"""Wait for the actual065 evaluation, then audit and compare its saved pixels."""
from pathlib import Path
import subprocess,sys,time,argparse
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human';ap=argparse.ArgumentParser();ap.add_argument('--result-root',type=Path,default=LAB/'outputs/dynamic-human-enerf-direct-pose-065');ap.add_argument('--branch',default='direct_reprojection');ap.add_argument('--before-root',type=Path,default=LAB/'outputs/dynamic-human-enerf-articulated-control-064/estimated_offsets');ap.add_argument('--before-label',default='Capture-time body prior');ap.add_argument('--after-label',default='Direct pose refinement');args=ap.parse_args();R=args.result_root;A=args.before_root;B=R/args.branch
deadline=time.monotonic()+1200
while not (B/'native-evaluation/metrics.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Evaluation did not finish; preserve all outputs, inspect GPU chain')
 time.sleep(5)
commands=[
 [str(S/'audit_enerf_articulated_images.py'),'--branch',args.branch,'--result-root',str(R)],
 [str(S/'compare_enerf_articulated_control.py'),'--before-root',str(A),'--after-root',str(B),'--output-root',str(R),'--before-label',args.before_label,'--after-label',args.after_label],
 [str(S/'temporal_enerf_articulated_control.py'),'--before-root',str(A),'--after-root',str(B),'--output-root',str(R),'--before-label',args.before_label,'--after-label',args.after_label]
]
for command in commands:
 print('START',Path(command[0]).name,flush=True)
 subprocess.run([sys.executable,*command],check=True)
print('Actual saved-pixel audits and temporal comparisons complete; visual review remains',flush=True)
