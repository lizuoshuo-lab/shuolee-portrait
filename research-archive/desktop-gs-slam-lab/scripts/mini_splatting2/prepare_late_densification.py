"""Isolated native full-resolution densification intervention, no new loss terms."""
import ast
import difflib
import hashlib
import json
import shutil
import time
from pathlib import Path

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')

def main():
    source=RUNTIME/'repo-depth-cache';dest=RUNTIME/'repo-depth-late'
    assert not dest.exists()
    stage=ROOT/'native-late-runtime-001';stage.mkdir(exist_ok=False)
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','build'))
    path=dest/'msv2/train_depth_guided.py';before=path.read_text();text=before
    text=text.replace('from native_depth_guidance import DepthGuidance','from native_depth_guidance import DepthGuidance\nfrom late_topology_audit import audit_topology',1)
    text=text.replace('    first_iter = 0','    late_until = int(os.environ.get("MINIGS_LATE_DENSIFY_UNTIL", "0"))\n    first_iter = 0',1)
    old='            if iteration < opt.densify_until_iter:'
    new='            late_active = args.simp_iteration1 < iteration < late_until\n            if iteration < opt.densify_until_iter or late_active:'
    assert text.count(old)==1;text=text.replace(old,new,1)
    old='                if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:'
    new='                if iteration < opt.densify_until_iter and iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:'
    assert text.count(old)==1;text=text.replace(old,new,1)
    old='                gaussians.extend_features_rest()\n\n                gaussians.training_setup(opt)'
    new='                gaussians.extend_features_rest()\n                if late_until > args.simp_iteration1:\n                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")\n\n                gaussians.training_setup(opt)'
    assert text.count(old)==1;text=text.replace(old,new,1)
    anchor='            # Optimizer step'
    audit='''            if late_until > args.simp_iteration1 and (iteration == args.simp_iteration1 or
                    args.simp_iteration1 < iteration <= late_until + 100 and iteration % 100 == 0):
                audit_topology(gaussians, mask_blur, iteration, viewpoint_cam, dataset.model_path, loss)

'''
    text=text.replace(anchor,audit+anchor,1)
    path.write_text(text);ast.parse(text)
    # A neutral gate must recover the original statements, not a reimplemented loss.
    neutral=text.replace('from late_topology_audit import audit_topology\n','')
    neutral=neutral.replace('    late_until = int(os.environ.get("MINIGS_LATE_DENSIFY_UNTIL", "0"))\n','')
    neutral=neutral.replace('            late_active = args.simp_iteration1 < iteration < late_until\n            if iteration < opt.densify_until_iter or late_active:', '            if iteration < opt.densify_until_iter:')
    neutral=neutral.replace('                if iteration < opt.densify_until_iter and iteration >= args.aggressive_clone_from_iter','                if iteration >= args.aggressive_clone_from_iter')
    neutral=neutral.replace('                if late_until > args.simp_iteration1:\n                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")\n','')
    neutral=neutral.replace(audit,'')
    assert ast.dump(ast.parse(neutral),include_attributes=False)==ast.dump(ast.parse(before),include_attributes=False)
    lo=before.index('        # Loss');hi=before.index('        iter_end.record()')
    assert before[lo:hi]==text[text.index('        # Loss'):text.index('        iter_end.record()')]
    helper=Path(__file__).with_name('late_topology_audit.py');shutil.copy2(helper,dest/helper.name)
    (stage/'training.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='repo-depth-cache/train',tofile='repo-depth-late/train')))
    result=dict(status='prepared_pending_topology_smoke',created=time.time(),repository=str(dest),
        neutral_ast_equivalent=True,joint_loss_and_backward_byte_identical=True,
        unchanged_files={p:hashlib.sha256((dest/p).read_bytes()).hexdigest() for p in ['native_depth_guidance.py','scene/gaussian_model.py','gaussian_renderer/__init__.py','native_mip_filter.py','config/fast']},
        changed_files={p:hashlib.sha256((dest/p).read_bytes()).hexdigest() for p in ['msv2/train_depth_guided.py','late_topology_audit.py']},
        intervention='Ordinary gradient clone/split/prune after first simplification,3000<iteration<5000; no extension of aggressive cloning',
        caveat='Includes native large-footprint/opacity pruning; this is a topology schedule intervention, not a pure point-count increase',
        baseline='native-resolution-cache-full-18000-s0',runtime_gradient_equivalence_tested=False,
        added_losses=[],requires_smoke=True)
    for p,digest in result['unchanged_files'].items():assert hashlib.sha256((source/p).read_bytes()).hexdigest()==digest
    (stage/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
