"""Regression checks for the real training-entry defect; no GPU or human weights.

These checks prove rejection before allocation, not successful alignment.
The development release intentionally has no positive admission path yet.
"""
import argparse
import builtins
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.admission import AlignmentNotAdmitted, inspect_alignment, require_alignment
from shuolee_portrait import cli, training


def check():
    cases=[]
    with tempfile.TemporaryDirectory(prefix='shuolee-admission-check-') as temporary:
        root=Path(temporary)
        (root/'fit').mkdir();(root/'fit/fitted.npz').write_bytes(b'synthetic-fit-identity-only')
        rt.save(root/'run.json',dict(video_sha256='0'*64,from_scratch=True,old_subject_loaded=False))
        rt.save(root/'config.json',dict(quality_accepted=False))
        rt.save(root/'data/training-inputs.json',dict(records=[dict(frame=f'frame_{i:06d}') for i in range(3)]))
        rt.save(root/'fit/geometry-gate.json',dict(coarse_initialization_gate_passed=True,fit_sha256=rt.sha(root/'fit/fitted.npz')))
        rt.save(root/'inference/crop-mapping.json',dict(records=[]))
        before={p.relative_to(root).as_posix():rt.sha(p) for p in root.rglob('*') if p.is_file()}
        report=inspect_alignment(root)
        assert report['coarse_fit']['passed'] and report['coarse_fit']['matches_current_fit']
        assert not report['training_admitted'] and not report['independent_alignment_validation_implemented']
        assert before=={p.relative_to(root).as_posix():rt.sha(p) for p in root.rglob('*') if p.is_file()}
        cases.append('Coarse fit pass does not admit training; inspection is read-only.')

        # A prior pass file is an output artifact, never permission.
        rt.save(root/'alignment/training-readiness.json',dict(training_admitted=True,quality_accepted=True))
        try:require_alignment(root)
        except AlignmentNotAdmitted:pass
        else:raise AssertionError('Stale approval admitted training')
        assert rt.read(root/'alignment/training-readiness.json')['training_admitted'] is False
        cases.append('A hand-written or stale pass report cannot bypass the boundary.')

        real_import=builtins.__import__
        def no_cuda(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','scene','gaussian_renderer'):
                raise AssertionError(f'GPU/model import happened before rejection: {name}')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=no_cuda),patch.object(rt,'configure',side_effect=AssertionError('Assets configured before rejection')):
            for invoke in (lambda:cli.stage('alignment-gate',root,root/'missing-assets'),
                           lambda:cli.stage('train',root,root/'missing-assets'),
                           lambda:training.train(root,{})):
                try:invoke()
                except AlignmentNotAdmitted:pass
                else:raise AssertionError('Entry did not reject missing alignment')
        assert not (root/'gaussian-run').exists()
        cases.append('Stage and direct training entries reject before GPU imports, assets or Gaussian allocation.')

        command=[sys.executable,str(PROJECT/'run.py'),'_stage','--stage','train','--output',str(root),'--assets',str(root/'missing-assets')]
        proc=subprocess.run(command,capture_output=True,text=True)
        assert proc.returncode==3 and 'No Gaussian model or optimizer' in proc.stderr,(proc.returncode,proc.stderr)
        status=subprocess.run([sys.executable,str(PROJECT/'run.py'),'alignment-status','--output',str(root)],capture_output=True,text=True)
        assert status.returncode==3 and json.loads(status.stdout)['training_admitted'] is False
        cases.append('Actual CLI returns exit3 for unavailable alignment; no body weights or CUDA needed.')

        # Exercise the real parent-stage dispatch and status handling. Only the
        # expensive preprocessing stages are simulated, never alignment/train.
        video=root/'synthetic-input.mp4';video.write_bytes(b'not-decoded-in-this-negative-path-test')
        output=root/'new-parent-run';called=[];real_run=subprocess.run
        def stages(command,*args,**kwargs):
            name=command[command.index('--stage')+1];called.append(name)
            if name=='alignment-gate':return real_run(command,*args,**kwargs)
            if name in ('train','export'):raise AssertionError('Parent dispatched training after missing alignment')
            if name=='fit-gate':
                (output/'fit').mkdir();(output/'fit/fitted.npz').write_bytes(b'coarse-pass-fixture')
                rt.save(output/'fit/geometry-gate.json',dict(coarse_initialization_gate_passed=True,fit_sha256=rt.sha(output/'fit/fitted.npz')))
                rt.save(output/'data/training-inputs.json',dict(records=[]))
            return subprocess.CompletedProcess(command,0)
        argv=['run.py','reconstruct','--video',str(video),'--assets',str(root/'missing-assets'),'--output',str(output),'--config',str(PROJECT/'configs/smoke.json')]
        with patch.object(sys,'argv',argv),patch.object(cli,'doctor',return_value={}),patch.object(cli,'gpu_lock',return_value=nullcontext()),patch.object(cli.shutil,'which',return_value='test-executable'),patch.object(cli.subprocess,'run',side_effect=stages),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            try:cli.main()
            except SystemExit as exc:assert exc.code==3
            else:raise AssertionError('Full entry reported completion without alignment')
        parent=rt.read(output/'status.json')
        assert parent['state']=='alignment_not_admitted' and parent['training_started'] is False
        assert called==list(cli.STAGES[:cli.STAGES.index('train')]) and not (output/'gaussian-run').exists()
        cases.append('Parent reconstruct stops at alignment-gate with an explicit non-training status; preprocessing was simulated only for this test.')

        (root/'fit/fitted.npz').write_bytes(b'changed-fit')
        changed=inspect_alignment(root)
        assert changed['coarse_fit']['passed'] and not changed['coarse_fit']['matches_current_fit']
        assert not changed['training_admitted']
        cases.append('A changed fitted artifact invalidates its recorded coarse-fit identity.')
    return dict(passed=True,cases=cases,scope='Negative-path training admission only',
        successful_alignment_verified=False,independent_alignment_validation_implemented=False,
        training_started=False,gpu_task=False,quality_accepted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=check();rt.save(args.output,result)
    print(json.dumps(result,indent=2))
