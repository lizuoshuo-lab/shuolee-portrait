"""Versioned CPU preparation: preserve completed00, adopt live01, use verifiedPS7."""
from pathlib import Path
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    scripts = ROOT / 'scripts/large_scene_gs'
    pwsh = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
    child_env = os.environ.copy()
    child_env['PSModulePath'] = os.pathsep.join([str(pwsh.parent / 'Modules'), str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')])
    command = [str(pwsh), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
               r"[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $taskCommand=Get-Command Get-FileHash -ErrorAction Stop; [pscustomobject]@{version=$PSVersionTable.PSVersion.ToString();command=$taskCommand.Name;source=$taskCommand.Source;coarseOutcomeSha=(Get-FileHash -LiteralPath 'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\large-scene-gs-001\official-training\small-city-coarse-i30000-r4-003\outcome.json' -Algorithm SHA256).Hash;pythonCount=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' }).Count} | ConvertTo-Json -Compress"]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', env=child_env)
    directory = ROOT / 'outputs/large-scene-gs-001/official-training-pipeline-review/small-city-four-chunks-005'
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'powershell7-runtime-verification.json'
    if report_path.exists():
        raise ValueError('Existing PS7 verification; refusing overwrite')
    report = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'verified_runtime': str(pwsh),
              'child_only_PSModulePath': child_env['PSModulePath'], 'actual_exit_code': result.returncode,
              'actual_stdout': result.stdout, 'actual_stderr': result.stderr,
              'system_or_user_environment_modified': False, 'CUDA_initialized': False,
              'previous_WindowsPowerShell5_failure_preserved': True}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if result.returncode:
        raise RuntimeError('PinnedPS7 command compatibility verification failed')
    m = (scripts / 'run_small_city_full_pipeline_004.py').read_text(encoding='utf-8')
    m = m.replace('run_small_city_full_pipeline_004.py', 'run_small_city_full_pipeline_005.py').replace('small-city-four-chunks-004', 'small-city-four-chunks-005')
    m = m.replace("POWERSHELL = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'", f"POWERSHELL = Path(r'{pwsh}')\nCHILD_POWERSHELL_ENV = os.environ.copy()\nCHILD_POWERSHELL_ENV['PSModulePath'] = os.pathsep.join([str(POWERSHELL.parent / 'Modules'), str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')])")
    m = m.replace("cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)", "cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, env=CHILD_POWERSHELL_ENV)", 1)
    m = m.replace("'first00_adopted_without_restart': True", "'completed00_preserved_without_restart': True, 'existing01_adopted_without_restart': True")
    m = m.replace("if index > 0:", "if index > 1:", 1)
    m = m.replace("            if index == 0:\n                adopted_sup_path", "            if index == 1:\n                adopted_sup_path", 1)
    m = m.replace('adopted00-strong-binding-004.json', 'adopted01-strong-binding-005.json')
    m = m.replace("'Already completed00 actual final receipt; no restart'", "'Already completed01 actual final receipt; no restart'")
    marker = "            if index > 1:\n"
    seed = """            if index == 0:
                provenance = binding_module.completed_gate(run, 'chunk', chunk)
                audit_path = BASE / 'recovery-audit' / f'{run}-completion-004' / 'completion-audit.json'
                if sha(audit_path) != 'f108f64b88e8e9a136c86e9d110fc0f274a3cfe5681b30988a42c4f9f091b183':
                    raise ValueError('Previously completed00 physical audit bytes changed')
                audit = read(audit_path)
                if audit['model_parameter_sha256'] != provenance['model_parameter_sha256'] or not all(g['float32_bitexact'] for g in audit['shared_sky_all_parameter_groups_bitexact'].values()):
                    raise ValueError('Completed00 physical/source provenance differs')
                save(directory / 'preserved-completed00-fresh005-gate.json', provenance)
                state['completed_chunks'].append({'chunk': chunk, 'run': run, 'audit': str(audit_path), 'audit_sha256': sha(audit_path),
                    'model_sha256': audit['model_parameter_sha256'], 'G': audit['actual_saved_gaussians'],
                    'logged_torch_peak_allocated_bytes': audit['logged_torch_peak_allocated_bytes'], 'existing_completed_model_preserved': True})
                save(directory / 'state.json', state)
                continue
"""
    if m.count(marker) != 1:
        raise AssertionError('Controller revision marker mismatch')
    m = m.replace(marker, seed + marker, 1)
    target = scripts / 'run_small_city_full_pipeline_005.py'
    if target.exists():
        raise ValueError('Existing005 controller; refusing overwrite')
    target.write_text(m, encoding='utf-8')
    launcher = (scripts / 'launch_small_city_full_pipeline_004.ps1').read_text(encoding='utf-8')
    launcher = launcher.replace('run_small_city_full_pipeline_004.py', 'run_small_city_full_pipeline_005.py').replace('small-city-four-chunks-004', 'small-city-four-chunks-005')
    launcher = launcher.replace('first00_adopted_without_restart = $true', 'existing01_adopted_without_restart = $true\n    completed00_preserved_without_restart = $true')
    target = scripts / 'launch_small_city_full_pipeline_005.ps1'
    if target.exists():
        raise ValueError('Existing005 launcher; refusing overwrite')
    target.write_text(launcher, encoding='utf-8')
    print(json.dumps({'PS7_compatibility': 'passed', 'report': str(report_path), 'created005': True, 'previous003_004_sources_modified': False}))


if __name__ == '__main__':
    main()
