"""CPU reproduction of WindowsPowerShell5 command resolution and local repair."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    shell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    command = [str(shell), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
               '[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-Command Get-FileHash -ErrorAction Stop | Select-Object Name,Source | ConvertTo-Json -Compress']
    baseline = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    local = os.environ.copy()
    local['PSModulePath'] = str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')
    corrected = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', env=local)
    report = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'baseline_PSModulePath': os.environ.get('PSModulePath'),
              'baseline_exit_code': baseline.returncode, 'baseline_stdout': baseline.stdout, 'baseline_stderr': baseline.stderr,
              'process_local_PSModulePath': local['PSModulePath'], 'corrected_exit_code': corrected.returncode,
              'corrected_stdout': corrected.stdout, 'corrected_stderr': corrected.stderr,
              'scope': 'Process-local child environment only. System/user environment unchanged. No CUDA or training launched.'}
    directory = ROOT / 'outputs/large-scene-gs-001/official-training-pipeline-review/small-city-four-chunks-005'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'powershell-module-diagnosis.json'
    if target.exists():
        raise ValueError('Existing diagnosis; refusing overwrite')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if corrected.returncode != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
