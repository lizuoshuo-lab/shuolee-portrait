"""Hold an existing Windows process handle and record its actual terminal exit."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PID = 32184
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splatam-ablation-repeat-001/executions/20261010T093555Z_b8bac22d/controller-recovery'
EXPECTED_TIME = datetime.fromisoformat('2026-10-10T10:50:25.834767+00:00')
EXPECTED_CMD = ['run_ablation_wsl.py', '--frames -1', '--width 320', '--sequence freiburg1_desk', '--seed 1', '--tracking-loss rgb_only']

def save_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

def main():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, PID)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        assert kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in times))
        ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        created = datetime.fromtimestamp((ticks - 116444736000000000) / 10000000, timezone.utc)
        assert abs((created - EXPECTED_TIME).total_seconds()) < .001, 'PID creation time differs'
        raw = subprocess.run(['powershell.exe', '-NoProfile', '-Command', f'Get-CimInstance Win32_Process -Filter "ProcessId={PID}" | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress'], capture_output=True, check=True)
        observed = json.loads(raw.stdout.decode('utf-8-sig'))
        assert observed['ProcessId'] == PID and all(value in observed['CommandLine'] for value in EXPECTED_CMD)
        OUT.mkdir(exist_ok=True)
        record = {'status': 'holding_original_wrapper_handle', 'pid': PID, 'process_creation_filetime_ticks': ticks,
                  'process_creation_utc': created.isoformat(), 'cim_observation': observed,
                  'run_name': 'freiburg1_desk_full_320x240_rgb_only_s1_20261010T105026Z_80268175',
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'handle_acquired_utc': datetime.now(timezone.utc).isoformat(), 'actual_exit_code': None,
                  'no_process_started_or_killed': True}
        save_new(OUT / 'original-wrapper-handle-started.json', record)
        print(json.dumps(record), flush=True)
        while True:
            wait_result = kernel.WaitForSingleObject(handle, 10000)
            if wait_result == 0:
                break
            assert wait_result == 258, f'Windows wait failed: {wait_result}'
        code = wintypes.DWORD()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        record.update(status='original_wrapper_terminal_exit_observed', actual_exit_code=code.value,
                      terminal_observed_utc=datetime.now(timezone.utc).isoformat())
        save_new(OUT / 'original-wrapper-handle-exit.json', record)
        print(json.dumps(record), flush=True)
    finally:
        kernel.CloseHandle(handle)

if __name__ == '__main__':
    main()
