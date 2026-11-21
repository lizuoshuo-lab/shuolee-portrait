"""CPU-only exact Windows argv classification for local stage process checks.

Matches script argument names rather than source text inside Python -c command
arguments. Windows venv shim and underlying interpreter can both be reported.
"""
import ctypes
from ctypes import wintypes
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def windows_argv(command):
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel.LocalFree.restype = wintypes.HLOCAL
    count = ctypes.c_int()
    values = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not values:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [values[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(ctypes.cast(values, wintypes.HLOCAL))


def observe(run):
    spec = importlib.util.spec_from_file_location('read_only_supervisor', ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual, related = module.python_processes(), []
    for process in actual:
        argv = windows_argv(process.get('CommandLine') or '')
        names = {Path(argument).name.lower() for argument in argv[1:]}
        is_trainer = 'train_coarse.py' in names
        is_wrapper = bool(names & {'supervise_small_city_stage.py', 'run_small_city_stage.py'}) and run in argv
        if is_trainer or is_wrapper:
            related.append(dict(process, parsed_argv=argv, role='official_coarse_trainer' if is_trainer else 'stage_wrapper_or_supervisor'))
    return {'actual_python_processes': actual, 'related_training_processes': related,
            'scope': 'Exact Windows argv script-name classification; does not match literal source text inside Python -c arguments.'}
