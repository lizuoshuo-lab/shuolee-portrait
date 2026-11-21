"""Isolated Windows file-sharing controls; never opens an experiment progress file.

This demonstrates a possible replacement failure mechanism, not the historical
cause of any particular training failure. No optimizer, data loader or GPU.
"""
import ctypes
import datetime as dt
import hashlib
import json
import os
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "windows-progress-file-sharing-controls-002"


def main():
    if os.name != "nt":
        raise RuntimeError("These controls require Windows")
    OUT.mkdir(exist_ok=False)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    rows = []
    for label, share in [("no_external_handle", None),
                         ("reader_without_delete_sharing", 1 | 2),
                         ("reader_with_delete_sharing", 1 | 2 | 4)]:
        folder = OUT / label
        folder.mkdir()
        destination, candidate = folder / "progress.json", folder / "progress.json.tmp"
        old, new = b'{"version":"old"}\n', b'{"version":"new"}\n'
        destination.write_bytes(old)
        candidate.write_bytes(new)
        handle = None
        row = {"control": label, "share_flags": share, "read_access": "GENERIC_READ",
               "bytes_before": destination.read_bytes().decode("ascii")}
        try:
            if share is not None:
                handle = create(str(destination), 0x80000000, share, None, 3, 0x80, None)
                if handle == invalid:
                    raise ctypes.WinError(ctypes.get_last_error())
            try:
                os.replace(candidate, destination)
                row.update(replace_succeeded=True, error=None)
            except OSError as exc:
                row.update(replace_succeeded=False, error={"type": type(exc).__name__,
                           "winerror": getattr(exc, "winerror", None), "errno": exc.errno,
                           "message": str(exc)})
            row["bytes_after_attempt"] = destination.read_bytes().decode("ascii")
            row["candidate_remains_after_attempt"] = candidate.exists()
        finally:
            if handle is not None and handle != invalid and not close(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        if not row["replace_succeeded"]:
            os.replace(candidate, destination)
            row["same_candidate_succeeds_after_control_handle_closed"] = destination.read_bytes() == new
        else:
            row["same_candidate_succeeds_after_control_handle_closed"] = None
        row["final_bytes"] = destination.read_bytes().decode("ascii")
        rows.append(row)
    passed = (rows[0]["replace_succeeded"] and not rows[1]["replace_succeeded"]
              and rows[1]["bytes_after_attempt"] == old.decode("ascii")
              and rows[1]["candidate_remains_after_attempt"]
              and rows[1]["same_candidate_succeeds_after_control_handle_closed"])
    result = {"status": "passed" if passed else "failed", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows,
              "scope": "Small exclusive fixtures in this output directory only. The failure and reopening controls establish this Windows handle-sharing mechanism; they do not identify the handle owner or unique cause in the historical GS run. No historical file was opened, edited, replaced or retried."}
    (OUT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True))
    if not passed:
        raise RuntimeError("File-sharing controls did not demonstrate the proposed mechanism; checks retained")


if __name__ == "__main__":
    main()
