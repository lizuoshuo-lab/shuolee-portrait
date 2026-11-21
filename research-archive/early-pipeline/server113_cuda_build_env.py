"""Resolve CUDA development headers for both host C++ and nvcc native builds.

This only emits shell exports and optionally compiles a tiny C++ probe. It does
not install packages, alter the toolkit, load human data, or start training.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def unique_existing(paths):
    result = []
    seen = set()
    for p in paths:
        p = Path(p)
        if p.is_dir() and str(p.resolve()) not in seen:
            result.append(str(p.resolve()))
            seen.add(str(p.resolve()))
    return result


def resolve(toolkit, package_root):
    toolkit = Path(toolkit).resolve()
    if not (toolkit/'bin/nvcc').is_file():
        raise RuntimeError(f'CUDA compiler missing: {toolkit}/bin/nvcc')
    # Conda toolkits can put all development headers under targets/, whereas
    # the pip runtime package includes cuda_runtime.h without crt/host_config.h.
    # These ordered directories must reach the host C++ compiler as well as nvcc.
    includes = unique_existing([toolkit/'include', toolkit/'targets/x86_64-linux/include',
                                *sorted(Path(package_root).glob('*/include'))])
    header_sources = {}
    for name in ('cuda.h', 'cuda_runtime.h', 'crt/host_config.h'):
        found = next((str(Path(p)/name) for p in includes if (Path(p)/name).is_file()), None)
        if found is None:
            raise RuntimeError(f'CUDA development header missing: {name}; searched {includes}. '
                               'Keep the existing toolkit; provide this report instead of reinstalling blindly.')
        header_sources[name] = found
    libs = unique_existing([toolkit/'lib64', toolkit/'lib', toolkit/'targets/x86_64-linux/lib',
                            *sorted(Path(package_root).glob('*/lib'))])
    return dict(CUDA_HOME=str(toolkit), CPATH=':'.join(includes), LIBRARY_PATH=':'.join(libs)), header_sources


def host_probe(exports, output_dir):
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('g++ is missing.')
    run = Path(tempfile.mkdtemp(prefix='cuda-host-header-', dir=output_dir))
    source = run/'probe.cpp'
    source.write_text('#include <cuda.h>\n#include <cuda_runtime.h>\n'
                      'static_assert(CUDART_VERSION >= 12080 && CUDART_VERSION < 12090, "CUDA12.8 required");\n'
                      'int shuolee_cuda_header_probe() { return CUDART_VERSION; }\n')
    env = os.environ.copy()
    for key in ('C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'OBJC_INCLUDE_PATH'):
        env.pop(key, None)
    env.update(exports)
    command = [compiler, '-std=c++17', '-fPIC', '-c', str(source), '-o', str(run/'probe.o')]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    (run/'compiler.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Host C++ CUDA header probe failed; inspect {run}/compiler.log:\n{result.stderr}')
    return dict(command=command, returncode=0, object_bytes=(run/'probe.o').stat().st_size,
                log=str(run/'compiler.log'), gpu_executed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    toolkit = os.environ.get('CUDA_HOME')
    if not toolkit:
        raise RuntimeError('CUDA_HOME is missing; source ~/shuolee/cuda128.env first.')
    package_root = Path(sysconfig.get_paths()['purelib'])/'nvidia'
    exports, headers = resolve(toolkit, package_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = dict(header_sources=headers, exports=exports, training_started=False)
    if args.probe:
        report['host_cpp_probe'] = host_probe(exports, args.report.parent)
        print('CUDA_HOST_HEADERS_OK: host C++ compiled cuda_runtime.h and crt/host_config.h', file=sys.stderr)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    for key, value in exports.items():
        print('export '+key+'='+shlex.quote(value))
    # Do not allow a caller's CPLUS_INCLUDE_PATH to override this pinned header set.
    print('unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH')


if __name__ == '__main__':
    main()
