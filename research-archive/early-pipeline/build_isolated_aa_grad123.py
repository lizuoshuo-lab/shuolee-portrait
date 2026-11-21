"""Isolated original-vs-direct atomic AA backward build. No runtime replacement."""
from pathlib import Path
import difflib,os,shutil,time,sys
import run_official_gaussians123 as io
from probe_native_mesh_mask123 import OUT

SOURCE=Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/install/native-lt2pb6wk/nvdiffrast')
WORK=Path('/opt/gs-slam-lab/shuolee-aa-gradient123')
OUT=OUT/'isolated-atomic-control'
CUDA=Path('/opt/gs-slam-lab/splatam/cuda-extracted/usr/local/cuda-12.8')


def main():
    repair='--complete-build-context' in sys.argv
    if repair:
        assert WORK.exists() and (OUT/'plan.json').exists() and not (OUT/'build-result.json').exists()
        assert not (OUT/'build-context-repair.json').exists()
        io.save(OUT/'build-context-repair.json',dict(updated=io.now(),
            failure='Compile-only first attempt lacked installed CUDA library development include paths: cusparse.h.',
            correction='Add existing environment nvidia/*/include directories to both variants; no package downloads or kernel logic changes.',
            script_sha256=io.sha(__file__)))
    else:
        assert not WORK.exists() and not OUT.exists()
        WORK.mkdir();OUT.mkdir()
    assert (CUDA/'bin/nvcc').is_file()
    os.environ.update(CUDA_HOME=str(CUDA),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='12.0')
    os.environ['PATH']=str(CUDA/'bin')+os.pathsep+os.environ['PATH']
    from torch.utils.cpp_extension import load
    import torch
    original=(SOURCE/'csrc/common/antialias.cu').read_text()
    patched=original
    for vertex in (1,2):
        old=f'caAtomicAdd3_xyw(p.gradPos + 4 * vi{vertex}, gp{vertex}x, gp{vertex}y, gp{vertex}w);'
        new='\n        '.join(f'atomicAdd(p.gradPos + 4 * vi{vertex} + {index}, gp{vertex}{component});'
                              for index,component in [(0,'x'),(1,'y'),(3,'w')])
        assert patched.count(old)==1
        patched=patched.replace(old,new)
    host=(SOURCE/'csrc/torch/torch_antialias.cpp').read_text()
    body=host[host.index('std::tuple<torch::Tensor, torch::Tensor> antialias_grad'):]
    prefix=host[:host.index('#include')]
    cpp=(prefix+'#include "torch_common.inl"\n#include "../common/common.h"\n#include "../common/antialias.h"\n'
        +'void AntialiasGradKernel(const AntialiasKernelParams p);\n'+body
        +'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("antialias_grad", &antialias_grad); }\n')
    diff=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='original/antialias.cu',tofile='direct-atomic/antialias.cu'))
    (OUT/'two-call-atomic-change.diff').write_text(diff)
    source_hashes={str(p):io.sha(p) for p in SOURCE.joinpath('csrc').rglob('*') if p.is_file()}
    plan=dict(created=io.now(),hypothesis='The position-gradient coalesced atomic accumulation loses updates on this installed CUDA/GPU combination. Replace only its two3-component accumulation calls with direct global atomicAdd; equations/forward/raster/thresholds unchanged.',
        source= str(SOURCE),source_hashes=source_hashes,script_sha256=io.sha(__file__),
        variants=['recompiled unchanged accumulation','recompiled direct atomic accumulation'],
        max_build_seconds=300,workers=2,cuda_arch='12.0',cuda_home=str(CUDA),
        no_installed_file_modification=True,no_live_source_modification=True,no_gpu_fitting=True,
        validation='Same synthetic triangle, fullK, fixed-sign objective, epsilon1e-4, six repeats. Compare installed, recompiled original and corrected derivative before any59-frame test.',quality_accepted=False)
    if not repair:io.save(OUT/'plan.json',plan)
    include_dirs=sorted(str(p) for p in Path(sys.prefix).glob('lib/python3.10/site-packages/nvidia/*/include'))
    assert any(Path(p,'cusparse.h').is_file() for p in include_dirs)
    started=time.monotonic();builds=[]
    for name,cu in [('original',original),('direct',patched)]:
        root=WORK/name
        if not (root/'csrc').exists():shutil.copytree(SOURCE/'csrc',root/'csrc')
        (root/'csrc/common/antialias.cu').write_text(cu)
        (root/'csrc/torch/isolated_grad.cpp').write_text(cpp)
        build=root/'build';build.mkdir(exist_ok=True)
        print('BUILD_AA',name,flush=True)
        module=load(name='shuolee_aa_grad123_'+name,
            sources=[str(root/'csrc/common/antialias.cu'),str(root/'csrc/torch/isolated_grad.cpp')],
            extra_cflags=['-O2','-DNVDR_TORCH'],extra_cuda_cflags=['-O2','-DNVDR_TORCH','-lineinfo'],
            extra_include_paths=include_dirs,
            build_directory=str(build),verbose=True)
        assert time.monotonic()-started<plan['max_build_seconds']
        builds.append(dict(name=name,module=str(module.__file__),sha256=io.sha(module.__file__),
            cuda_source_sha256=io.sha(root/'csrc/common/antialias.cu'),host_source_sha256=io.sha(root/'csrc/torch/isolated_grad.cpp')))
        io.save(OUT/'build-progress.json',dict(updated=io.now(),completed=builds,elapsed=time.monotonic()-started))
    for p,h in source_hashes.items():assert io.sha(p)==h,p
    assert not torch.cuda.is_initialized()
    io.save(OUT/'build-result.json',dict(updated=io.now(),state='isolated_backward_variants_built',builds=builds,
        elapsed_seconds=time.monotonic()-started,cuda_context_created=False,production_modified=False,quality_accepted=False))
    print('ISOLATED_AA_BUILD_COMPLETE',flush=True)


if __name__=='__main__':main()
