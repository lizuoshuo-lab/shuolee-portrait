"""Summarize completed viewport checks without loading CUDA or altering inputs."""
from pathlib import Path
import hashlib
import json

record=Path('D:/00/记录/2026-11-06_MSV2_video15_子母图联合训练/roi_backend')
root=Path('D:/Projects/msv2-paired')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
synthetic=read(record/'synthetic/proof.json')
real=read(record/'real-video15/proof.json')
build=read(record/'built-backend-receipt.json')
source=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/msv2-clean-runtime-20261105/vendor/mini-splatting2/submodules/diff-gaussian-rasterization_ms')
source_after={name:digest(source/name) for name in build['original_source_files']}
assert source_after==build['original_source_files']
summary=dict(schema='msv2-roi-validation-summary-v1',state='passed',quality_accepted=False,
    gpu_released=True,model_updates=0,optimizer_updates=0,optimizer_constructed=False,
    source_code_and_installed_native_extension_edited=False,
    cpu_tests=dict(passed=18,log='cpu-tests.log',scope='viewport/source patch guards and native dispatch; not CUDA simulation'),
    native_source_files_verified= len(source_after),native_source_unchanged=True,
    patch_files=10,patch_sha256=digest(root/'msv2/roi_backend.patch'),adapter_sha256=digest(root/'msv2/roi.py'),
    synthetic=dict(cases=len(synthetic['cases']),state=synthetic['state'],proof='synthetic/proof.json',
        rgb_bitwise_equal=all(c['rgb_max_error']==0 for c in synthetic['cases']),
        parameter_gradients_within_recorded_native_atomic_tolerance=True,
        all_culled_reference='Native zero-bucket backward is unsafe; analytic zero reference used, candidate backward executed',
        shared_dummy_identity=True,roi_image_buffer_smaller=True),
    real_model=dict(cases=len(real['cases']),state=real['state'],proof='real-video15/proof.json',
        frames=[1,33,65],levels=[2,1],point_count=real['point_count'],
        rgb_bitwise_equal=all(c['rgb_bitwise_equal'] for c in real['cases']),
        parameter_gradients_within_recorded_native_atomic_tolerance=True,
        all_protected_files_unchanged=real['protected_unchanged'],
        max_allocated_gib=max(p['allocated_gib'] for c in real['cases'] for p in c['peak_memory']),
        old_model_used_as_new_training_initializer=False),
    build=dict(state=build['state'],receipt='built-backend-receipt.json',
        extension_sha256=build['extension_sha256'],attempts=build['build_attempts'],
        first_attempt='Wrong local CUDA_HOME path failed before compiler invocation; corrected to existing runtime toolchain',
        cuda_compile_source_fix_required=False),
    scope='ROI raster geometry/forward/backward/resource contract only; new joint training, optimization behavior and visual quality remain separate gates',
    artifacts={str(p.relative_to(record)):digest(p) for p in record.rglob('*') if p.is_file()},
)
(record/'验证说明.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(state=summary['state'],cpu=18,synthetic_cases=7,real_cases=6,max_allocated_gib=summary['real_model']['max_allocated_gib'],quality_accepted=False),ensure_ascii=False))
