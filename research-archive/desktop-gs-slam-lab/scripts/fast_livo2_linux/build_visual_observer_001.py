"""Build an isolated, observation-only derivative; never modify the frozen replay workspace."""
from pathlib import Path
import hashlib, json, shutil, subprocess, traceback, importlib.util

LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
BASE=Path('/opt/gs-slam-lab/fast-livo2')
ROOT=Path('/opt/gs-slam-lab/fast-livo2-visual-observer-001')
OUT=LAB/'outputs/fast-livo2-visual-observer-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main():
    OUT.mkdir(exist_ok=False);ROOT.mkdir(exist_ok=False)
    script=LAB/'scripts/fast_livo2_linux/replay_hilti_gt.py'
    spec=importlib.util.spec_from_file_location('frozen',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    receipt=LAB/'outputs/fast-livo2-source-audit-001/repository-acquisition.json'
    override=LAB/'outputs/fast-livo2-shutdown-fix-001/source-override-receipt.json'
    before=mod.pinned_files(BASE/'catkin_ws/src/FAST-LIVO2',receipt,override)
    save(OUT/'baseline-source-before.json',before)
    package=ROOT/'catkin_ws/src/FAST-LIVO2';package.mkdir(parents=True)
    for row in before['files']:
        dest=package/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BASE/'catkin_ws/src/FAST-LIVO2'/row['path'],dest)
    shutil.copytree(BASE/'catkin_ws/src/rpg_vikit',ROOT/'catkin_ws/src/rpg_vikit')
    p=package/'src/vio.cpp';original=p.read_text();source=original
    marker='    if (error <= last_error)'
    assert source.count(marker)==2
    source=source.replace(marker,'''    // Observation only: residual belongs to the pre-step state at this iteration.
    const bool observer_accept = error <= last_error;
    printf("[GS_VIO_ITER] %d %d %d %d %d %.17g %.17g %d %.17g %.17g %.17g\\n", new_frame_->id_, level, iteration, total_points, n_meas, double(error), double(last_error), int(observer_accept), state->pos_end.x(), state->pos_end.y(), state->pos_end.z());
''' + marker)
    marker='    update_ekf_time += omp_get_wtime() - t3;'
    assert source.count(marker)==2
    source=source.replace(marker,'''    printf("[GS_VIO_STEP] %d %d %d %d %d %.17g %.17g %.17g\\n", new_frame_->id_, level, iteration, int(observer_accept), int(EKF_end), state->pos_end.x(), state->pos_end.y(), state->pos_end.z());
''' + marker)
    marker='  new_frame_.reset(new Frame(cam, img));'
    assert source.count(marker)==1
    source=source.replace(marker,marker+'\n  printf("[GS_VIO_FRAME] %d %.17g\\n", new_frame_->id_, img_time);')
    p.write_text(source)
    record=json.loads(override.read_text());record['scope']='Isolated clone; inherited allocator ABI fix plus printf-only VIO observations. Baseline source and executable untouched.'
    record['files'].append(dict(path='src/vio.cpp',original_sha256=sha(BASE/'catkin_ws/src/FAST-LIVO2/src/vio.cpp'),patched_sha256=sha(p),patched_bytes=p.stat().st_size))
    save(OUT/'source-override-receipt.json',record)
    import difflib
    (OUT/'observer.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='official/vio.cpp',tofile='observed/vio.cpp')))
    save(OUT/'plan.json',dict(status='registered_before_build',root=str(ROOT),observer='Residual and acceptance check before each update; resulting state after update or rollback; exact frame identifier and supplied image timestamp.',limits=['Independent replay, not the old historical run','Printing may change runtime scheduling; no throughput claim','No extra residual evaluation at the final untested state','No calibration, residual, optimizer or stopping-rule edits'],original_runner_sha256=sha(script)))
    with (OUT/'build.log').open('w') as log:
        cmd=['bash','-lc','source /opt/ros/noetic/setup.bash && catkin_make -j4 -l4']
        result=subprocess.run(cmd,cwd=ROOT/'catkin_ws',stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,'Isolated build failed; see build.log'
    libs=[ROOT/'catkin_ws/devel/lib'/f'lib{n}.so' for n in ('vio','lio','pre','imu_proc','laser_mapping')]
    binary=ROOT/'catkin_ws/devel/lib/fast_livo/fastlivo_mapping'
    save(OUT/'build-after-verification.json',dict(status='isolated_observer_build_completed',binary_after_sha256=sha(binary),libraries_after=[dict(path=str(p),sha256=sha(p)) for p in libs]))
    derived=script.read_text()
    old="lab/'outputs/fast-livo2-shutdown-fix-001/build-after-verification.json'"
    assert derived.count(old)==1
    derived=derived.replace(old,"lab/'outputs/fast-livo2-visual-observer-001/build-after-verification.json'")
    (OUT/'replay-observer.py').write_text(derived)
    assert mod.pinned_files(BASE/'catkin_ws/src/FAST-LIVO2',receipt,override)==before
    save(OUT/'build-status.json',dict(status='completed',baseline_source_unchanged=True,derived_runner_only_change='Build receipt path; all replay controls and gates retained'))
    print('Isolated visual observer compiled and baseline source unchanged.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
