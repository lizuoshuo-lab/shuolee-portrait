"""Reuse the already verified prefit selector for additional native datasets."""
from pathlib import Path
import prepare_winterhat_sh0124 as original
from run_official_gaussians123 import read,save,sha,now

PEOPLE=[
    ('whiteT','室外白T人物','002','0a4d31cfcf31036ef2244b2709daaf29b425c1921c5516183302670728d030de'),
    ('grayT','室外灰T人物','003','7b1c0fd30822c3de879bd6f8649d63175d4d7fa9e2484f0e76d3c763452d79bb')]

def main():
    for subject,label,record,digest in PEOPLE:
        original.ROOT=original.REC/'cross-subject-sh0-124'/f'{subject}-input'
        original.DATA=Path('/opt/gs-slam-lab/shuolee-local-surface-123')/subject
        original.VIDEO=Path('/mnt/d/00/工作/数据集')/f'video{int(record)}.mp4'
        original.EXPECTED=digest
        # Only file/subject inputs change; the deterministic selection formula
        # and source image verification execute verbatim from the first intake.
        original.prepare()
        p=read(original.ROOT/'plan.json')
        p.update(subject=subject,label=label,record_id=record,
            user='继续目标任务，多做几个人物；SMPLify也加上连续帧约束',
            state='native_inputs_ready_no_fit_started',
            selector_source=str(Path(original.__file__)),selector_sha256=sha(original.__file__),
            wrapper_source=str(Path(__file__)),wrapper_sha256=sha(__file__),
            new_reference_fit_required=True,new_temporal_fit_required=True,
            no_old_network_or_individual_mesh_loaded=True,
            gaussian_admission=False,
            next='Review native masks and8+4 predetermined images; use the same verified temporal adapter after the first winterhat candidate review. No parallel GPU launch.')
        save(original.ROOT/'plan.json',p)
        print('ADDITIONAL_NATIVE_INPUT_READY',subject,flush=True)

if __name__=='__main__':main()
