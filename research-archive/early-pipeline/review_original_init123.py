"""Reuse the established native-crop comparison for the original initializer."""
from pathlib import Path
import argparse
import os
import review_shared_offset_lr123 as r


def main(root,step,novel):
    def fullpath(p):
        p=Path(p).resolve()
        return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
    r.ROOT=fullpath(root);r.O=fullpath(r.O);r.SNAP=r.ROOT/f'gaussian-run/step-{step:06d}'
    r.OUT=r.ROOT/f'comparison-step-{step:06d}'
    if step==1000:
        r.OLD=r.O/'shared-vertex-offset-adc-1000/gaussian-run/step-001000'
        r.BASELINE_NAME='200k_init_ADC300_60_step1000'
    else:
        r.OLD=r.O/'shared-vertex-offset-18000/gaussian-run/step-018000'
        r.BASELINE_NAME='historical200k_ADC10000_2000_step18000_not_single_factor'
    r.CANDIDATE_NAME='original_triangle_init_ADC300_60_step'+str(step)
    r.RECORD_PREFIX=f'20261030_原版逐三角初始化{step}步_左原图_中历史对照_右当前'
    r.NOVEL_RECORD=f'20261030_原版逐三角初始化{step}步_连续未知视角000至035.png'
    budget_trial='original-200k-18000' in str(root)
    if budget_trial:
        r.RECORD_PREFIX=f'20261030_20万点预算{step}步_左原图_中历史对照_右当前'
        r.NOVEL_RECORD=f'20261030_20万点预算{step}步_连续未知视角000至035.png'
        if step==1000:
            r.OLD=r.O/'shared-vertex-offset-original-init-18000-io-repair/gaussian-run/step-001000'
            r.BASELINE_NAME='original_triangle_unbudgeted_step1000'
    if novel:
        r.novel()
    else:
        r.compare()
        data=r.io.read(r.OUT/'comparison.json');data.update(steps=step,
            baseline_root=str(r.OLD),baseline_same_step_and_ADC=(step==1000),
            isolated_initialization_comparison=(step==1000 and not budget_trial),
            isolated_density_budget_comparison=(step==1000 and budget_trial))
        r.io.save(r.OUT/'comparison.json',data)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--novel',action='store_true')
    args=p.parse_args();main(args.root,args.step,args.novel)
