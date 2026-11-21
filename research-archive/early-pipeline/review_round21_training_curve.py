"""Plot the one uninterrupted main optimizer log through R21's two snapshots."""

import csv
import json
from pathlib import Path

import numpy as np

from review_round21_io import pin,read,sha,require,descriptor


FIELDS = ('step','target_steps','points','loss','rgb_loss','foreground_l1','foreground_ssim','alpha_loss','seconds')


def build(out,loss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = pin(loss['file'],loss['sha256'])
    with raw.open(encoding='utf-8') as source:
        rows = [{key:item[key] for key in FIELDS} for item in map(json.loads,source)]
    steps = [row['step'] for row in rows]
    require(steps==[1,*range(100,18001,100)] and all(row['target_steps']==18000 for row in rows)
            and all(np.isfinite([row[key] for key in FIELDS]).all() for row in rows),
            'The reported curve is not the single uninterrupted 18,000-step optimizer log.')
    path = out/'training-curve.csv'
    with path.open('w',encoding='utf-8',newline='') as output:
        writer = csv.DictWriter(output,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    fig,axes = plt.subplots(3,1,figsize=(10,7.5),sharex=True,layout='constrained',dpi=120)
    for ax,field,label,color in zip(axes,('loss','foreground_l1','alpha_loss'),
                                   ('Total training loss','Foreground RGB L1','Alpha loss'),
                                   ('#265eaf','#12745c','#9c5d10'),strict=True):
        values = np.asarray([row[field] for row in rows])
        rolling = np.asarray([np.median(values[max(0,i-2):i+3]) for i in range(len(values))])
        ax.plot(steps,values,color=color,alpha=.38,linewidth=.8,label='Logged sample')
        ax.plot(steps,rolling,color=color,linewidth=1.6,label='5 logged samples, rolling median')
        for step in (10000,18000):
            ax.axvline(step,color='#657085',linestyle='--',linewidth=.8)
        ax.set_ylabel(label)
        ax.grid(True,axis='y',alpha=.2)
    axes[0].legend(loc='upper right',frameon=False,fontsize=8)
    axes[-1].set_xlabel('Main optimizer step')
    axes[-1].set_xlim(0,18100)
    axes[-1].set_xticks([0,2000,4000,6000,8000,10000,12000,14000,16000,18000])
    chart = out/'training-curve.png'
    fig.savefig(chart)
    plt.close(fig)
    checkpoint = {str(step):next(row for row in rows if row['step']==step) for step in (10000,18000)}
    return dict(raw=descriptor(raw),csv=descriptor(path),image=descriptor(chart),checkpoints=checkpoint,
                logged_samples=len(rows),main_optimizer_steps=18000,
                semantics='Logged main optimization only. Each marked raw checkpoint seeds its separately postprocessed export.')
