"""Show every-frame results and the worst foreground regressions, without image realignment."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/mnt/d/Projects/Shuolee-analysis-20261028/dynamic-face-fan-all-views')
report=json.loads((root/'report.json').read_text());records=report['records']
fig,axes=plt.subplots(2,1,figsize=(12,7))
for ax,key,title in zip(axes,['head_crop_masked_rgb_l1','head_foreground_rgb_l1'],['Head crop, including exterior ghosts','Head foreground only']):
    for label,name in [('static','Original static'),('head_motion','Dynamic face + face-crop observations')]:
        ax.plot(np.arange(1,len(records)+1),[r[label][key] for r in records],label=name,lw=1.5)
    ax.set(title=title,xlabel='Frame',ylabel='Mean absolute RGB error (0-1)');ax.grid(alpha=.2);ax.legend()
fig.suptitle('All 95 original training views | 18,000 steps | same cameras and masks',fontsize=13)
fig.tight_layout();fig.savefig(root/'all-frame-render-errors.png',dpi=160);plt.close(fig)
worst=sorted(records,key=lambda r:r['head_motion']['head_foreground_rgb_l1']/r['static']['head_foreground_rgb_l1'],reverse=True)[:6]
fig,axes=plt.subplots(6,1,figsize=(12,17))
for ax,row in zip(axes,worst):
    ax.imshow(Image.open(root/'head-crops'/f'{row["frame"]}.png'));ax.axis('off')
    change=row['head_motion']['head_foreground_rgb_l1']/row['static']['head_foreground_rgb_l1']-1
    ax.set_title(f'{row["frame"]}: foreground error +{change:.1%}',fontsize=11)
fig.suptitle('Worst foreground regressions | Original / Static / Dynamic face\nNo frame removal or image alignment',fontsize=13)
fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(root/'worst-foreground-regressions.png',dpi=140);plt.close(fig)
(root/'visual-review-selection.json').write_text(json.dumps(dict(selection='Six largest head-foreground relative regressions, selected from all 95 frames',frames=[r['frame'] for r in worst],quality_accepted=False),indent=2)+'\n')
print(json.dumps(dict(frames=len(records),worst=[r['frame'] for r in worst])))
