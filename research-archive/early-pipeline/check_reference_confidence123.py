import numpy as np,json
from pathlib import Path
R=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
z=np.load(R/'data/observations.npz');out={}
for k in z.files:
    a=z[k]
    if k=='scores':out[k]={'shape':list(a.shape),'range':[float(a.min()),float(a.max())],'per_selected':{str(s):{'min':float(a[s].min()),'max':float(a[s].max()),'over1':int((a[s]>1).sum())} for s in [0,31,62]}}
print(json.dumps(out))
