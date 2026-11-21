from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import numpy as np,json
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grayT-fixed-direction-review')
g=np.asarray(Image.open(O/'frame_064_native_rail_crop.png').convert('L'),float)
rows=[]
for y in [131,145,180,218]:
 d=np.gradient(gaussian_filter1d(g[y-1:y+2].mean(0),.7));x,_=find_peaks(d,height=10,distance=3)
 rows.append(dict(y=y,edges=[dict(x=int(k),gradient=float(d[k])) for k in x if 135<=k<=305]))
(O/'annotation-appearance-check.json').write_text(json.dumps(dict(purpose='Audit the seed endpoint chosen on a cast-shadow tip rather than the intended rod edge; no camera loaded',rows=rows),indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows))
