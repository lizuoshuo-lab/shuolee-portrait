from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import numpy as np,json
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/grayT-fixed-direction-review')
rows={56:[110,230,245,258],64:[60,218,235,253],72:[120,265,282,298],80:[110,240,257,267]}
out={}
for n,ys in rows.items():
 gray=np.asarray(Image.open(O/f'frame_{n:03d}_native_rail_crop.png').convert('L'),float)
 info=[]
 for y in ys:
  a=gray[max(0,y-1):y+2].mean(0);d=np.gradient(gaussian_filter1d(a,.7))
  peaks,_=find_peaks(d,height=2,distance=3)
  info.append(dict(y=y,positive_edges=[dict(x=int(x),gradient=round(float(d[x]),2)) for x in peaks]))
 out[n]=info
(O/'seed-row-appearance.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
