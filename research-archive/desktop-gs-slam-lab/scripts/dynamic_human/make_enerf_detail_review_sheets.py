from pathlib import Path
from PIL import Image
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-detail-weight-076/review'
for c in ['01','02','03','04','05','07','08','09','10','11','13','14','15','16','17']:
    ims=[Image.open(R/f'camera{c}-frame{t:06d}.png').convert('RGB') for t in [744,750,756]]
    out=Image.new('RGB',(sum(i.width for i in ims),max(i.height for i in ims)),'white');x=0
    for i in ims:out.paste(i,(x,0));x+=i.width
    out.save(R/f'camera{c}-three-times.png')
print('15 native-pixel contact sheets ready; none resampled')
