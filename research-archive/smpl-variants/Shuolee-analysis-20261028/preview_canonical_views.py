"""Contact sheet of unretouched novel-view renders of the standard PLY."""
from pathlib import Path
import argparse,json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--motion',choices=['face','fan'],default='face');args=parser.parse_args()
root=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029')/('gray-dynamic-face' if args.motion=='face' else 'gray-dynamic-face-fan')
out=Path('/mnt/d/Projects/Shuolee-analysis-20261028')/('dynamic-face-render-comparison' if args.motion=='face' else 'dynamic-face-fan-render-comparison')
assert json.loads((root/'export/validation.json').read_text())['format_passed']
for kind in ['head','body']:
    fig,axes=plt.subplots(2,6,figsize=(18,6 if kind=='head' else 9))
    for i,ax in enumerate(axes.flat):
        ax.imshow(Image.open(root/f'export/review/{kind}_{i:02d}.png'));ax.set_title(f'{i*30} degrees',fontsize=10);ax.set_axis_off()
    fig.suptitle('Canonical standard PLY | novel views, no corresponding ground truth | quality not accepted',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/f'canonical-{kind}-novel-views.png',dpi=150);plt.close(fig)
print(str(out))
