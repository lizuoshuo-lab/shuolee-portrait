from pathlib import Path
import json
from PIL import Image
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/solver-track-membership')
q=json.loads((O/'rendered-evidence.json').read_text(encoding='utf-8'))
rows=[x for x in q['figures'] if 'track_id' in x]
for chunk in range(3):
    rr=rows[6*chunk:6*(chunk+1)]
    sheet=Image.new('RGB',(768,384*((len(rr)+1)//2)),(240,240,240))
    for i,r in enumerate(rr):
        p=Path(r['path'].replace('/mnt/d/','D:/',1))
        sheet.paste(Image.open(p),((i%2)*384,(i//2)*384))
    path=H/f'membership123-patches-{chunk}.png';sheet.save(path)
    print(json.dumps({'sheet':str(path),'left_to_right_top_to_bottom':[dict(id=r['track_id'],slots=r['slots']) for r in rr]}))
print(json.dumps({'context':[r for r in q['figures'] if r.get('kind')=='native_scene_context']},ensure_ascii=False))
