from pathlib import Path
import json,hashlib
p=Path('D:/00/工作/人体重建/configs/server-assets.json')
j=json.loads(p.read_text(encoding='utf-8-sig'))
relative='gvhmr/hmr4d/utils/body_model/coco_aug_dict.pth'
asset=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets')/relative
j['files']=[x for x in j['files'] if x['path']!=relative]
j['files'].append(dict(path=relative,bytes=asset.stat().st_size,sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),source='Existing official GVHMR repository resource'))
j['files'].sort(key=lambda x:x['path'])
p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print({'files':len(j['files']),'new_asset_bytes':asset.stat().st_size})
