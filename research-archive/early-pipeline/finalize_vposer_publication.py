from pathlib import Path
import hashlib
import json
import posixpath
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime

build = Path(__file__).resolve().parent
receipt_path = build / 'vposer-deck-receipt.json'
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
source, final = Path(receipt['source']), Path(receipt['finalPath'])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source) == receipt['source_sha256'], 'Canonical changed; inspect before replacing.'
assert sha(final) == receipt['final_sha256']
validation = json.loads((build / 'vposer-final-validation-v3.json').read_text('utf-8'))
assert validation['firstPartyImport']['passed']
assert validation['packageIntegrity']['finding_count'] == 0
assert validation['presentationLayout']['finding_count'] == 0

rels_ns = '{http://schemas.openxmlformats.org/package/2006/relationships}'
rel_attr = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

def normalized_slide(package, number):
    name = f'ppt/slides/slide{number}.xml'
    rel_name = f'ppt/slides/_rels/slide{number}.xml.rels'
    relationships = {}
    if rel_name in package.namelist():
        for r in ET.fromstring(package.read(rel_name)).findall(rels_ns + 'Relationship'):
            target = r.get('Target')
            if r.get('TargetMode') == 'External':
                value = ('external', target)
            else:
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), target)).lstrip('/')
                value = ('internal_sha256', hashlib.sha256(package.read(target)).hexdigest())
            relationships[r.get('Id')] = repr(value)
    root = ET.fromstring(package.read(name))
    for item in root.iter():
        if item.tag == '{http://schemas.microsoft.com/office/drawing/2014/main}creationId':
            item.set('id', 'creation-id-ignored')
        for key in list(item.attrib):
            if key.startswith(rel_attr):
                item.set(key, relationships[item.get(key)])
    return ET.tostring(root)

with zipfile.ZipFile(source) as old, zipfile.ZipFile(final) as new:
    unchanged = [i for i in range(1, 55) if normalized_slide(old, i) == normalized_slide(new, i)]
    assert unchanged == list(range(1, 55)), f'Unexpected earlier slide changes: {set(range(1,55))-set(unchanged)}'
    for i in unchanged:
        name = f'ppt/notesSlides/notesSlide{i}.xml'
        assert old.read(name) == new.read(name), f'Notes {i} changed'

shutil.copyfile(final, source)
assert sha(source) == receipt['final_sha256']
receipt.update(canonical_replaced=True, preserved_slides=unchanged,
               preservation_scope='All slide XML, text, layout and image bytes agree after normalizing creation IDs and relationship IDs; notes unchanged for slides 1-54.',
               final_slide_visually_reviewed=True,
               verified_at=datetime.now().astimezone().isoformat(timespec='seconds'))
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')

page = source.parents[2] / 'viewer' / 'body-smplx.html'
text = page.read_text('utf-8')
before = '下一项拟合将用三维人体姿态先验约束关节深度，同时统一共享体形、真实相机与逐帧运动。先检查原图叠加、身体侧面和时间稳定性，再从新高斯与新优化器开始训练。此前路线的动态表示与静态背景仍作为可参考的组成部分。'
after = '这组历史对照之后，身体姿态先验已完成上方的从头验证，整体对齐仍未通过。当前先核查真实图像对应与人物运动，再决定如何约束关节深度。通过原图叠加、固定身体侧面和时间一致性检查后，再从新高斯与新优化器开始训练。此前路线的动态表示与静态背景仍作为可参考的组成部分。'
assert before in text
page.write_text(text.replace(before, after, 1), encoding='utf-8')
print(json.dumps({'canonical_replaced': True, 'preserved_slides': len(unchanged), 'sha256': sha(source)}, ensure_ascii=False))
