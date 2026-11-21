from pathlib import Path
from datetime import datetime
import hashlib
import json
import posixpath
import shutil
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image, ImageChops

build = Path(__file__).resolve().parent
receipt_file = build / 'floor-deck-receipt.json'
receipt = json.loads(receipt_file.read_text('utf-8'))
source, final = Path(receipt['source']), Path(receipt['finalPath'])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source) == receipt['source_sha256']
assert sha(final) == receipt['final_sha256']
validation = json.loads((build / 'floor-final-validation.json').read_text('utf-8'))
assert validation['firstPartyImport']['passed']
assert validation['packageIntegrity']['finding_count'] == 0
assert validation['presentationLayout']['finding_count'] == 0

rel_ns = '{http://schemas.openxmlformats.org/package/2006/relationships}'
rel_attr = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
def normalize(package, number):
    name = f'ppt/slides/slide{number}.xml'
    rel_name = f'ppt/slides/_rels/slide{number}.xml.rels'
    mapping = {}
    for rel in ET.fromstring(package.read(rel_name)).findall(rel_ns + 'Relationship'):
        target = rel.get('Target')
        if rel.get('TargetMode') == 'External':
            value = ('external', target)
        else:
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), target)).lstrip('/')
            value = ('sha256', hashlib.sha256(package.read(target)).hexdigest())
        mapping[rel.get('Id')] = repr(value)
    tree = ET.fromstring(package.read(name))
    for element in tree.iter():
        if element.tag == '{http://schemas.microsoft.com/office/drawing/2014/main}creationId':
            element.set('id', 'ignored-creation-id')
        for key in list(element.attrib):
            if key.startswith(rel_attr):
                element.set(key, mapping[element.get(key)])
    return ET.tostring(tree)

with zipfile.ZipFile(source) as old, zipfile.ZipFile(final) as new:
    for i in range(1, 55):
        assert normalize(old, i) == normalize(new, i), f'Slide {i} changed'
        name = f'ppt/notesSlides/notesSlide{i}.xml'
        assert old.read(name) == new.read(name), f'Notes {i} changed'
pixel_unchanged = []
for i in range(1, 55):
    previous = Image.open(build / 'vposer-final-renders' / f'{i:02d}.png').convert('RGB')
    current = Image.open(build / 'floor-final-renders' / f'{i:02d}.png').convert('RGB')
    assert previous.size == current.size
    assert ImageChops.difference(previous, current).getbbox() is None, f'Render {i} changed'
    pixel_unchanged.append(i)
shutil.copyfile(final, source)
assert sha(source) == receipt['final_sha256']
receipt.update(canonical_replaced=True, visually_reviewed_slides=[55],
               unchanged_slide_xml_media_and_notes=list(range(1,55)),
               unchanged_rendered_pixels=pixel_unchanged,
               verified_at=datetime.now().astimezone().isoformat(timespec='seconds'))
receipt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), 'utf-8')
status_file = source.parent.parent / 'work-status.json'
status = json.loads(status_file.read_text('utf-8-sig'))
status['surface_contact_closeout']['presentation'] = {'path':str(source), 'sha256':sha(source), 'changed_slides':[55], 'earlier_slides_pixel_identical':True}
status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), 'utf-8')
print(json.dumps({'canonical_replaced':True, 'unchanged_slides':54, 'sha256':sha(source)}))
