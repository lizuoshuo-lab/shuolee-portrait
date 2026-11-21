from pathlib import Path
import json, zipfile, hashlib, xml.etree.ElementTree as ET

B = Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
r = json.loads((B/'server-delivery-build.json').read_text(encoding='utf-8'))
source, candidate = Path(r['source']), Path(r['candidatePath'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == r['source_sha256']
allowed = {'ppt/slides/slide52.xml', 'ppt/notesSlides/notesSlide52.xml'}
out = B/'server-delivery-exact-candidate.pptx'
with zipfile.ZipFile(source) as a, zipfile.ZipFile(candidate) as b, zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    assert a.read('ppt/media/image5.jpeg') == b.read('ppt/media/image5.jpeg')
    replacements = {}
    for member, rel in [('ppt/slides/slide52.xml', 'ppt/slides/_rels/slide52.xml.rels'),
                        ('ppt/notesSlides/notesSlide52.xml', 'ppt/notesSlides/_rels/notesSlide52.xml.rels')]:
        old = {(e.attrib['Type'], e.attrib['Target']): e.attrib['Id'] for e in ET.fromstring(a.read(rel))}
        new = {(e.attrib['Type'], e.attrib['Target']): e.attrib['Id'] for e in ET.fromstring(b.read(rel))}
        assert old.keys() == new.keys(), 'Slide relationship targets changed'
        data = b.read(member)
        for key, new_id in new.items():
            data = data.replace(new_id.encode(), old[key].encode())
        replacements[member] = data
    for info in a.infolist():
        z.writestr(info, replacements.get(info.filename, a.read(info.filename)))
with zipfile.ZipFile(source) as a, zipfile.ZipFile(out) as b:
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert set(changed) == allowed
    assert b.testzip() is None
(B/'server-delivery-exact-package.json').write_text(json.dumps(dict(candidate=str(out), source=str(source), changed=changed, other_members_byte_identical=True), indent=2), encoding='utf-8')
print(out)
