from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib, json, xml.etree.ElementTree as ET

B = Path(__file__).parent
r = json.loads((B/'r1c122-deck-build.json').read_text(encoding='utf-8'))
source, candidate = Path(r['source']), Path(r['candidatePath'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == r['source_sha256']
allowed = {'ppt/slides/slide52.xml', 'ppt/notesSlides/notesSlide52.xml'}
out = B/'r1c122-exact-candidate.pptx'
with ZipFile(source) as a, ZipFile(candidate) as b, ZipFile(out, 'w', ZIP_DEFLATED) as z:
    replacements = {}
    for member, rel in [('ppt/slides/slide52.xml', 'ppt/slides/_rels/slide52.xml.rels'),
                        ('ppt/notesSlides/notesSlide52.xml', 'ppt/notesSlides/_rels/notesSlide52.xml.rels')]:
        old = {(e.attrib['Type'], e.attrib['Target']): e.attrib['Id'] for e in ET.fromstring(a.read(rel))}
        new = {(e.attrib['Type'], e.attrib['Target']): e.attrib['Id'] for e in ET.fromstring(b.read(rel))}
        assert old.keys() == new.keys()
        data = b.read(member)
        for key, rid in new.items():
            data = data.replace(rid.encode(), old[key].encode())
        replacements[member] = data
    for info in a.infolist():
        z.writestr(info, replacements.get(info.filename, a.read(info.filename)))
with ZipFile(source) as a, ZipFile(out) as b:
    changed = [n for n in a.namelist() if a.read(n) != b.read(n)]
    assert set(changed) == allowed and b.testzip() is None
(B/'r1c122-exact-package.json').write_text(json.dumps(dict(changed=changed, other_members_byte_identical=True), indent=2), encoding='utf-8')
