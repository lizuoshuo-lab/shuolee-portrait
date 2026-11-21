"""Preserve the original package while inserting Artifact Tool authored slide52 and notes only."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,xml.etree.ElementTree as ET
B=Path(__file__).parent
r=json.loads((B/'free-transport123-native-deck-build.json').read_text(encoding='utf-8'))
source,candidate=Path(r['source']),Path(r['candidatePath'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
out=B/'free-transport123-native-exact-candidate.pptx'
allowed={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
with ZipFile(source) as a,ZipFile(candidate) as b,ZipFile(out,'w',ZIP_DEFLATED) as z:
    replacements={}
    for member,rel in [('ppt/slides/slide52.xml','ppt/slides/_rels/slide52.xml.rels'),('ppt/notesSlides/notesSlide52.xml','ppt/notesSlides/_rels/notesSlide52.xml.rels')]:
        def relation_keys(archive):
            values={}
            for e in ET.fromstring(archive.read(rel)):
                key=(e.attrib['Type'],e.attrib['Target'])
                assert key not in values
                values[key]=e.attrib['Id']
            return values
        before,after=relation_keys(a),relation_keys(b)
        assert all(not kind.endswith('/image') for kind,target in after), after
        assert set(after).issubset(before), (before,after)
        missing=set(before)-set(after)
        assert all(kind.endswith('/image') for kind,target in missing),missing
        data=b.read(member)
        for key,rid in after.items():
            data=data.replace(rid.encode(),before[key].encode())
        replacements[member]=data
    assert b'<a:tbl>' in replacements['ppt/slides/slide52.xml']
    assert b'<p:pic>' not in replacements['ppt/slides/slide52.xml']
    for info in a.infolist():
        z.writestr(info,replacements.get(info.filename,a.read(info.filename)))
with ZipFile(source) as a,ZipFile(out) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)==allowed and b.testzip() is None
(B/'free-transport123-native-exact-package.json').write_text(json.dumps(dict(changed=changed,other_members_byte_identical=True,removed_slide52_image_reference_only=True,native_table=True),indent=2),encoding='utf-8')
