from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,xml.etree.ElementTree as ET
B=Path(__file__).parent
r=json.loads((B/'visibility122-deck-build.json').read_text(encoding='utf-8'))
source,candidate=Path(r['source']),Path(r['candidatePath']);figure=Path(r['figure'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
assert hashlib.sha256(figure.read_bytes()).hexdigest()==r['figure_sha256']
allowed={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml','ppt/slides/_rels/slide52.xml.rels'}
new_asset='ppt/media/n1_material122.png';out=B/'visibility122-exact-candidate.pptx'
with ZipFile(source) as a,ZipFile(candidate) as b,ZipFile(out,'w',ZIP_DEFLATED) as z:
    assert new_asset not in a.namelist()
    assert any(e.attrib.get('Extension')=='png' for e in ET.fromstring(a.read('[Content_Types].xml')))
    replacements={}
    for member,rel in [('ppt/slides/slide52.xml','ppt/slides/_rels/slide52.xml.rels'),('ppt/notesSlides/notesSlide52.xml','ppt/notesSlides/_rels/notesSlide52.xml.rels')]:
        old=ET.fromstring(a.read(rel));new=ET.fromstring(b.read(rel));data=b.read(member)
        for n in new:
            matches=[e for e in old if e.attrib['Type']==n.attrib['Type'] and (n.attrib['Type'].endswith('/image') or e.attrib['Target']==n.attrib['Target'])]
            assert len(matches)==1
            o=matches[0];data=data.replace(n.attrib['Id'].encode(),o.attrib['Id'].encode())
            if n.attrib['Type'].endswith('/image'):
                # The old image may be shared. Give slide52 a new asset; retain all existing media.
                rel_bytes=a.read(rel).replace(o.attrib['Target'].encode(),('/'+new_asset).encode())
                replacements[rel]=rel_bytes
        replacements[member]=data
    for info in a.infolist():z.writestr(info,replacements.get(info.filename,a.read(info.filename)))
    z.writestr(new_asset,figure.read_bytes())
with ZipFile(source) as a,ZipFile(out) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    added=list(set(b.namelist())-set(a.namelist()))
    assert set(changed)==allowed and added==[new_asset] and b.testzip() is None
(B/'visibility122-exact-package.json').write_text(json.dumps(dict(changed=changed,added=added,other_members_byte_identical=True),indent=2),encoding='utf-8')
