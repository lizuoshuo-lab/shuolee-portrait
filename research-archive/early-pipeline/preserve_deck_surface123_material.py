"""Package preservation after native Artifact Tool edit; no hand-authored slide XML."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,xml.etree.ElementTree as ET,posixpath
B=Path(__file__).parent;r=json.loads((B/'surface123-material-deck-build.json').read_text(encoding='utf-8'))
source,candidate=Path(r['source']),Path(r['candidatePath'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
out=B/'surface123-material-exact-candidate.pptx';allowed={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
with ZipFile(source) as a,ZipFile(candidate) as b,ZipFile(out,'w',ZIP_DEFLATED) as z:
    replacements={}
    for member,rel in [('ppt/slides/slide52.xml','ppt/slides/_rels/slide52.xml.rels'),('ppt/notesSlides/notesSlide52.xml','ppt/notesSlides/_rels/notesSlide52.xml.rels')]:
        def relation_keys(archive):
            values={}
            for e in ET.fromstring(archive.read(rel)):
                kind,target=e.attrib['Type'],e.attrib['Target'];asset=None
                if kind.endswith('/image'):
                    asset=posixpath.normpath(posixpath.join(posixpath.dirname(member),target)).lstrip('/');target='owned_slide52_image'
                key=(kind,target);assert key not in values
                values[key]=(e.attrib['Id'],asset)
            return values
        before=relation_keys(a);after=relation_keys(b);assert before.keys()==after.keys()
        data=b.read(member)
        for key,(rid,asset) in after.items():
            oldrid,oldasset=before[key];data=data.replace(rid.encode(),oldrid.encode())
            if asset:
                value=b.read(asset);assert hashlib.sha256(value).hexdigest()==r['figure_sha256']
                # Ensure replacing this owned image cannot affect the other51slides or shared layouts.
                owners=[]
                for name in a.namelist():
                    if not name.endswith('.rels'):continue
                    if name=='_rels/.rels':base=''
                    else:base=posixpath.dirname(posixpath.dirname(name))
                    for rr in ET.fromstring(a.read(name)):
                        if rr.attrib.get('TargetMode')=='External':continue
                        resolved=posixpath.normpath(posixpath.join(base,rr.attrib['Target'])).lstrip('/')
                        if resolved==oldasset:owners.append(name)
                assert owners==['ppt/slides/_rels/slide52.xml.rels'],owners
                replacements[oldasset]=value
                if a.read(oldasset)!=value: allowed.add(oldasset)
        replacements[member]=data
    for info in a.infolist():z.writestr(info,replacements.get(info.filename,a.read(info.filename)))
with ZipFile(source) as a,ZipFile(out) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
    assert set(changed)==allowed and b.testzip() is None
(B/'surface123-material-exact-package.json').write_text(json.dumps(dict(changed=changed,other_members_byte_identical=True),indent=2),encoding='utf-8')
