from pathlib import Path
import json,zipfile,hashlib
B=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
r=json.loads((B/'human117-build.json').read_text(encoding='utf-8'))
source=Path(r['source']); candidate=Path(r['candidatePath'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256']
allowed={'ppt/slides/slide52.xml','ppt/slides/_rels/slide52.xml.rels','ppt/notesSlides/notesSlide52.xml','ppt/notesSlides/_rels/notesSlide52.xml.rels','ppt/media/image5.jpeg'}
out=B/'human117-exact-candidate.pptx'
with zipfile.ZipFile(source) as a,zipfile.ZipFile(candidate) as b,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 assert [n for n in a.namelist() if n.endswith('.rels') and b'image5.jpeg' in a.read(n)]==['ppt/slides/_rels/slide52.xml.rels']
 assert [n for n in b.namelist() if n.endswith('.rels') and b'image5.jpeg' in b.read(n)]==['ppt/slides/_rels/slide52.xml.rels']
 for info in a.infolist():z.writestr(info,b.read(info.filename) if info.filename in allowed else a.read(info.filename))
with zipfile.ZipFile(source) as a,zipfile.ZipFile(out) as b:
 changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 assert set(changed)<=allowed and 'ppt/slides/slide52.xml' in changed
 assert b.testzip() is None
(B/'human117-exact-package.json').write_text(json.dumps(dict(candidate=str(out),source=str(source),changed=changed,other_members_byte_identical=True),indent=2))
print(out)
