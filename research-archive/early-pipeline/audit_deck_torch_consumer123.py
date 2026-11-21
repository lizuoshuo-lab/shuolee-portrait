from pathlib import Path
from zipfile import ZipFile
import hashlib,json,xml.etree.ElementTree as ET
from PIL import Image,ImageChops
B=Path(__file__).parent
tag='free-transport123-torch-consumer'
rp=B/(tag+'-deck-receipt.json')
r=json.loads(rp.read_text(encoding='utf-8'))
source=Path(r['source']); candidate=Path(r['finalPath'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with ZipFile(source) as a,ZipFile(candidate) as b:
 assert a.namelist()==b.namelist()
 changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 assert set(changed)=={'ppt/slides/slide52.xml','ppt/notesSlides/notesSlide52.xml'}
 ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
 ta=ET.fromstring(a.read('ppt/slides/slide52.xml')).find('.//a:tbl',ns)
 tb=ET.fromstring(b.read('ppt/slides/slide52.xml')).find('.//a:tbl',ns)
 table_identical=ET.tostring(ta)==ET.tostring(tb)
 assert table_identical
before=Image.open(B/(tag+'-before-52.png')).convert('RGB')
after=Image.open(B/(tag+'-final-52.png')).convert('RGB')
assert before.size==after.size==(1280,720)
assert ImageChops.difference(before.crop((0,0,1280,550)),after.crop((0,0,1280,550))).getbbox() is None
assert sha(source)==r['source_sha256']
r.update(actual_changed_slide_review_pending=False,actual_changed_slide_visual_review={
 'reviewed':True,'image':str(B/(tag+'-final-52.png')),
 'conclusion':'No overflow or overlap. Existing title, subtitle and native14 table remain identical. Only bottom evidence scope text and notes changed. New GPU mathematical control remains distinct from the prior native bridge.',
 'original_slide_top550_pixels_identical':True},
 changed_zip_members=changed,other_members_byte_identical=True,main_table_xml_identical=True,
 canonical_replaced=False)
rp.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
ep=B/(tag+'-exact-package.json')
e=json.loads(ep.read_text(encoding='utf-8'))
e.pop('removed_slide52_image_reference_only',None)
e.update(image_assets_changed=False,main_table_xml_identical=True,actual_changed_slide_reviewed=True)
ep.write_text(json.dumps(e,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:r[k] for k in ['finalPath','final_sha256','changed_zip_members','first51_render_hashes_identical','main_table_xml_identical','canonical_replaced']},ensure_ascii=False))
