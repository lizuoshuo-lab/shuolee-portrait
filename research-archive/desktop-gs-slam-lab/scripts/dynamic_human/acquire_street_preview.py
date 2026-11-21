"""Download small original-image previews of an official public AV2 street log."""
from pathlib import Path
import json,requests,xml.etree.ElementTree as ET,hashlib,concurrent.futures
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003/street-preflight'
ROOT='https://argoverse.s3.amazonaws.com/'
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}

def main():
    first=json.loads((OUT/'argoverse-first-page-index.json').read_text());items=first['items'];token=first['next_token'];prefix=items[0]['key'].rsplit('/',1)[0]+'/'
    while token:
        r=requests.get(ROOT,params={'list-type':2,'prefix':prefix,'continuation-token':token},timeout=60);r.raise_for_status();doc=ET.fromstring(r.content)
        items.extend(dict(key=e.find('s:Key',NS).text,size=int(e.find('s:Size',NS).text)) for e in doc.findall('s:Contents',NS));token=doc.findtext('s:NextContinuationToken',namespaces=NS)
    assert len(items)==len({i['key'] for i in items})
    (OUT/'argoverse-complete-object-index.json').write_text(json.dumps(dict(source=ROOT,prefix=prefix,objects=items,total_bytes=sum(i['size'] for i in items)),indent=2))
    target=json.loads((OUT/'human-track-screening.json').read_text())['ranked_tracks'][0]['closest_timestamp_ns'];cameras=sorted({i['key'].split('/')[-2] for i in items if '/sensors/cameras/ring_' in i['key']})
    selected=[]
    for camera in cameras:
        candidates=[i for i in items if f'/sensors/cameras/{camera}/' in i['key']]
        item=min(candidates,key=lambda i:abs(int(Path(i['key']).stem)-target));selected.append(dict(item,camera=camera,timestamp_ns=int(Path(item['key']).stem),time_offset_ms=(int(Path(item['key']).stem)-target)/1e6))
    images=OUT/'source-images';images.mkdir(exist_ok=True)
    def download(item):
        path=images/f"{item['camera']}-{item['timestamp_ns']}.jpg"
        if not path.exists():
            r=requests.get(ROOT+item['key'],timeout=60);r.raise_for_status();assert len(r.content)==item['size'];path.write_bytes(r.content)
        assert path.stat().st_size==item['size'];item.update(file=path.relative_to(OUT).as_posix(),source=ROOT+item['key'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dimensions=list(Image.open(path).size));return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download,selected))
    font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',20);sheet=Image.new('RGB',(1600,480*4),'white');d=ImageDraw.Draw(sheet)
    for j,item in enumerate(records):
        im=Image.open(OUT/item['file']).convert('RGB');im.thumbnail((790,440));x=(j%2)*800;y=(j//2)*480;sheet.paste(im,(x,y+30));d.text((x+5,y+5),item['camera'],font=font,fill='#183747')
    sheet.save(OUT/'street-source-camera-review.png')
    (OUT/'street-source-preview-receipt.json').write_text(json.dumps(dict(status='seven_original_ring_camera_previews_only',images=records,whole_scene_downloaded=False,whole_scene_available_bytes=sum(i['size'] for i in items),whole_scene_objects=len(items),license='CC BY-NC-SA 4.0',attribution='© 2021 Argo AI, LLC',scope='source data screening, not a trained reconstruction; original dataset anonymization retained'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(downloaded_images=len(records),downloaded_bytes=sum(i['size'] for i in records),whole_scene_objects=len(items),whole_scene_bytes=sum(i['size'] for i in items)),indent=2))

if __name__=='__main__':main()
