"""Build read-only native evidence for every ambiguous SFD association."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/sfd-face-presence124/full167')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def win(p):
    if p.startswith('/mnt/'):
        return Path(p[5].upper()+':/'+p[7:])
    if p.startswith('/'):
        return Path('//wsl.localhost/Ubuntu-20.04'+p)
    return Path(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = ROOT/'association-review'
    assert not (out/'result.json').exists()
    out.mkdir(exist_ok=True)
    result = read(ROOT/'result.json')
    rows = {(r['subject'],r['frame']):r for r in read(ROOT/'plan.json')['schedule']}
    cases = [r for r in result['records'] if len(r['associated_indices'])>1]
    assert len(cases)==16
    pages = []
    receipts = []
    for r in cases:
        row=rows[r['subject'],r['frame']]
        src=win(row['image'])
        assert sha(src)==r['source_sha256']
        raw=Image.open(src).convert('RGB')
        overlay=raw.copy();draw=ImageDraw.Draw(overlay)
        draw.rectangle(row['initial_box_xyxy'],outline=(0,180,255),width=3)
        for b in r['detections']:
            draw.rectangle(b['xyxy'],outline=(0,230,90) if b['associated'] else (230,150,0),width=3)
        # Full native frame is retained, so the head crop cannot hide false
        # positives elsewhere in the tracked-person rectangle.
        pair=Image.new('RGB',(raw.width*2,raw.height));pair.paste(raw,(0,0));pair.paste(overlay,(raw.width,0))
        number='006' if r['subject']=='sofa' else '008'
        path=Path('D:/00/记录')/number/f'特征点_124_SFD_{r["frame"]}_全图_原图_人物框与所有人脸框.png'
        assert not path.exists();pair.save(path)
        crops=[]
        for i in r['associated_indices']:
            b=np.asarray(r['detections'][i]['xyxy'])
            margin=max(float(max(b[2:]-b[:2]))*.5,16.)
            lo=np.maximum(np.floor(b[:2]-margin).astype(int),0)
            hi=np.minimum(np.ceil(b[2:]+margin).astype(int),raw.size)
            crop=raw.crop(tuple(np.r_[lo,hi]));crop.thumbnail((280,280))
            crops.append(crop)
        panel=Image.new('RGB',(1200,600))
        thumbnail=pair.copy();thumbnail.thumbnail((1200,300));panel.paste(thumbnail,(0,0))
        for i,crop in enumerate(crops):panel.paste(crop,(i*300,310))
        panel_path=out/f'{r["subject"]}-{r["frame"]}.png';panel.save(panel_path)
        pages.append(panel)
        receipts.append(dict(subject=r['subject'],frame=r['frame'],native_image=str(path),sha256=sha(path),panel=str(panel_path),associated_indices=r['associated_indices']))
    page_paths=[]
    for start in range(0,len(pages),3):
        sheet=Image.new('RGB',(1200,1800))
        for i,panel in enumerate(pages[start:start+3]):sheet.paste(panel,(0,i*600))
        path=out/f'page-{start//3+1:02d}.png';sheet.save(path);page_paths.append(str(path))
    (out/'result.json').write_text(json.dumps(dict(source_result_sha256=sha(ROOT/'result.json'),actual_visual_review=False,new_detector_calls=0,cases=receipts,pages=page_paths),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(cases),pages=page_paths),ensure_ascii=False))


if __name__=='__main__':main()
