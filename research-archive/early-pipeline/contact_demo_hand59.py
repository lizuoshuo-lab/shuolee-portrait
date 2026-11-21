"""Local inspection sheets; native record images are left unchanged."""
from pathlib import Path
import math
from PIL import Image
import audit_gaussianavatars_params123 as a


def main():
    run=a.REC/'demo-vhap-initialization/fullbody-smplifyx/hand-data-control59'
    report=a.read(run/'comparison/result.json')
    root=a.H/'review-demo-hand59';a.lp(root).mkdir(exist_ok=True)
    items=[]
    for region in ['body','face','left_hand','right_hand','feet']:
        for start in range(0,len(report['per_frame']),6):
            rows=report['per_frame'][start:start+6];images=[]
            for row in rows:
                item=next(i for i in row['images'] if i['region']==region)
                path=Path(item['record_path'].replace('/mnt/d/','D:/'))
                assert a.sha(path)==item['sha256']
                image=Image.open(a.lp(path)).convert('RGB')
                if region=='body':
                    factor=min(1.,360/image.height)
                    image=image.resize((round(image.width*factor),round(image.height*factor)),Image.Resampling.LANCZOS)
                images.append(image)
            width=max(i.width for i in images);height=max(i.height for i in images)
            page=Image.new('RGB',(2*width,math.ceil(len(images)/2)*height),(245,245,245))
            for index,im in enumerate(images):page.paste(im,((index%2)*width,(index//2)*height))
            path=root/f'{region}-{start//6+1:02d}.png';page.save(a.lp(path))
            items.append(dict(region=region,path=str(path),sha256=a.sha(path),frames=[r['frame_id'] for r in rows],
                order='row-major; within each frame raw/baseline/candidate',inspection_only=True,
                native_record_images_unchanged=True,uniform_body_downscale=region=='body',detail_resampling=False))
    a.save(run/'comparison/contact-sheet-manifest.json',dict(frames=report['frames'],pages=items,actual_review=False))
    print('Created',len(items),'inspection sheets for',report['frames'],'completed frames.')


if __name__=='__main__':main()
