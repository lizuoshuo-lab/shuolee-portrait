"""All-frame inspection sheets; record crops stay at native resolution."""
import argparse,math
from pathlib import Path
from PIL import Image
import audit_gaussianavatars_params123 as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',choices=['surface','joint','mask','offset-init','normalized-laplacian','single-landmark-weight','sequential-stage','direct-body','body-lr','continuous','monocular-lr','body-continuity','pose-prior-sum','pose-prior-long','root-continuity'],required=True);p.add_argument('--epoch',type=int,choices=[30,60,90,120]);args=p.parse_args()
    run=a.REC/'demo-vhap-initialization'/('smplify-vhap-'+args.run)
    comparison=f'comparison-epoch{args.epoch:03d}' if args.epoch else 'comparison'
    report=a.read(run/comparison/'result.json')
    root=a.H/('review-smplify-vhap-'+args.run+(f'-epoch{args.epoch:03d}' if args.epoch else ''));a.lp(root).mkdir(exist_ok=True)
    items=[]
    for region in ('body','face','left_hand','right_hand','feet'):
        count=6 if region=='body' else 12
        for start in range(0,len(report['per_frame']),count):
            rows=report['per_frame'][start:start+count];images=[]
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
            path=root/f'{region}-{start//count+1:02d}.png';page.save(a.lp(path))
            items.append(dict(region=region,path=str(path),sha256=a.sha(path),frames=[r['frame_id'] for r in rows],
                order='row-major; within each frame '+' / '.join(item['panels']),inspection_only=True,
                native_record_images_unchanged=True,uniform_body_downscale=region=='body',detail_resampling=False))
    a.save(run/comparison/'contact-sheet-manifest.json',dict(frames=report['frames'],pages=items,actual_visual_review=False))
    print('Created',len(items),'inspection sheets for',report['frames'],'completed frames.')

if __name__=='__main__':main()
