"""Native mask panels for the same deterministic8+4 input selection."""
from pathlib import Path
import numpy as np
from PIL import Image
from run_official_gaussians123 import read, save, sha, now
from prepare_core_subjects124 import REC, paths


def main():
    for subject in ('sofa','indoor'):
        source=REC/(subject+'-input');out=source/'mask-review'
        assert not out.exists(), 'Preserve the previous review.'
        selection=read(source/'selection.json')
        rows={r['frame']:r for r in read(source/'data/inputs.json')['records']}
        chosen=sorted(selection['training_frames']+selection['review_only_frames'])
        assert len(chosen)==12 and len(set(chosen))==12
        number=paths(subject)[3];record=Path('/mnt/d/00/记录')/number
        record.mkdir(exist_ok=True);out.mkdir()
        panels=[];items=[]
        for name in chosen:
            row=rows[name]
            assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
            raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
            assert raw.size==mask.size==tuple(row['wh'])
            box=mask.getbbox();assert box is not None
            box=(max(0,box[0]-40),max(0,box[1]-40),min(raw.width,box[2]+40),min(raw.height,box[3]+40))
            region=raw.crop(box);alpha=mask.crop(box)
            cutout=Image.composite(region,Image.new('RGB',region.size),alpha)
            panel=Image.new('RGB',(3*region.width,region.height))
            for j,im in enumerate((region,alpha.convert('RGB'),cutout)):panel.paste(im,(j*region.width,0))
            file=record/f'mask_124_{name}_原图_mask_mask内部人像.png'
            assert not file.exists(),str(file)
            panel.save(file)
            items.append(dict(frame=name,image=str(file),role='training' if name in selection['training_frames'] else 'review_only',
                crop_integer=list(box),source_sha256=row['source_sha256'],mask_sha256=row['mask_sha256']))
            panel.thumbnail((1000,520));panels.append(panel)
        pages=[]
        for first in range(0,12,4):
            group=panels[first:first+4];page=Image.new('RGB',(max(x.width for x in group),sum(x.height for x in group)))
            y=0
            for panel in group:page.paste(panel,(0,y));y+=panel.height
            file=out/f'page-{first//4+1:02d}.png';page.save(file);pages.append(str(file))
        with np.load(source/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
        raster=[]
        for i,name in enumerate(obs['frame_ids'].astype(str)):
            p=obs['keypoints'][i];w,h=rows[name]['wh'];inside=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
            raster.append(dict(frame=name,outside_body_feet_indices=np.flatnonzero(~inside[:23]).tolist()))
        save(out/'result.json',dict(updated=now(),subject=subject,records=items,pages=pages,
            all_frame_raster_bounds=raster,native_mask_unchanged=True,training_roster_unchanged=True,
            actual_visual_review=False,gaussian_admitted=False,fit_started=False,
            source_sha256=sha(__file__),limits='Image bounds are not an occlusion or correctness test. No points or masks were changed.'))
        print('CORE_MASK_REVIEW_READY',subject,flush=True)


if __name__=='__main__':main()
