"""Audit saved refinement images and record the inspected negative result."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-appearance-refinement-008'
DATA=LAB/'data/dynamic-human/dynamic3dgaussians/data/basketball'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=json.loads((OUT/'protocol.json').read_text())
    r=json.loads((OUT/'evaluation.json').read_text())
    assert sha(Path(__file__).with_name('refine_first_frame_appearance.py'))==p['implementation_sha256']
    assert sha(LAB/p['source'])==p['source_sha256']
    metadata=json.loads((DATA/'test_meta.json').read_text())
    files={int(fn.split('/')[0]):fn for fn in metadata['fn'][0]}
    checks=[]
    for row in r['per_view']:
        path=OUT/f"renders/{row['mode']}/camera{row['camera']}.png"
        pred=np.asarray(Image.open(path),dtype=np.float64)/255
        gt=np.asarray(Image.open(DATA/'ims'/files[row['camera']]).convert('RGB'),dtype=np.float64)/255
        x0,y0,x1,y1=p['evaluation_rois'][str(row['camera'])]
        mse=float(np.mean((pred[y0:y1,x0:x1]-gt[y0:y1,x0:x1])**2))
        psnr=-10*np.log10(max(mse,1e-12))
        delta=abs(psnr-row['person_region']['psnr'])
        assert delta<.02
        checks.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),png_psnr=float(psnr),float_difference_db=float(delta)))
    means=r['summary']
    receipt=dict(status='verified_negative_result_for_restoring_clothing_detail',
        protocol_sha256=sha(OUT/'protocol.json'),metrics_sha256=sha(OUT/'evaluation.json'),
        saved_images_checked=len(checks),max_png_float_psnr_difference_db=max(x['float_difference_db'] for x in checks),
        spatial_minus_equal_update_photometric_psnr_db=means['spatial_color']['psnr']-means['photometric']['psnr'],
        spatial_minus_original_source_psnr_db=means['spatial_color']['psnr']-means['source']['psnr'],
        spatial_minus_original_source_ssim=means['spatial_color']['ssim']-means['source']['ssim'],
        visual_samples=[f'assets/camera{c}-person.png' for c in [0,10,15,30]],
        findings=['All four heldout first-frame camera panels inspected.',
            'Near back/shoulder view camera15: photometric-only extra training amplifies small colored fragments. Spatial prior suppresses this deterioration but leaves the original large uneven/mottled clothing.',
            'Camera30 silhouette and sleeves largely unchanged; camera0/10 hands and clothing remain softer than source photographs.',
            'Against original source, mean PSNR is essentially unchanged and SSIM worsens. Do not promote as a clear-person reconstruction improvement.',
            'This finite color-only intervention does not prove geometry is the sole cause. Next work must examine surface/visibility consistency or a body prior.'],
        decision='Do not propagate either refinement into the dynamic production branch; preserve both diagnostics and original models.',
        limitations=p['limitations'],images=checks)
    (OUT/'independent-audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['images','findings','limitations','visual_samples']},indent=2))

if __name__=='__main__':main()
