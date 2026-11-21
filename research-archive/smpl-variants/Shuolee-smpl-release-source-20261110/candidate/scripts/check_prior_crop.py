"""CPU regression of actual crop pixels against full-image coordinate ramps."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import runtime as rt


def check():
    import cv2
    import numpy as np
    rt.configure(PROJECT/'assets-unused-for-crop-check')
    from shuolee_portrait.crops import prepare_prior_crop

    y, x = np.mgrid[:768, :1024].astype(np.float32)
    # The image itself encodes source coordinates, independently of our receipts.
    ramp = np.stack([x, y, x+2*y], -1)
    boxes = [[290, 180, 550, 580], [110.25, 99.75, 500.5, 620.25],
             [470.1, 210.8, 650.6, 390.4]]
    samples = np.array([[0, 0], [255, 0], [127.5, 127.5], [255, 255]], float)
    dense_y, dense_x = np.mgrid[:256, :256]
    dense = np.stack([dense_x, dense_y, np.ones_like(dense_x)], -1)
    results=[]
    for box in boxes:
        crop, cs, receipt = prepare_prior_crop(ramp, box)
        A, B = np.array(receipt['source_to_crop']), np.array(receipt['crop_to_source'])
        source = dense @ B.T
        inside=(source[:,:,0]>=0)&(source[:,:,0]<=1023)&(source[:,:,1]>=0)&(source[:,:,1]<=767)
        outside=(source[:,:,0]<-1)|(source[:,:,0]>1024)|(source[:,:,1]<-1)|(source[:,:,1]>768)
        raster_error = float(np.max(np.abs(crop[:,:,:2][inside]-source[:,:,:2][inside])))
        padding_error=float(np.max(np.abs(crop[outside]))) if outside.any() else 0.
        roundtrip_error = float(np.max(np.abs((np.c_[samples, np.ones(4)]@B.T@A.T)[:,:2]-samples)))
        # Independent corner specification: the ROI spans exactly 0..255.
        centre, size = cs[:2], cs[2]
        src_corners = np.array([centre-size/2, centre+[size/2, -size/2], centre, centre+size/2])
        endpoint_error = float(np.max(np.abs((np.c_[src_corners, np.ones(4)]@A.T)[:,:2]-samples)))
        assert raster_error < .05 and padding_error==0 and roundtrip_error < 1e-8 and endpoint_error < .001, dict(raster=raster_error,padding=padding_error,roundtrip=roundtrip_error,endpoint=endpoint_error)
        old = np.array([[256/size, 0, 127.5-centre[0]*256/size],
                        [0, 256/size, 127.5-centre[1]*256/size]], np.float32)
        old_crop=cv2.warpAffine(ramp,old,(256,256),flags=cv2.INTER_LINEAR)
        old_error=float(np.max(np.abs(old_crop[:,:,:2][inside]-source[:,:,:2][inside])))
        assert old_error > .2, 'Regression fixture must detect the previous 256/size map.'
        results.append(dict(box=box,source_pixel_raster_error_max=raster_error,
                            outside_image_padding_error_max=padding_error,
                            roundtrip_error_max=roundtrip_error,crop_endpoint_error_max=endpoint_error,
                            previous_map_source_error_max=old_error))
    return dict(passed=True,checks=results,weights_loaded=False,gpu_task=False,
                new_video_fit_run=False,ghosting_resolved=False,quality_accepted=False,
                scope='Coordinate-coded synthetic RGB only; prior crop mapping, not physical calibration',
                source_sha256=rt.sha(PROJECT/'src/shuolee_portrait/crops.py'),
                prior_source_sha256=rt.sha(PROJECT/'src/shuolee_portrait/prior.py'),
                upstream_sha256=rt.sha(PROJECT/'third_party/gvhmr/hmr4d/network/hmr2/utils/preproc.py'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=check();rt.save(args.output,result);print(json.dumps(result,indent=2))
