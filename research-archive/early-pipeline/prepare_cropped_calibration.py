"""Explicit crop/resize calibration conversion. No image, camera-pose or training fit.

This diagnostic does not modify the frozen 113 input or admit Gaussian training.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def numeric(values, count, label, integer=False):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f'{label} requires {count} explicit values')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f'{label} has missing/non-finite values')
    if integer and any(int(v) != v for v in values):
        raise ValueError(f'{label} must use integer pixel dimensions/offsets')
    return values


def convert(spec, decoded_wh):
    sw, sh = numeric(spec.get('source_wh'), 2, 'source_wh', True)
    x0, y0, cw, ch = numeric(spec.get('crop_xywh'), 4, 'crop_xywh', True)
    ow, oh = numeric(spec.get('output_wh'), 2, 'output_wh', True)
    if min(sw, sh, cw, ch, ow, oh) <= 0 or min(x0, y0) < 0 or x0+cw > sw or y0+ch > sh:
        raise ValueError('Crop is outside the calibrated image or has invalid dimensions')
    if list(decoded_wh) != [ow, oh]:
        raise ValueError('Decoded video dimensions do not match the declared output pixel domain')
    if spec.get('rotation_clockwise_degrees') != 0:
        raise ValueError('Image rotation is unknown/nonzero; review rotation before applying this converter')
    if spec.get('stabilization') != 'off' or spec.get('fixed_crop_across_frames_confirmed') is not True:
        raise ValueError('A fixed image transform across frames must be confirmed; no EIS assumption')
    convention = spec.get('pixel_convention')
    if convention not in ('integer_centers', 'pixel_corners'):
        raise ValueError('Specify the calibration pixel convention')
    domain = spec.get('pixel_domain')
    if domain not in ('distorted', 'undistorted'):
        raise ValueError('Specify whether the supplied image/calibration domain is distorted')
    K = spec.get('K')
    if not isinstance(K, list) or len(K) != 3:
        raise ValueError('Supply the original 3x3 calibration matrix')
    for row in K:
        numeric(row, 3, 'K row')
    if K[2] != [0, 0, 1] or K[0][0] <= 0 or K[1][1] <= 0 or K[1][0] != 0 or K[0][1] != 0:
        raise ValueError('This converter supports a zero-skew standard pinhole intrinsic matrix')
    distortion = spec.get('distortion')
    if not isinstance(distortion, dict) or not isinstance(distortion.get('coefficients'), list):
        raise ValueError('Explicit distortion model/coefficients are required, including the none case')
    counts = {'none': 0, 'opencv': 5, 'opencv_fisheye': 4}
    if distortion.get('model') not in counts:
        raise ValueError('Unsupported or unspecified distortion model')
    numeric(distortion['coefficients'], counts[distortion['model']], 'distortion coefficients')
    if domain == 'undistorted' and distortion['model'] != 'none':
        raise ValueError('Undistorted pixels must not silently receive source-image distortion again')
    sx, sy = ow/cw, oh/ch
    # Half-pixel resize: u_out + .5 = s * (u_in - crop_x + .5).
    dx = (sx-1)/2 if convention == 'integer_centers' else 0
    dy = (sy-1)/2 if convention == 'integer_centers' else 0
    A = [[sx, 0, -sx*x0+dx], [0, sy, -sy*y0+dy], [0, 0, 1]]
    result = [[sum(A[r][k]*K[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    return dict(schema='shuolee-explicit-crop-calibration-v1', source_wh=[sw, sh],
                crop_xywh=[x0, y0, cw, ch], output_wh=[ow, oh], source_K=K,
                K=result, source_pixel_to_video_pixel=A, pixel_convention=convention,
                pixel_domain=domain, distortion=copy.deepcopy(distortion),
                anisotropic_resize=abs(sx-sy) > 1e-9,
                extrinsics='Must be estimated from this video; crop does not supply R/t.',
                numerical_transform_only=True, physical_calibration_validated=False,
                training_started=False)


def self_check():
    # Synthetic dimensions only: these are NOT the user's camera parameters.
    spec = dict(source_wh=[3840, 2880], crop_xywh=[1110, 0, 1620, 2880], output_wh=[1080, 1920],
                K=[[3000, 0, 1919.5], [0, 3001, 1439.5], [0, 0, 1]],
                rotation_clockwise_degrees=0, stabilization='off', fixed_crop_across_frames_confirmed=True,
                pixel_convention='integer_centers', pixel_domain='distorted',
                distortion={'model': 'opencv', 'coefficients': [-.01, .001, 0, 0, 0]})
    maximum_error = 0
    count = 0
    for convention in ('integer_centers', 'pixel_corners'):
        for output in ([1080, 1920], [2160, 3840], [1620, 2880]):
            case = copy.deepcopy(spec);case['pixel_convention'] = convention;case['output_wh'] = output
            converted = convert(case, output)
            for xn, yn in [(-.2, -.3), (.001, .17), (.2, .6), (0., 0.)]:
                r2=xn*xn+yn*yn;scale=1-.01*r2+.001*r2*r2
                xd, yd=xn*scale, yn*scale
                u=3000*xd+1919.5;v=3001*yd+1439.5
                sx, sy=output[0]/1620, output[1]/2880
                if convention == 'integer_centers':
                    mapped=[(u-1110+.5)*sx-.5, (v+.5)*sy-.5]
                else:
                    mapped=[(u-1110)*sx, v*sy]
                k=converted['K'];projected=[k[0][0]*xd+k[0][2], k[1][1]*yd+k[1][2]]
                maximum_error=max(maximum_error, *(abs(a-b) for a,b in zip(mapped,projected)))
                count += 1
    negative = []
    changes = [('rotation_clockwise_degrees', None), ('rotation_clockwise_degrees', 90),
               ('stabilization', 'on'), ('fixed_crop_across_frames_confirmed', False),
               ('crop_xywh', [None, 0, 1620, 2880]), ('crop_xywh', [3000, 0, 1620, 2880]),
               ('K', None), ('pixel_convention', None), ('distortion', None)]
    for key, value in changes:
        case=copy.deepcopy(spec);case[key]=value
        try:
            convert(case, [1080,1920])
        except ValueError:
            negative.append(key)
        else:
            raise AssertionError(f'Invalid {key} was not rejected')
    try:
        convert(spec, [1920,1080])
    except ValueError:
        negative.append('video_dimensions')
    else:
        raise AssertionError('Mismatched video dimensions were accepted')
    assert maximum_error < 1e-9
    return {'synthetic_projection_checks': count, 'max_pixel_error':maximum_error,
            'invalid_inputs_rejected':len(negative), 'real_video_tested':False, 'training_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--decoded-wh',type=int,nargs=2);p.add_argument('--self-check',action='store_true')
    a=p.parse_args()
    if a.self_check:
        print(json.dumps(self_check()));return
    if a.spec is None or a.output is None or a.decoded_wh is None:
        p.error('--spec, --output and --decoded-wh are required')
    data=a.spec.read_bytes(); result=convert(json.loads(data.decode('utf-8-sig')),a.decoded_wh)
    result['source_spec_sha256']=hashlib.sha256(data).hexdigest()
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    print(a.output)


if __name__ == '__main__':
    main()
