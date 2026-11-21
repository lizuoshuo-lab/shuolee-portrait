"""Validate derived pixels without weakening the original named-pose contract."""
import numpy as np


def layout_for_derived_pixels(fit, rows, source_rows, mappings, original_layout, sha):
    layout=original_layout(fit,source_rows)
    if len(rows)!=len(source_rows) or len(rows)!=len(mappings):
        raise ValueError('Derived roster differs')
    for row,source,item in zip(rows,source_rows,mappings):
        if row['frame']!=source['frame'] or row['frame']!=item['frame'] or row['source_frame_index']!=source['source_frame_index']:
            raise ValueError('Derived frame identity differs')
        if sha(source['image'])!=source['source_sha256'] or sha(source['mask'])!=source['mask_sha256']:
            raise ValueError('Original fitted input changed')
        Kint=np.asarray(item['integer_center_transform'])@np.asarray(source['K_integer_centers'])
        K=Kint.copy();K[:2,2]+=.5
        if not np.array_equal(Kint,row['K_integer_centers']) or not np.array_equal(K,row['K']):
            raise ValueError('Derived projection differs from registered affine')
        if not np.array_equal(row['w2c'],source['w2c']) or row['wh']!=item['derived_wh']:
            raise ValueError('Derived camera pose or dimensions changed')
    return layout
