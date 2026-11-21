"""Split-isolated CPU observations with explicit distortion validity masks.

Consumes audit_enerf_input's immutable manifest. Original RGB files stay intact.
This is a data adapter, not an implementation or a trained result of 4K4D.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from audit_enerf_input import digest


class EnerfNativeDataset:
    def __init__(self, manifest, split, root_override=None, allow_calibration_only=False):
        if split not in ('train', 'heldout_view', 'heldout_time'):
            raise ValueError('An explicit train or heldout split is required')
        self.manifest_path = Path(manifest).resolve()
        spec = json.loads(self.manifest_path.read_text(encoding='utf8'))
        if spec.get('appearance_training_permitted') is False and not allow_calibration_only:
            raise ValueError('Calibration-only observations must not enter appearance training/evaluation')
        self.root = Path(spec['root'] if root_override is None else root_override).resolve()
        self.split = split
        self.cameras = spec['cameras']
        # Heldout RGB and masks are never opened while creating a train loader.
        self.rows = tuple(dict(r) for r in spec['images'] if r['split'] == split)
        if not self.rows: raise ValueError('Empty requested split')
        ids = [(r['camera'], r['frame']) for r in spec['images']]
        if len(ids) != len(set(ids)): raise ValueError('Duplicate/overlapping image assignments')
        for r in spec['images']:
            expected = ('heldout_time' if r['frame'] in spec['heldout_frames'] else
                'heldout_view' if r['camera'] in spec['heldout_cameras'] else 'train')
            if r['split'] != expected: raise ValueError('Split assignment contradicts frozen protocol')
        for filename, expected in spec['calibration_hashes'].items():
            path=(self.root/filename).resolve()
            if not path.is_relative_to(self.root) or digest(path) != expected:
                raise ValueError('Calibration changed after audit')
        self._maps = {}
        self._verified = set()

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or index < 0 or index >= len(self):
            raise IndexError(index)
        r = self.rows[index]
        p = (self.root/r['path']).resolve()
        if not p.is_relative_to(self.root): raise ValueError('Image outside audited dataset root')
        # Check each source on first access. A new loader rechecks all used files.
        if p not in self._verified:
            if digest(p) != r['sha256']: raise ValueError('Native RGB changed after audit')
            self._verified.add(p)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (r['height'],r['width']): raise ValueError('Image decode/dimensions changed')
        c = self.cameras[r['camera']]
        K = np.array(c['K'],dtype=np.float64)
        D = np.array(c['D'],dtype=np.float64)
        key = (r['camera'],r['width'],r['height'])
        if key not in self._maps:
            # Keep native K/size. Black border pixels cannot be used as image evidence.
            mx,my = cv2.initUndistortRectifyMap(K,D,np.eye(3),K,
                (r['width'],r['height']),cv2.CV_32FC1)
            valid = ((mx>=0)&(my>=0)&(mx<r['width']-1)&(my<r['height']-1)&
                np.isfinite(mx)&np.isfinite(my))
            self._maps[key] = (mx,my,valid)
        mx,my,valid = self._maps[key]
        rgb = cv2.cvtColor(cv2.remap(bgr,mx,my,cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0)),cv2.COLOR_BGR2RGB)
        result=dict(rgb=rgb, valid_pixels=valid.copy(), K=K,
            w2c=np.array(c['w2c'],dtype=np.float64), camera=r['camera'], frame=r['frame'],
            source_sha256=r['sha256'], split=self.split, native_size=(r['width'],r['height']),
            pixel_coordinates='OpenCV; unchanged K and native size; rectified pixels only',
            supervision='RGB with valid_pixels; no author geometry or heldout RGB initialization')
        if 'common_frame' in r:result['common_frame']=r['common_frame']
        if 'mask_path' in r:
            mask=(self.root/r['mask_path']).resolve()
            if not mask.is_relative_to(self.root):raise ValueError('Matte outside audited root')
            if mask not in self._verified:
                if digest(mask)!=r['mask_sha256']:raise ValueError('Author matte changed after audit')
                self._verified.add(mask)
            value=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if value is None or value.shape!=bgr.shape[:2]:raise ValueError('Matte dimensions changed')
            result['foreground_matte']=cv2.remap(value,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(np.float32)/255
            result['matte_semantics']=r['mask_semantics']
        return result
