from pathlib import Path
import hashlib,json
import numpy as np
import pycolmap
from plyfile import PlyData
root=Path('/opt/gs-slam-lab/shuolee-git119-validation-20261027/sofa-git-ea95a8d')
export=root/'export'
ply=PlyData.read(export/'point_cloud.ply',mmap=False)['vertex'].data
for name in ply.dtype.names: assert np.isfinite(ply[name]).all(),name
recon=pycolmap.Reconstruction(str(export/'colmap'))
cameras=json.loads((export/'cameras.json').read_text())
assert len(recon.images)==len(cameras)==89
assert len(ply)==204529
result=dict(state='passed',ply_reader='plyfile',points=len(ply),fields=list(ply.dtype.names),finite=True,sha256=hashlib.sha256((export/'point_cloud.ply').read_bytes()).hexdigest(),camera_reader='pycolmap 3.10.0',colmap_images=len(recon.images),colmap_cameras=len(recon.cameras),json_cameras=len(cameras),sparse_points=len(recon.points3D),steps=json.loads((root/'result.json').read_text())['steps'])
Path('/mnt/d/Projects/Shuolee-validation-20261027/independent-export119.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
