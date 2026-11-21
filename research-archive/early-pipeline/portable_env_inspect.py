import sys,json,importlib.metadata as md,subprocess
from pathlib import Path
sys.path[:0]=['/opt/gs-slam-lab/shuolee-ga-raster/target','/opt/gs-slam-lab/mini-splatting2-native-001/deps','/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/novel-view-face/head-structure/private-runtime/python-deps']
mods=['diff_gaussian_rasterization','simple_knn','roma','pytorch3d','nvdiffrast','einops','trimesh']
for name in mods:
 try:
  m=__import__(name);print(name,getattr(m,'__file__',None),getattr(m,'__version__','unknown'))
 except Exception as e:print(name,type(e).__name__,str(e))
for p in [Path('/opt/gs-slam-lab/shuolee-ga-raster'),Path('/opt/gs-slam-lab/mini-splatting2-native-001')]:
 print('DIR',p)
 print('\n'.join(str(x) for x in p.glob('*')))
for name in ['pytorch3d','nvdiffrast']:
 try:
  d=md.distribution(name);print(name,'direct_url',d.read_text('direct_url.json'))
 except Exception as e:print(name,str(e))
