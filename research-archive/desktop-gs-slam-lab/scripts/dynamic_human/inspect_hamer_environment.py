import sys,importlib.util
print(sys.executable)
print({n:importlib.util.find_spec(n) is not None for n in ['pytorch_lightning','smplx','yacs','timm','einops','hydra','pyrender','webdataset','cv2','skimage']})
