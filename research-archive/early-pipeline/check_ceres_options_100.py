import importlib.util,json,sys
import pycolmap
o=pycolmap.BundleAdjustmentOptions()
print(json.dumps(dict(pycolmap=pycolmap.__version__,fields=[x for x in dir(o) if not x.startswith('_')],pyceres_installed=importlib.util.find_spec('pyceres') is not None)))
try:
    print(o.solver_options)
except Exception as e:
    print(type(e).__name__+': '+str(e))
