import inspect, json
import torch
try:
    import gsplat
    f = getattr(gsplat, 'rasterization_2dgs', None)
    print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'gsplat': gsplat.__file__, 'version': getattr(gsplat, '__version__', None), 'has_2dgs': f is not None, 'signature': str(inspect.signature(f)) if f else None}, indent=2))
except Exception as e:
    print(json.dumps({'error': type(e).__name__, 'message': str(e)}))
