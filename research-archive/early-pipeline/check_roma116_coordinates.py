from pathlib import Path
import datetime as dt
import hashlib
import importlib.metadata as md
import inspect
import json
import sys
import torch
sys.path[:0]=['/opt/gs-slam-lab/shuolee-roma116/python','/opt/gs-slam-lab/shuolee-roma116/RoMa']
def no_download(*a,**kw):raise RuntimeError('P2 coordinate check cannot download models')
torch.hub.load_state_dict_from_url=no_download
torch.hub.load=no_download
import romatch
from romatch.models.matcher import RegressionMatcher
from roma116_coordinates import raw_centers_to_normalized,normalized_to_raw_centers,sample_direction
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/roma')
class NativeCoordinateAPI:
    _to_pixel_coordinates=RegressionMatcher._to_pixel_coordinates
api=NativeCoordinateAPI()
torch.set_num_threads(4)
points=torch.tensor([[0.,0.],[1519.,2703.],[759.5,1351.5],[22.25,58.75],[1187.125,2356.75]],dtype=torch.float64)
normalized=raw_centers_to_normalized(points,(2704,1520))
native=RegressionMatcher.to_pixel_coordinates(api,normalized,2704,1520)
native_error=float((native-0.5-points).abs().max())
assert native_error<1e-9
# Unequal image dimensions and a non-identity map expose x/y swaps, half-frame errors,
# and accidental reuse of the forward output for reverse queries.
h,w=56,64
yy,xx=torch.meshgrid(torch.linspace(-1+1/h,1-1/h,h,dtype=torch.float64),torch.linspace(-1+1/w,1-1/w,w,dtype=torch.float64),indexing='ij')
grid=torch.stack([xx,yy],-1)
delta=grid.new_tensor([0.08,-0.12])
warp=torch.cat([torch.cat([grid,grid+delta],-1),torch.cat([grid-delta,grid],-1)],1)[None]
certainty=torch.ones((1,h,2*w),dtype=torch.float64)
q=torch.tensor([[400.,600.],[950.,1800.],[760.,1300.]],dtype=torch.float64)
ab,c,valid=sample_direction(warp,certainty,q,(2704,1520),(1200,800),'AtoB')
expected=normalized_to_raw_centers(raw_centers_to_normalized(q,(2704,1520))+delta,(1200,800))
forward_error=float((ab-expected).abs().max());assert forward_error<1e-9 and valid.all()
back,c,valid=sample_direction(warp,certainty,ab,(1200,800),(2704,1520),'BtoA')
reverse_error=float((back-q).abs().max());assert reverse_error<1e-9 and valid.all()
import PIL
from romatch.utils import get_tuple_transform_ops
transforms=get_tuple_transform_ops(resize=(560,560),normalize=True,clahe=False)
im=PIL.Image.new('RGB',(1520,2704),color=(127,63,31))
processed=transforms((im,im))
assert tuple(processed[0].shape)==(3,560,560)
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='passed_coordinate_and_import_checks_only',
    repo_commit='77f8d68803526dcddfd9b7a46bc76125bdc25f15',imports_passed=True,
    added_dependency=dict(name='loguru',version=md.version('loguru'),scope='isolated /opt/gs-slam-lab/shuolee-roma116/python; reused cached61KB wheel'),
    native_to_pixel_origin='pixel edge coordinates: first pixel center is (0.5,0.5)',
    internal_observation_origin='original raster center: first pixel center is (0,0)',
    conversion='normalized=(raw_center+0.5)*2/[W,H]-1; raw_center=native_pixel-0.5; COLMAP=native_pixel',
    native_coordinate_max_error_px=native_error,nonidentity_forward_max_error_px=forward_error,nonidentity_roundtrip_max_error_px=reverse_error,
    symmetric_contract='match returns BxHx2Wx4 and BxHx2W; forward uses left,last2; backward uses right,first2; batch must be1',
    source_preprocess='Official full-frame resize is mapped exactly back to original axes. No image is warped for display or training.',
    evidence_scope='CPU numeric and official preprocessing only; no RoMa neural prediction or accuracy claim',
    previous_ghosting_cause=False,weights_loaded=False,subject_inference_started=False,gaussian_training_started=False,
    adapter_sha256=hashlib.sha256((Path(__file__).parent/'roma116_coordinates.py').read_bytes()).hexdigest())
(O/'coordinate-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(O/'coordinate-api-source.txt').write_text(inspect.getsource(RegressionMatcher.to_pixel_coordinates)+'\n'+inspect.getsource(RegressionMatcher._to_pixel_coordinates)+'\n'+inspect.getsource(RegressionMatcher.match_keypoints))
p=json.loads((O/'preflight.json').read_text());p['imports']=dict(ok=True,module_path=str(romatch.__file__));p['state']='imports_and_coordinates_ready_weights_missing';p['updated']=result['updated'];p['repair']='Added loguru0.7.3 from cached wheel to isolated target only';p['coordinate_check']='coordinate-check.json'
(O/'preflight.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
