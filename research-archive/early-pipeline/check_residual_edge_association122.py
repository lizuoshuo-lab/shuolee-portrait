"""D3: change only association across the exact D1 native gradient profiles.

All branches/old observations remain. Stop at ambiguity; do not straighten a curve,
fit distortion, snap to a line, or declare numerical continuity physical truth.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import cv2
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'residual-distortion-direction';O=R/'residual-edge-association'
DATA=Path('/mnt/d/GS-SLAM-Data/robust122-whiteT-native-supervision/rgb')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def follow(peaks,anchor,aim,maxgap=2.,anchor_tolerance=2.):
    selected=[None]*len(peaks);states=['not_reached']*len(peaks)
    near=[k for k,x in enumerate(peaks[anchor]) if abs(x['offset']-aim)<=anchor_tolerance]
    if len(near)!=1:return selected,states,dict(anchor='missing_or_ambiguous',candidates=near)
    selected[anchor]=near[0];states[anchor]='seed_proposal';stops={}
    for direction in [-1,1]:
      j=anchor
      while 0<=j+direction<len(peaks):
        k=j+direction;old=peaks[j][selected[j]]['offset']
        neighbours=[z for z,p in enumerate(peaks[k]) if abs(p['offset']-old)<=maxgap]
        if len(neighbours)!=1:
          states[k]='missing' if not neighbours else 'branch_ambiguity';stops[str(direction)]=dict(index=k,state=states[k],candidates=neighbours);break
        z=neighbours[0];back=[q for q,p in enumerate(peaks[j]) if abs(p['offset']-peaks[k][z]['offset'])<=maxgap]
        if back!=[selected[j]]:
          states[k]='reverse_branch_ambiguity';stops[str(direction)]=dict(index=k,state=states[k],candidates=back);break
        selected[k]=z;states[k]='unique_local_branch';j=k
    return selected,states,stops

def controls():
    # A curved target plus a stronger parallel competing edge: no straightness term exists.
    t=np.linspace(0,1,120);curve=1.8*np.sin(2*np.pi*t)
    peaks=[[dict(offset=float(v),strength=float(2-t[i])),dict(offset=float(v+5),strength=float(1+2*t[i]))] for i,v in enumerate(curve)]
    sel,st,_=follow(peaks,0,0);uv=np.array([peaks[i][x]['offset'] for i,x in enumerate(sel)])
    assert all(x==0 for x in sel) and np.array_equal(uv,curve)
    winner=np.array([np.argmax([x['strength'] for x in p]) for p in peaks]);assert (winner==1).any()
    ambiguous=[p.copy() for p in peaks];ambiguous[30]=[dict(offset=float(curve[30]+v),strength=2) for v in [-.4,.4]]
    s,ss,stop=follow(ambiguous,0,0);assert s[30] is None and ss[30]=='branch_ambiguity'
    missing=[p.copy() for p in peaks];missing[40]=[];s,ss,_=follow(missing,0,0);assert s[40] is None and ss[40]=='missing'
    return dict(curved_target_exactly_preserved=True,curvature_peak_to_peak=float(np.ptp(uv)),
      stronger_adjacent_edge_not_selected=True,ambiguous_branch_stopped=True,missing_profile_stopped=True,
      source_pixels_or_curve_smoothed=False,physical_accuracy_claim=False)

assert not O.exists();O.mkdir()
P=read(D/'plan.json');M=read(D/'measurements.json');RR=read(D/'line-review.json');cfg=P['extraction']
anchors={k:dict(index=i,normal_offset_px=offset) for k,i,offset in [
 ('A3',10,0.),('A4',10,-3.),('A5',10,-5.),('C1',60,1.),('C2',60,2.),('C3',60,1.5),
 ('B1',60,-2.5),('B2',60,0.),('B3',60,-1.5),('D1',60,4.),('D2',60,1.)]}
save(O/'plan.json',dict(stage='D3_association_only',updated=now(),source_plan_sha256=sha(D/'plan.json'),source_measurements_sha256=sha(D/'measurements.json'),
 hypothesis='Independent strongest-gradient selection can switch physical edge. Keep identical D1 images/profiles and change only association to a unique bidirectional local branch.',
 method='All signed local maxima above the original3gray/px threshold are preserved. From one explicit native approximate edge seed, continue only if exactly one forward neighbour within2px and exactly one reverse neighbour; stop on missing/branching. No gap filling, line fit, smoothing or residual-based selection.',
 anchor_selection='Approximate native physical edge seeds frozen from full architecture/source crops; not derived from a new coefficient fit. Still proposals requiring actual review.',
 anchors=anchors,normal_gap_px=2.,anchor_tolerance_px=2.,
 scope='Measurement component experiment on same11lines only. Old heldout groups are exposed development validation, not a fresh independent benchmark. No new calibration or training.',
 budget='One CPU extraction of exactly3existingnativePNGs andsame120profiles per line. No threshold sweep or second run except demonstrated implementation error.',
 stop='Save all candidates and stopped prefixes. No claim of improved physical accuracy from removing unsupported observations. Register new calibration only after independent material and full-field support.' ))
save(O/'controls.json',controls());cv2.setNumThreads(2);cache={};records=[]
for line in M['records']:
  if line['id'] not in anchors:continue
  f=line['frame'];path=DATA/f'frame_{f:06d}.png'
  if f not in cache:
    assert sha(path)==P['input_hashes'][path.name];cache[f]=cv2.imread(str(path))
  gray=cv2.cvtColor(cache[f],cv2.COLOR_BGR2GRAY).astype(np.float32)
  a,b=np.array(line['a'],float),np.array(line['b'],float);tangent=(b-a)/np.linalg.norm(b-a);normal=np.array(line['normal'])
  fraction=np.linspace(0,1,cfg['samples_per_line']);center=a[None]+fraction[:,None]*(b-a)
  offs=np.arange(-cfg['normal_search_half_width_px'],cfg['normal_search_half_width_px']+.001,cfg['normal_step_px']);profiles=[]
  for toff in cfg['tangent_average_offsets_px']:
    pos=center[:,None]+offs[None,:,None]*normal+tangent*toff
    profiles.append(cv2.remap(gray,pos[:,:,0].astype(np.float32),pos[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
  profile=np.mean(profiles,0);grad=np.gradient(profile,cfg['normal_step_px'],axis=1)
  rowpeak=np.argmax(np.abs(grad[:,1:-1]),axis=1)+1;sign=1. if np.median(grad[np.arange(len(grad)),rowpeak])>=0 else -1.;g=grad*sign
  peaks=[];old_offsets=[]
  for j in range(len(center)):
    # Recompute the old strongest winner solely to verify the exact D1 photometric input.
    w=int(np.argmax(g[j,1:-1])+1);den=float(g[j,w-1]-2*g[j,w]+g[j,w+1]);delta=0 if abs(den)<1e-10 else .5*float(g[j,w-1]-g[j,w+1])/den
    old_offsets.append(float(offs[w]+np.clip(delta,-.5,.5)*cfg['normal_step_px']))
    row=[]
    if not any(lo<=fraction[j]<=hi for lo,hi in line['exclude_fraction']):
      for k in range(3,len(offs)-3):
        if g[j,k]>=3 and g[j,k]>=g[j,k-1] and g[j,k]>g[j,k+1]:
          den=float(g[j,k-1]-2*g[j,k]+g[j,k+1]);delta=0 if abs(den)<1e-10 else .5*float(g[j,k-1]-g[j,k+1])/den
          offset=float(offs[k]+np.clip(delta,-.5,.5)*cfg['normal_step_px']);uv=center[j]+normal*offset
          if 8<uv[0]<1512 and 8<uv[1]<2696:row.append(dict(offset=offset,strength=float(g[j,k]),uv=uv.tolist()))
    peaks.append(row)
  parity=float(np.abs(np.array(old_offsets)-line['offsets']).max());assert parity<1e-12
  anchor=anchors[line['id']];sel,states,stops=follow(peaks,anchor['index'],anchor['normal_offset_px'])
  selected_uv=[None if x is None else peaks[j][x]['uv'] for j,x in enumerate(sel)]
  count=sum(x is not None for x in sel)
  records.append(dict(id=line['id'],frame=f,split=line['split'],physical_group=line['physical_group'],anchor=anchor,
     all_peaks=peaks,selected_peak_indices=sel,states=states,stops=stops,selected_uv=selected_uv,
     selected_count=count,old_eligible_count=line['valid_count'],old_profile_parity_max_px=parity,
     physical_identity_confirmed=False))
save(O/'result.json',dict(stage='D3',updated=now(),state='complete_pending_review',records=records,
 exact_old_photometric_parity=True,all_branches_retained=True,K_changed=False,camera_fit=False,coefficient_fit=False,training=False,
 physical_acceptance=False,scope='Unique local edge branches are proposals only; stopped tails stay unknown.'))
print(json.dumps([dict(id=r['id'],selected=r['selected_count'],old_eligible=r['old_eligible_count'],stops=r['stops']) for r in records]))
