"""Register one temporal-density control and decode only24new native observations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'; B=R/'video10-static-material'; I=R/'video9-10-native-intake/video10'; O=R/'video10-temporal-bridge'
source=Path('D:/00/工作/数据集/video10.mp4');ff=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
now=lambda:datetime.now(timezone.utc).isoformat()
assert not O.exists(),'Already registered: do not repeat decode'
base=read(B/'plan.json'); review=read(B/'visual-review.json'); assert review['actual_saved_visual_review'] and not review['geometry_admitted']
assert sha(source)==base['input_video_sha256']
anchors=base['frames']; added=sorted({round(a+(b-a)*j/4) for a,b in zip(anchors[:-1],anchors[1:]) for j in [1,2,3]});frames=sorted(anchors+added)
assert len(added)==24 and len(frames)==33 and not set(anchors)&set(added)
cached={p.name:sha(p) for p in sorted((B/'run').glob('*.npz'))}
O.mkdir();start=time.monotonic()
save(O/'plan.json',dict(registered=now(),state='registered_native_decode_only',source=str(source),source_sha256=base['input_video_sha256'],script_sha256=sha(Path(__file__)),baseline=str(B),baseline_plan_sha256=sha(B/'plan.json'),baseline_review_sha256=sha(B/'visual-review.json'),
 hypothesis='Shorter temporal gaps may preserve physical patch visibility and reduce descriptor aliasing. Existing9frames are about4.5seconds apart. Add only intermediate observations; no K/pose/model change. New masking must be visually reviewed and is not a claimed calibration correction.',
 frozen_variables=['native2704x1520 pixels and decoder','existing9keypoints/descriptors/rawfamilyidentities','RootSIFT .75 mutual ratio','same descriptor support guard and quotas','wholephysicalfamilyholdout/noresidualselection','conditionalK unchanged; no geometry in this job'],
 anchor_frames=anchors,new_frames=added,frames=frames,max_frame_gap=max(b-a for a,b in zip(frames[:-1],frames[1:])),cached_npz_sha256=cached,
 image_contract='Decode directly from existing rectified/cropped MP4 with same ffmpeg RGB output. No resize/rectification/rotation. Existing9PNGs reused from intake. ROI boxes for24newimages must be reviewed before extraction; they are not physical material labels.',
 feature_method=base['detector'],matching=dict(descriptor='RootSIFT',method='mutualBF',ratio=.75,new_pairs='Indexdistance<=2 among33; cache all36anchor pairs without recomputing; evaluate all available closedtriplets',camera_filter=False),
 family_contract='Seed old physical-family groups from baseline including8px-near equivalence. Retain old FIT/HELD labels and contradictions/ambiguities. After new observations, mixed old split connectedfamilies are quarantined as a whole, never reassigned. Pure new families get deterministic20percentHELD before geometry. Review representatives using same spatial/role quotas, up to113rows+16contexts. Unknown identity stays unknown.',
 acceptance='Same coarse support as baseline:20FIT/8HELD fixed material families,3bins,6FIT/3HELD rigid nonground, differentdepthsupport. Initialpair12reviewedFIT and later8reviewedFIT2D3D. Face physicalheldout and alltrainingframes are separate gates.',
 budget=dict(new_decoded_frames=24,decode_runs=1,decode_timeout_seconds=90,new_feature_passes=1,new_feature_frames=24,CPU_feature_match_seconds=600,GPU_tasks=0,new_cameras=0,training_steps=0,review_rows_max=113,context_rows_max=16),
 stop='No thresholdsweep, second featurepass, interpolationofcameras, dynamicnetwork, heldouttransfer or automatictraining. Fail/insufficient support preserved. 33diagnosticframes do not represent full860frame calibration.',
 new_feature_execution_admitted=False))
save(O/'progress.json',dict(updated=now(),state='decoding_new24_native_frames',pid=None))
imgs=O/'native';imgs.mkdir()
expr="select='"+'+'.join(f'eq(n,{i})' for i in added)+"'"
cmd=[str(ff),'-nostdin','-hide_banner','-loglevel','error','-threads','2','-noautorotate','-i',str(source),'-map','0:v:0','-vf',expr,'-fps_mode','passthrough','-frames:v','24',str(imgs/'sample_%02d.png')]
c=subprocess.run(cmd,capture_output=True,text=True,timeout=90);(O/'decode.log').write_text(c.stderr,encoding='utf-8');assert c.returncode==0,c.stderr
files=sorted(imgs.glob('sample_*.png'));assert len(files)==24
entries=[]
for f,n in zip(files,added):
 p=imgs/f'native_frame_{n:06d}.png';f.rename(p)
 with Image.open(p) as im:assert im.size==(2704,1520)
 entries.append(dict(frame=n,path=str(p),sha256=sha(p),native_size=[2704,1520]))
for j in range(8):
 page=Image.new('RGB',(1352,2280),'white')
 for k,e in enumerate(entries[3*j:3*j+3]):
  with Image.open(e['path']) as im:page.paste(im.resize((1352,760),Image.Resampling.LANCZOS),(0,k*760))
 page.save(O/f'overview_{j:02d}.jpg',quality=93)
assert sha(source)==base['input_video_sha256']
assert cached=={p.name:sha(p) for p in sorted((B/'run').glob('*.npz'))}
save(O/'native-inputs.json',dict(updated=now(),entries=entries,anchor_image_sha256=base['image_sha256'],anchor_root=str(I),new_geometric_resampling=False,overview_only_resized=True,burned_text=False))
save(O/'progress.json',dict(updated=now(),state='native24_ready_region_review_pending',pid=None,decode_seconds=time.monotonic()-start,new_features_extracted=False,new_camera=False,training_started=False))
print(json.dumps(dict(state='native24_ready_region_review_pending',new_frames=added,all_frames=frames,decode_seconds=time.monotonic()-start),ensure_ascii=False))
