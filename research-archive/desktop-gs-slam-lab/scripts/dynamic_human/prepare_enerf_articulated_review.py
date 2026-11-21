from pathlib import Path
import json,math,shutil
import numpy as np
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';SOURCE=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';path=OUT/'review-protocol.json';assert not path.exists()
obs={(r['camera'],r['frame']):r for r in json.loads((SOURCE/'observations.json').read_text())['records']};regions=[]
for c in ['01','04']:
 for t in [745,751,755]:
  r=obs[c,t]
  for label,person,ids,pad in [('dark_forearm',0,[8,10],30),('light_forearm',1,[7,9],30),('dark_shoe',0,[16,20,21,22],30),('dark_head',0,[0,1,2,3,4],45)]:
   d=r['detections'][person];xy=np.array(d['keypoints'])[ids];lo=np.floor(xy.min(0)-pad).astype(int);hi=np.ceil(xy.max(0)+pad+1).astype(int);box=[max(0,int(lo[0])),max(0,int(lo[1])),min(1920,int(hi[0])),min(1080,int(hi[1]))]
   regions.append(dict(camera=c,frame=t,name=label,bbox=box,source='062RTMWoriginalphoto prediction; fixedwithoutlookingatreconstruction'))
protocol=dict(status='fixed_before_candidate_render_review',full_person_views=[dict(camera=c,frame=t) for c in ['01','04','08','17'] for t in [745,751,755]],regions=regions,
 temporal='Allsixreservedoddframes foreachof01/04/08/17; displaysequenceonly, realdatasetFPSnotindependentlycertified.',
 qualification='Rectangularregionscancontainoccluders/background. Theyarecomparisonwindows, notcleanbodypartsemanticGT.',goal_complete=False)
path.write_text(json.dumps(protocol,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name)
