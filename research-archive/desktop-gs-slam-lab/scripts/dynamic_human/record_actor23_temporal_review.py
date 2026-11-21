from pathlib import Path
import json,hashlib
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';p=O/'visual-review.json';v=json.loads(p.read_text(encoding='utf8'));t=json.loads((O/'continuity/summary.json').read_text(encoding='utf8'));known={(r['camera'],r['frame']):r for r in t['panels']};checked=[]
for c in ['00','13']:
    for f in [600,612,624]:
        r=known[c,f];path=O/'continuity'/r['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];checked.append(dict(file='continuity/'+r['file'],sha256=r['sha256']))
v['temporal_sampled_stills']=checked;v['temporal_findings']=['Bothdynamicmodelsfollowthecoarse stand/lowerlegandcrouch/reachposturechanges in sixsampledstills;staticshapeisunchangedacrosstime.','Earlyraisedfootremains distorted or accompaniedbytranslucentskin. In13/600timecorrection makespartoftheshoe clearer than nominal, but doesnotremoveallartifact, and00/600stillfails.','These fullpersonpanels were downscaled bytheimageviewer,particularly00;finefingerdetailcannotbeacceptedfromthem.'];v['video_review_scope']='Both25framevideos encoded/probed/decoded;6sampledstillsactuallyviewed,notanentirevideo orallframesvisualacceptance.';v['clarity_accepted']=False;p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8');print('6temporalstillsreviewed;fulldetail/occlusionacceptancenotclaimed.')
