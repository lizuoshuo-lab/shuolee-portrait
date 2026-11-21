"""Repeat first failed interpolation probes without rendering or changing source."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/large_scene_gs'))
import evaluate_official_hierarchy as e
from read_hierarchy import read_hierarchy
import torch
OUTPUT=Path(__file__).resolve().parent
FIRST=e.OUTROOT/'truck-official-continuous-heldout-001'
def main():
 started=time.perf_counter();allrows=[]
 source=e.source_audit(OUTPUT)
 cams=e.read_json(FIRST/'official-camera-audit.json')['per_image']
 for model in ('baseline-bridge','merged-unoptimized','merged-post'):
  audit=e.read_json(FIRST/model/'model-audit.json');path=Path(audit['source_path'])
  e.require(e.digest(path)==audit['source_sha256'],'Source changed')
  out=OUTPUT/model;out.mkdir()
  h=read_hierarchy(path);gh,tensors,leaves=e.author_cpu_load(h,out)
  nodes=tensors[5].to('cuda');boxes=tensors[6].to('cuda');zero=torch.zeros(3)
  probes=[]
  for setting in ('tau1','tau3','tau6','tau15'):
   rows=e.read_json(FIRST/model/setting/'per-frame-audit.json')
   j=next((i for i,x in enumerate(rows) if x['status']=='invalid_interpolation_no_render'),0)
   row=rows[j];selpath=Path(row['selected']['path']);e.require(e.digest(selpath)==row['selected']['file_sha256'],'Saved cut changed')
   sel=np.load(selpath);ids=sel['node_ids'];indices=torch.from_numpy(ids).to('cuda')
   weights=torch.empty(len(ids),device='cuda');kids=torch.empty(len(ids),dtype=torch.int32,device='cuda')
   camera=torch.tensor(cams[j]['camera_center'],dtype=torch.float32)
   gh.get_interpolation_weights(indices,row['threshold'],nodes,boxes,camera,zero,weights,kids)
   torch.cuda.synchronize();actual=weights.cpu().numpy();siblings=kids.cpu().numpy()
   e.require(e.array_digest(actual)==row['interpolation_weight_bytes_sha256'],'Repeated author weights differ from first run')
   e.require(e.array_digest(siblings)==row['sibling_count_bytes_sha256'],'Repeated author child counts differ')
   filename=out/f'{setting}-{j:02d}-{Path(row["name"]).stem}-actual-weights.npz'
   np.savez(filename,child_weights=actual,num_parent_children=siblings,node_ids=ids)
   ok,details=e.weight_audit_CPU(h,ids,camera.numpy(),row['threshold'],actual,siblings)
   probe=dict(model=model,setting=setting,original_status=row['status'],name=row['name'],theta=row['threshold'],
    author_weight_bytes_exactly_reproduced=True,actual_path=str(filename),actual_sha256=e.digest(filename),
    interval_check=details,status='passed_derived_float32_interval_and_negative_control' if ok else 'interval_failure_preserved')
   probes.append(probe);allrows.append(probe);e.save(out/'probes.json',probes)
   print(json.dumps(dict(model=model,setting=setting,accepted=ok,error=details['canonical_separate_float32_abs_difference'],width=details['interval_width'])),flush=True)
   del indices,weights,kids
  del nodes,boxes,tensors,h,leaves;torch.cuda.empty_cache()
 e.save(OUTPUT/'outcome.json',dict(status='all12_probes_within_derived_float32_intervals' if all(x['interval_check']['accepted'] for x in allrows) else 'failures_preserved',
  elapsed_seconds=time.perf_counter()-started,source_audit=source,script_sha256=e.digest(__file__),probes=allrows,
  scope='Actual unmodified author GPU weight generation only, no rendering. Exact input cut and weight bytes reproduced from first run; prior partial run immutable.'))
if __name__=='__main__':main()
