from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-snapshot-091';p=L/'manifests/dynamic-human-work.json';read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert read(O/'website-qa.json')['new_anchor_verified'];proof=O/'website-visible.jpg';assert proof.exists()
save(O/'website-visible-proof.json',dict(file=proof.name,sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),browser='existingIABtab9',image_native_size=[640,448],image_displayed_width=581,loaded=True,whole_comparison_and_ray_formula_visually_verified=True,viewport_override=False))
m=read(p);m.update(status='087to091complete_verified_site_updated_geometry_correspondence_next',current_runs=[],latest_verified_site='outputs/'+O.name+'/website-qa.json',latest_visible_result='http://localhost:8799/gs-lab/read/dynamic-human-scene-reconstruction#为什么单视角拟合不等于三维重建',website_preview_pid=45796,previous_interval_reference='outputs/dynamic-human-enerf-residual-graph-067/residual_graph',current_best_continuous_control='outputs/dynamic-human-enerf-second-interval-079/local_bspline',current_temporal_candidate='outputs/dynamic-human-enerf-second-interval-079/local_bspline',outdoor_hand_consistency_website_qa='outputs/'+O.name+'/website-qa.json',next_actions=['保留079作为当前720–732区间的开发对照；088、089、091均未通过人物清晰度验收。','检查近同步训练机位04／17在732帧可见手部的跨视角对应和射线交会；不把估计指尖与掩码当真值。','依据几何核验决定局部表面初始化或约束修改，先过多视角再过原120个跨时刻案例及连续画面。','人物清晰度通过后，再推进独立场景和静态背景联合重建。'],updated_at=datetime.now(timezone.utc).isoformat())
save(p,m)
for n in ['publish_enerf_hand_consistency.py','audit_enerf_hand_consistency_website.py','record_enerf_hand_consistency_delivery.py']:shutil.copy2(L/'scripts/dynamic_human'/n,O/n)
print('Final site verified; no GPU runs pending; current development baseline079; clarity goal active.')
