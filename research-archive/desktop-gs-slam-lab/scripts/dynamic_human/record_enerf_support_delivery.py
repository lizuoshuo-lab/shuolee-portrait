from pathlib import Path
from datetime import datetime,timezone
import json
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-support-correction-081'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert read(O/'website-qa.json')['status']=='two_pages_and_seven_new_assets_HTTP_verified'
assert (O/'website-visible.png').exists()
save(O/'website-visual-review.json',dict(status='updated_knowledge_section_actual_browser_verified',screenshot='website-visible.png',observed='Reloaded existing tab15 after build; browsed to updated foreground-support section. New native hand comparison loaded; paragraph, small source caption and evidence link visible without clipping. DOM reported zero KaTeX errors.',production_build='passed',preview_parent_pid=6184,goal_complete=False))
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(latest_verified_site='outputs/'+O.name+'/website-qa.json',latest_visible_result='http://localhost:8799/gs-lab/read/dynamic-human-scene-reconstruction#人物轮廓与背景变化怎样区分',current_runs=[],updated_at=datetime.now(timezone.utc).isoformat());m['goal_turn_classification']+=' Existing knowledge/evidence updated, production build passed, bothpages and7newasset bytes verified overHTTP, actualbrowser image/text inspected.';save(p,m);print('Delivery verified; research goal remains active and clarity unaccepted.')
