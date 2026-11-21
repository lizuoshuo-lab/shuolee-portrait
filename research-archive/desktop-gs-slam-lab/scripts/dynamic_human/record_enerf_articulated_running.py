from pathlib import Path
import json
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];p=LAB/'manifests/dynamic-human-work.json';m=json.loads(p.read_text(encoding='utf8'))
m['current_runs']=[dict(kind='exec_session',session_id=81398,script='scripts/dynamic_human/run_enerf_articulated_control.sh',output='outputs/dynamic-human-enerf-articulated-control-064',planned='two15000updatebranches_and102nativeevaluations_each')]
m['status']='continuous_body_priors_fitted_actual_paired_articulated_Gaussian_training_running';m['outdoor_continuous_body']='outputs/dynamic-human-enerf-continuous-body-063/completion.json';m['outdoor_articulated_control']='outputs/dynamic-human-enerf-articulated-control-064/protocol.json';m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();m['goal_turn_classification']='Progress:063two2500updatecontinuousSMPLfits and195affinesperbranchcompleted;064nativeinputs/CUDAgradientidentityandbindingtestscomplete; actualpaired15000trainingrunning.'
p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
