from pathlib import Path
import shutil
S=Path(__file__).resolve().parent;OUT=S.parents[1]/'outputs/dynamic-human-enerf-surface-long-056'
t=(S/'compare_enerf_surface_control.py').read_text(encoding='utf8')
t=t.replace("ROOT=LAB/'outputs/dynamic-human-enerf-surface-gs-055'","ROOT=LAB/'outputs/dynamic-human-enerf-surface-long-056'")
t=t.replace("A=ROOT/'surface_only';B=ROOT/'surface_regularized'","A=LAB/'outputs/dynamic-human-enerf-surface-gs-055/surface_regularized';B=ROOT")
t=t.replace("read(ROOT/'paired-training-state-audit.json')['status'].startswith('paired6000updates_exact')","read(ROOT/'training-state-audit.json')['status'].startswith('30000updates_same_initial_inputs')")
t=t.replace("'Surface only'","'Surface 6000'").replace("'Surface regularized'","'Surface 30000'")
compile(t,'compare_enerf_surface_long.py','exec');(S/'compare_enerf_surface_long.py').write_text(t,encoding='utf8')
shutil.copy2(S/'compare_enerf_surface_long.py',OUT/'source-snapshot/compare_enerf_surface_long.py')
