from pathlib import Path
import shutil
S=Path(__file__).resolve().parent;OUT=S.parents[1]/'outputs/dynamic-human-enerf-surface-gs-055'
t=(S/'compare_enerf_absgrad.py').read_text().replace('dynamic-human-enerf-absgrad-052','dynamic-human-enerf-surface-gs-055')
t=t.replace("A=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame';B=ROOT", "A=ROOT/'surface_only';B=ROOT/'surface_regularized'")
t=t.replace("'Standard gradient'", "'Surface only'").replace("'Absolute gradient'", "'Surface regularized'")
(S/'compare_enerf_surface_control.py').write_text(t)
for n in ['train_enerf_surface_control.py','prepare_enerf_surface_control.py','prepare_enerf_surface_compare.py']:shutil.copy2(S/n,OUT/'source-snapshot'/n)
print('055comparison and updated training snapshots prepared')
