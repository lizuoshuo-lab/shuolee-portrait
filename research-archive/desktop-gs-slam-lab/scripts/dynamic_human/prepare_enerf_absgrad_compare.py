from pathlib import Path
S=Path(__file__).resolve().parent
t=(S/'compare_enerf_camera10_time.py').read_text()
t=t.replace("ROOT=LAB/'outputs/dynamic-human-enerf-camera10-time-050'", "ROOT=LAB/'outputs/dynamic-human-enerf-absgrad-052'")
t=t.replace("A=ROOT/'same_frame';B=ROOT/'camera10_previous_frame'", "A=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame';B=ROOT")
t=t.replace("read(ROOT/'fixed-detail-regions.json')", "read(LAB/'outputs/dynamic-human-enerf-camera10-time-050/fixed-detail-regions.json')")
t=t.replace("'Same frame'", "'Standard gradient'").replace("'Camera10 previous'", "'Absolute gradient'")
t=t.replace("ROOT/'independent-state-audit.json'", "ROOT/'paired-training-state-audit.json'")
compile(t,'compare_enerf_absgrad.py','exec');(S/'compare_enerf_absgrad.py').write_text(t)
print('052comparison prepared using unchanged050regions')
