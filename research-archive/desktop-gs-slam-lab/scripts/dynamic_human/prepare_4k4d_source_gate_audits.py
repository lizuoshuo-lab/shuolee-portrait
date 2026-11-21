from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'audit_4k4d_projection_state.py').read_text(encoding='utf8')
text=text.replace('dynamic-human-4k4d-projection-control-040','dynamic-human-4k4d-source-gate-training-044')
text=text.replace("parent=LAB/'outputs/dynamic-human-4k4d-mask-control-039/model/latest.pt'", "parent=LAB/'outputs/dynamic-human-4k4d-projection-control-040/original_projection/model/latest.pt'")
text=text.replace("['original_projection','calibrated_projection']","['original_blending','source_support_blending']")
text=text.replace("done['history_updates']==2000","done['history_updates']==3000").replace('max(steps)==2000','max(steps)==3000').replace('2000history_checkpoint_verified','3000history_checkpoint_verified')
old="assert configs[1]['projection_offset_file'].endswith('dynamic-human-enerf-static-calibration-037/calibration.json')"
assert old in text;text=text.replace(old,"assert configs[1]['projection_offset_file']==''\nassert not configs[0]['source_foreground_gate'] and configs[1]['source_foreground_gate']")
text=text.replace('Empirical calibration uses separate static photos from all18cameras.','Original projection and alpha retained; source foreground-support rule differs.')
a=text.index("names=['prepare_4k4d_projection_control.py'");b=text.index('\nfor n in names:',a)
text=text[:a]+"names=['prepare_4k4d_source_gate_training.py','train_4k4d_source_gate.py','run_4k4d_source_gate_training.sh','fourkd_source_support.py','check_4k4d_source_support.py','evaluate_4k4d_pilot.py','audit_4k4d_native_images.py','audit_4k4d_source_gate_state.py','compare_4k4d_source_gate.py']"+text[b:]
target=folder/'audit_4k4d_source_gate_state.py';assert not target.exists();target.write_text(text,encoding='utf8')
text=(folder/'compare_4k4d_projection.py').read_text(encoding='utf8')
text=text.replace('dynamic-human-4k4d-projection-control-040','dynamic-human-4k4d-source-gate-training-044').replace("'original_projection'","'original_blending'").replace("'calibrated_projection'","'source_support_blending'")
text=text.replace('Original projection 2000','Original blend 3000').replace('Corrected projection 2000','Source support 3000')
text=text.replace('only empirical projection treatment differs. Same039parent,1000newupdates each.','only source-support blending rule differs. Same040originalparent2000history,1000newupdates each.')
target=folder/'compare_4k4d_source_gate.py';assert not target.exists();target.write_text(text,encoding='utf8')
print('044 checkpoint and paired image auditors prepared')
