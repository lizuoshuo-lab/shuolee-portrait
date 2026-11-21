from pathlib import Path
import json
LAB=Path(__file__).resolve().parents[2];S=LAB/'scripts/dynamic_human';OUT=LAB/'outputs/dynamic-human-enerf-representation-048'
t=(S/'compare_4k4d_source_gate.py').read_text(encoding='utf8')
t=t.replace("ROOT=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044'", "ROOT=LAB/'outputs/dynamic-human-enerf-representation-048'")
t=t.replace("A=ROOT/'original_blending';B=ROOT/'source_support_blending'", "A=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044/source_support_blending';B=ROOT")
t=t.replace("assert read(ROOT/'independent-state-audit.json')['matched_recorded_sampling']", "assert read(ROOT/'independent-state-audit.json')['step']==6000")
t=t.replace("['camera','split','source_cameras','source_rgb_sha256','source_matte_sha256','bbox']", "['camera','source_rgb_sha256','source_matte_sha256','bbox']")
t=t.replace("('Original blend 3000',old),('Source support 3000',new)","('4K4D support blend 3000',old),('Direct3DGS 6000',new)")
t=t.replace('Fixed native photograph and imperfect released-matte reference; only source-support blending rule differs. Same040originalparent2000history,1000newupdates each.',
 'Same native training images/masks, cameras and references. Independent direct3DGS versus4K4Dpoint-IBR pipeline: initialization, representation, objective, optimization and budget differ. Not a single-factor ablation.')
exec(compile(t,'adapted_representation_comparison','exec'))
