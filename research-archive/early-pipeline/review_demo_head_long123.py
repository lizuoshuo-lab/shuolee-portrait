"""Reuse native-pixel review panels for the independent 300k run (CPU only)."""
from pathlib import Path
import audit_gaussianavatars_params123 as a

source = a.lp(a.H / 'review_demo_head_gaussians123.py').read_text(encoding='utf-8')
source = source.replace("OUT=ROOT/'head-gaussian-first-cycle'", "OUT=ROOT/'head-gaussian-long300k'")
source = source.replace("d=OUT/'inputs'/name", "d=Path(r['image'].replace('/mnt/c/', 'C:/')).parent")
exec(compile(source, str(a.H / 'review_demo_head_gaussians123.py'), 'exec'))
