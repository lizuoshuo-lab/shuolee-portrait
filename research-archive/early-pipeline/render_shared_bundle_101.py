from pathlib import Path
import ast
src=Path(__file__).with_name('review_named_points_099.py').read_text(encoding='utf-8')
src=src.replace("O=B/'current/shared-point-ray-consistency-099';J=O/'joint-camera-points'", "O=B/'current/shared-point-bundle-convergence-101';J=O")
ast.parse(src)
exec(compile(src,str(__file__),'exec'))
