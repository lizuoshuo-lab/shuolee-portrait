from pathlib import Path

# Use the established preservation checks; only the one edited slide may differ.
build=Path(__file__).resolve().parent
source=(build/'finalize_floor_publication.py').read_text('utf-8')
for before,after in [
    ('floor-deck-receipt','gvhmr021-deck-receipt'),
    ('floor-final-validation','gvhmr021-final-validation'),
    ('floor-final-renders','gvhmr021-final-renders'),
    ("['surface_contact_closeout']","['gvhmr_articulation_021']"),
]:
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'))
