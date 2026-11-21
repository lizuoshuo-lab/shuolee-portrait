from pathlib import Path
build=Path(__file__).resolve().parent
source=(build/'finalize_floor_publication.py').read_text('utf-8')
for before,after in [('floor-deck-receipt','observation026-deck-receipt'),
    ('floor-final-validation','observation026-final-validation'),
    ('floor-final-renders','observation026-final-renders'),
    ("['surface_contact_closeout']","['body_observation_026']")]:
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'))
