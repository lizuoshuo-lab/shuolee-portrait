from pathlib import Path
build=Path(__file__).resolve().parent
source=(build/'finalize_floor_publication.py').read_text('utf-8')
for before,after in [('floor-deck-receipt','gvhmr025-deck-receipt'),
                     ('floor-final-validation','gvhmr025-final-validation-r2'),
                     ('floor-final-renders','gvhmr025-final-renders'),
                     ("['surface_contact_closeout']","['gvhmr_shared_world_025']")]:
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'))
