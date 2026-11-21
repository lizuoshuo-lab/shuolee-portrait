from pathlib import Path
build=Path(__file__).resolve().parent
source=(build/'finalize_floor_publication.py').read_text('utf-8')
for before,after in [('floor-deck-receipt','gvhmr023r3-deck-receipt'),
                     ('floor-final-validation','gvhmr023r3-final-validation'),
                     ('floor-final-renders','gvhmr023r3-final-renders'),
                     ("['surface_contact_closeout']","['gvhmr_motion_023']")]:
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'))
