from pathlib import Path

# Reuse the previously verified preservation check with new, unique receipts.
# It compares slide XML, media, notes and all first54 rendered slides before
# replacing the one canonical file. No layout or preservation check is waived.
build=Path(__file__).resolve().parent
source=(build/'finalize_floor_publication.py').read_text('utf-8')
for before,after in [('floor-deck-receipt','convergence018-deck-receipt'),
 ('floor-final-validation','convergence018-final-validation'),
 ('floor-final-renders','convergence018-final-renders'),
 ("['surface_contact_closeout']","['motion_convergence_closeout']")]:
    source=source.replace(before,after)
exec(compile(source,str(__file__),'exec'))
