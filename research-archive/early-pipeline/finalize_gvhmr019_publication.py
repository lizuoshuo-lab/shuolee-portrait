from pathlib import Path

# Preserve the established deck's first 54 slides, notes, media, and rendered
# pixels before replacing its one canonical file. All checks stay enabled.
build = Path(__file__).resolve().parent
source = (build / 'finalize_floor_publication.py').read_text('utf-8')
for before, after in [
    ('floor-deck-receipt', 'gvhmr019-deck-receipt'),
    ('floor-final-validation', 'gvhmr019-final-validation'),
    ('floor-final-renders', 'gvhmr019-final-renders'),
    ("['surface_contact_closeout']", "['gvhmr_prior_019']"),
]:
    source = source.replace(before, after)
exec(compile(source, str(__file__), 'exec'))
