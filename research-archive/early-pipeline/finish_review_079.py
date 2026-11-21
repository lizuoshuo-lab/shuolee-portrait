"""Commit manually observed browser/deck checks, then publish reviewed079."""
from pathlib import Path
import json,datetime,shutil
import publish_surface_079 as p
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
p.save(p.R/'review/viewer-interaction-review.json',dict(checked_at=now,manual_open_file=True,
    gaussian_mode_seen=True,point_mode_seen=True,points_displayed=305174,
    local_standalone_file=str(p.V/'models/whole-body-static-surface/shuolee_079_s18000_f0100.ply'),
    source='Actual Open File chooser in local browser, not only a preset URL',
    up_axis='model default +Y',follow_latest_restored=True,local_geometry_sanity='upright full body without skinning',
    quality_accepted=False,pc00_opened_this_round=False))
p.save(p.W/'surface079-deck-visual-review.json',dict(checked_at=now,actual_render_inspection=True,
    slides_inspected_full_resolution=[51,52],no_clipping_or_overlap=True,
    source_images=['observed077-final-51.png','depth078-final-52.png'],
    final_images=['surface079-final-51.png','surface079-final-52.png'],
    unchanged_slide_render_hashes_verified=50,total_slides=52,
    source_style_preserved=True,limitations_on_last_slide=True,opened_in_PowerPoint=False))
p.main()
for n in ['review_person_079.sh','compare_transfer_079.py','record_surface_review_079.py','publish_surface_079.py','update_deck_079.mjs','finish_review_079.py']:
    shutil.copy2(p.W/n,p.R/'source-final'/n)
