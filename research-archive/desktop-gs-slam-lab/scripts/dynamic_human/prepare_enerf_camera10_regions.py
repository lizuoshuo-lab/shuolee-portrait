from pathlib import Path
import json,hashlib
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-camera10-time-050'
regions={
'00':{'dark_forearm':[762,446,830,590],'light_forearm':[946,395,1090,534],'right_hand':[1248,425,1324,594],
      'airborne_shoe':[920,760,1017,875],'shirt_print':[805,390,920,533],'held_objects':[944,471,1099,617]},
'06':{'crossed_forearms':[1030,377,1186,582],'back_forearm':[869,354,961,458],'airborne_shoe':[1173,756,1289,872],
      'shirt_folds':[933,220,1082,421],'held_objects':[871,433,965,565]},
'12':{'dark_forearm':[885,461,960,604],'light_forearm':[993,417,1099,533],'right_hand':[1267,440,1325,616],
      'airborne_shoe':[1080,787,1160,894],'shirt_print':[936,406,1026,543],'held_objects':[997,493,1112,626]}}
p=OUT/'fixed-detail-regions.json';assert not p.exists()
photos=LAB/'outputs/dynamic-human-enerf-representation-048/native-evaluation'
p.write_text(json.dumps(dict(status='regions_fixed_from_original750photographs_before_viewing050candidate',coordinates='Native1920x1080, x0 y0 x1 y1, endexclusive',
  regions=regions,reference_photos={c:hashlib.sha256((photos/f'camera{c}-frame000750-source.png').read_bytes()).hexdigest() for c in regions},
  qualification='Hand-selected diagnostic boxes on this known development case, not independent semantic masks. Metrics against imperfect released-matte reference; raw photograph included in every image.',goal_complete=False),indent=2)+'\n')
print('17 detail regions fixed on raw photographs')
