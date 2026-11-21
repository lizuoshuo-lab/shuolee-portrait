from pathlib import Path
import json,datetime
P=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/static-ncc-paired-fresh-093')
d=dict(reviewed=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),purpose='Check alternatives without restarting another model or changing the active paired experiment',
 sources=[
  dict(name='Gaussian Splatting on the Move',url='https://github.com/SpectacularAI/3dgs-deblur',
   verified='Official repository supports ordinary-video motion-blur compensation or rolling-shutter compensation. Simultaneous correction requires known exposure and readout times; Spectacular Rec is the recommended capture route.',
   relevance='A time-varying camera during one exposure differs from refining one fixed pose per frame.',
   limitation='No demonstrated rolling-shutter root cause in this subject. Prior080 row-dependent projection did not pass independent face/background protection. New model complexity is not justified solely by a blurred face.',
   action='Reference only. No installation, no data upload, no new training.'),
  dict(name='3DGUT / 3DGRUT',url='https://github.com/nv-tlabs/3dgrut',
   verified='Official implementations support distorted camera models and time-dependent projection, including rolling shutter. The current repository exposes image masks and Gaussian scene export.',
   relevance='Candidate projection implementation if independent raw-image evidence establishes a camera-model limitation.',
   limitation='Support for a camera model does not estimate the correct calibration automatically or resolve genuine per-frame subject inconsistency. Compatibility of its kernels, opacity and SH with the present viewer must be verified separately.',
   action='Reference only. Do not replace the static-person pipeline based on a paper claim.')],
 current_action='Finish093 matched NCC on/off comparison, examine early/late geometry and preserve full observation coverage.',
 camera_ground_truth_available=False,rolling_shutter_proven=False,new_gaussian_training=False,quality_accepted=False)
(P/'camera-alternatives-review.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(saved=str(P/'camera-alternatives-review.json'),new_training=False)))
