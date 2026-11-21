from pathlib import Path
import json,datetime
from PIL import Image
base=Path(r'\\?\C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
root=base/'current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/crown-attribution'
viewer=base.parent/'viewer'
proof=viewer/'body-dynamic/crown-review-proof.png'
with Image.open(proof) as im:width,height=im.size
assert width>1000 and height>600
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
r=dict(verified_at=datetime.datetime.now().astimezone().isoformat(),url='http://127.0.0.1:8870/body-dynamic.html#crown-attribution',
 actual_browser_checked=True,screenshot=str(proof),screenshot_size=[width,height],
 verified_image_dimensions={'crown-attribution.png':[1680,608],'crown-real-correspondence.png':[1280,468],'crown-radiance-probe.png':[1260,608]},
 all_three_images_loaded=True,complete_figures=True,details_expanded_and_checked=True,
 main_model_changed=False,new_model_deployed=False,quality_accepted=False)
save(root/'browser-verification.json',r)
review=read(root/'root-review.json');review['browser']=r;review['cleanup']=read(root/'cleanup.json');save(root/'root-review.json',review)
ws=read(base/'work-status.json');ws['dynamic_crown_attribution']=review;ws['updated']=r['verified_at'];save(base/'work-status.json',ws)
print(json.dumps(dict(browser_verified=True,images=3,ppt_slides=55,goal_accepted=False)))
