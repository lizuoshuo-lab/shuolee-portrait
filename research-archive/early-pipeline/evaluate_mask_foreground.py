from pathlib import Path
import json,datetime,numpy as np
from PIL import Image,ImageDraw
L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee';R=B/'current/person-static-mask-foreground-fresh-070';A=B/'current/static-white-edge-audit-068'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
rgb=read(R/'review/backgrounds/alpha-review.json');old=read(A/'boundary-before/boundary-audit.json');new=read(R/'review/coverage/boundary-audit.json')
assert [v['frame'] for v in old['rows']]==[v['frame'] for v in new['rows']]==[0,23,69,99,135,174]
keys=['inner_boundary_undercovered_fraction','inner_boundary_without_support_fraction','outer_boundary_overcovered_fraction','full_image_alpha_mae']
coverage={label:{k:float(np.mean([v[k] for v in rec['rows']])) for k in keys} for label,rec in [('064',old),('070',new)]}
per_view=[dict(frame=x['frame'],inner_undercoverage_increase_pp=100*(y['inner_boundary_undercovered_fraction']-x['inner_boundary_undercovered_fraction'])) for x,y in zip(old['rows'],new['rows'])]
a=rgb['summary']['064'];b=rgb['summary']['070'];c=coverage['064'];d=coverage['070']
gates=dict(interior_rgb_protected=b['inside_rgb_mae']<=a['inside_rgb_mae']*1.05,
    mean_internal_coverage_protected=100*(d['inner_boundary_undercovered_fraction']-c['inner_boundary_undercovered_fraction'])<=.5,
    every_view_internal_coverage_protected=max(x['inner_undercoverage_increase_pp'] for x in per_view)<=1.,
    boundary_alpha_target=b['boundary_alpha_mae']<=a['boundary_alpha_mae']*.9,
    outer_overcoverage_target=d['outer_boundary_overcovered_fraction']<=c['outer_boundary_overcovered_fraction']*.8)
save(R/'review/numeric-screening.json',dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    rgb=rgb['summary'],coverage=coverage,per_view=per_view,gates=gates,numeric_screening_passed=all(gates.values()),
    visual_review_completed=False,quality_accepted=False,scope='Six fixed observed training masks, not novel-view or geometry GT.',
    criterion_source=str(R/'analysis-before-training.json')))
# Native output crops retain exact camera pixels and identical crop limits.
panels=[];crop_metadata=[]
for i in [0,23,69,99,135,174]:
    p=R/'review/backgrounds';m=np.asarray(Image.open(p/f'observed-{i:03d}-mask.png'))/255
    yy,xx=np.where(m>.5);height=yy.max()-yy.min()+1
    box=(max(0,int(xx.min())-25),max(0,int(yy.min())-20),min(m.shape[1],int(xx.max())+26),min(m.shape[0],int(yy.min()+.33*height)))
    crop_metadata.append(dict(frame=i,crop=list(box)))
    ims=[Image.open(p/f'observed-{i:03d}-source.png'),Image.open(p/f'observed-{i:03d}-064-bg0.00.png'),Image.open(p/f'observed-{i:03d}-070-bg0.00.png')]
    panel=Image.new('RGB',(1200,475),(26,26,26));draw=ImageDraw.Draw(panel)
    for j,(im,label) in enumerate(zip(ims,['real image','064: full ROI RGB','070: foreground RGB only'])):
        crop=im.crop(box);crop.thumbnail((390,430));panel.paste(crop,(j*400+(400-crop.width)//2,35))
        draw.text((j*400+8,10),f'view {i:03d} | {label}',fill='white')
    panels.append(panel)
    panel.save(R/'review'/f'head-{i:03d}.jpg',quality=95)
sheet=Image.new('RGB',(1200,475*len(panels)),(26,26,26))
for j,panel in enumerate(panels):sheet.paste(panel,(0,475*j))
sheet.save(R/'review/head-comparison-all.jpg',quality=95)
save(R/'review/crop-provenance.json',dict(crops=crop_metadata,manipulation='identical pixel crop and display resize only; native renderer outputs',labels=['source','064','070']))
print(json.dumps(dict(rgb=rgb['summary'],coverage=coverage,gates=gates,per_view=per_view),ensure_ascii=False),flush=True)
