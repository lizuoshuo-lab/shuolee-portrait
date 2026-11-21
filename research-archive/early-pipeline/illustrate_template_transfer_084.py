import json,pathlib,numpy as np
from PIL import Image,ImageDraw,ImageFont
B=pathlib.Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
O=B/'human-surface-attribution-084';R=B/'person-static-human-surface-init-fresh-083'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=read(R/'data/training-inputs.json')['records'];records=read(O/'template-transfer.json')['records']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
def image(row):return Image.open('C:'+row['image'][6:]).convert('RGB')
cases=[]
for frame in [82,100,136]:
 for kind in ['lower_texture','upper_texture']:
  values=[r for r in records if r['raw_frame']==frame and r['kind']==kind]
  if values:cases.append(values[0])
for r in cases:
 p=np.array(r['source_pixel_full']);q=np.array(r['reference_pixel_full']);views=[]
 for j,(row,center) in enumerate([(rows[r['raw_frame']-1],p),(rows[r['reference_frame']-1],q)]):
  origin=np.rint(center-[130,145]).astype(int);im=image(row).crop((*origin,*(origin+[260,290])));d=ImageDraw.Draw(im)
  target=center-origin;d.ellipse((*list(target-4),*list(target+4)),outline=(0,230,255),width=2)
  if j==1:
   for name,color in [('template_prediction',(255,170,0)),('gaussian_prediction',(255,30,40))]:
    x=np.array(r[name])-origin;d.line((target[0],target[1],x[0],x[1]),fill=color,width=1)
    d.line((x[0]-4,x[1],x[0]+4,x[1]),fill=color,width=2);d.line((x[0],x[1]-4,x[0],x[1]+4),fill=color,width=2)
  views.append(im)
 board=Image.new('RGB',(780,355),'white');d=ImageDraw.Draw(board)
 board.paste(views[0],(0,55));board.paste(views[1],(260,55))
 d.text((5,7),f'query{r["index"]} | {r["kind"]}',font=font,fill='black')
 d.text((5,30),f'source raw{r["raw_frame"]}',font=font,fill='black');d.text((265,30),f'reference raw{r["reference_frame"]}',font=font,fill='black')
 text=['Cyan: tracked observation','Amber: neutral surface','Red: 083 center ray',f'template: {r["template_error"]:.2f} px',f'Gaussian: {r["gaussian_center_ray_error"]:.2f} px',f'triangulated: {r["triangulated_error"]:.2f} px','Same full-image pixel units','No photo warp']
 for i,line in enumerate(text):d.text((528,60+i*28),line,font=font,fill='black')
 board.save(O/f'transfer-{r["index"]:03d}.jpg',quality=97)
(O/'transfer-visual-selection.json').write_text(json.dumps({'selection':'First stored query per raw82/100/136 and lower/upper group; no error ranking','records':cases},indent=2),encoding='utf-8')
print([(r['index'],r['raw_frame'],r['kind']) for r in cases])
