"""Actual image patch contact sheets for manual correspondence identity review."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from enerf_native_loader import EnerfNativeDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--cases',nargs='*');a=p.parse_args();out=Path(a.out)
    ds=EnerfNativeDataset(a.manifest,'train',allow_calibration_only=True);lookup={(r['camera'],r['frame']):i for i,r in enumerate(ds.rows)}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);records=[]
    cases=[(x.split(':')[0],x.split(':')[1],int(x.split(':')[2]),int(x.split(':')[3])) for x in a.cases] if a.cases else [('09','01',49,1),('01','10',49,-1),('14','16',49,1),('04','03',29,1)]
    for c,d,t,lag in cases:
        v=np.load(out/f'tracks-{c}-{d}-t{t:03d}.npz');speed=v['source_speed'];x=v['source_xy'];y=v['target_xy'];offsets=v['offsets'].tolist();k=offsets.index(lag);zero=offsets.index(0)
        # Fastest mutually matched features, with 40px spatial separation.
        # No sorting/filtering on epipolar error or improvement.
        chosen=[]
        for i in np.argsort(-speed):
            if speed[i]<1:break
            if all(np.linalg.norm(x[i]-x[j])>40 for j in chosen):chosen.append(int(i))
            if len(chosen)==6:break
        obs=[ds[lookup[c,t]],ds[lookup[d,t]],ds[lookup[d,t+lag]]];can=Image.new('RGB',(3*160,len(chosen)*160+36),'white');dr=ImageDraw.Draw(can)
        for j,label in enumerate([f'{c}, frame {t}',f'{d}, frame {t}',f'{d}, frame {t+lag}']):dr.text((j*160+5,8),label,font=font,fill='black')
        for row,i in enumerate(chosen):
            for col,xy in enumerate([x[i],y[i,zero],y[i,k]]):
                xx,yy=np.rint(xy).astype(int);patch=Image.fromarray(obs[col]['rgb']).crop((xx-64,yy-64,xx+64,yy+64));can.paste(patch,(col*160+16,row*160+36));dr.ellipse((col*160+77,row*160+97,col*160+83,row*160+103),outline='red',width=1)
            dr.text((5,row*160+167),f'id {i}, speed {speed[i]:.1f}px/frame',font=font,fill='black')
            records.append(dict(source=c,target=d,anchor=t,lag=lag,match=i,source_xy=x[i].tolist(),speed=float(speed[i]),zero_error=float(abs(v['signed_sampson'][i,zero])),shifted_error=float(abs(v['signed_sampson'][i,k]))))
        can.save(out/f'identity-{c}-{d}-t{t:03d}.png')
    (out/'identity-review-selection.json').write_text(json.dumps(dict(selection='Fastest source features, >40px source separation, no error/improvement gate; manual judgment pending',patches=records),indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
