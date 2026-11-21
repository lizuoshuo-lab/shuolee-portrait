"""Render fixed native A/B paths and compact real-view comparisons for review."""
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from shuolee_da3_refinement import OUT, ROOT, SOURCE, PIPE, camera, load_model, render_imp, GaussianModel, sha, write_json


def load_arm(arm):
    pc=GaussianModel(3);pc.load_ply(str(OUT/arm/'model.ply'))
    pc._mip_iteration=18000;pc._mip_cameras=[]
    pc._mip_filter=torch.load(OUT/arm/'filter_3d.pt',map_location='cuda',weights_only=True)
    pc._mip_cache_key=(id(pc._xyz),len(pc.get_xyz),180);pc._mip_refreshes=0
    return pc


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    summary=json.loads((OUT/'summary.json').read_text());assert summary['status']=='complete_pending_visual_review'
    dest=OUT/'review';dest.mkdir(exist_ok=False)
    path_file=ROOT/'shuolee/camera-paths.json';paths=json.loads(path_file.read_text())
    protocol=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_sha256=sha(Path(__file__)),
        paths_sha256=sha(path_file),views_per_kind=60,reference='Pre-existing shuolee camera paths, unchanged after trial',
        renderer='Native Mini-Splatting2 Mip rasterizer, raw trained model with frozen3Dfilter and view-dependent2Dcompensation',
        anonymous_panels={'A':'da3_guided','B':'real_only'},blind_review=False,
        independent_ground_truth=False,accepted_as_best=False)
    write_json(dest/'protocol.json',protocol)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    head_panels=[]
    for name in ['000000','000016','000032','000048']:
        tiles=[]
        for arm,label in [('initial','Initial model'),('real_only','RGB control'),('da3_guided','DA3 candidate')]:
            im=Image.open(OUT/arm/'evaluation'/(name+'-head.png')).convert('RGB')
            # Preserve source pixel scale in the downloadable comparison.
            tile=Image.new('RGB',(im.width,im.height+34),'white');tile.paste(im,(0,34));ImageDraw.Draw(tile).text((8,8),label,fill='black',font=font);tiles.append(tile)
        panel=Image.new('RGB',(sum(t.width for t in tiles),max(t.height for t in tiles)),'white')
        x=0
        for tile in tiles:panel.paste(tile,(x,0));x+=tile.width
        panel.save(dest/f'observed-{name}.png')
    models={'A':load_arm('da3_guided'),'B':load_arm('real_only')}
    bg=torch.zeros(3,device='cuda')
    masks={key:torch.zeros(len(pc.get_xyz),device='cuda',dtype=torch.bool) for key,pc in models.items()}
    for kind in ['distance','elevation']:
        selected=[r for r in paths if r['kind']==kind];assert len(selected)==60
        w,h=selected[0]['width'],selected[0]['height'];vh=h+36
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{w*2}x{vh}','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/(kind+'.mp4'))]
        with (dest/(kind+'-encode.log')).open('w') as log:
            process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            try:
                for row in selected:
                    c=camera(row,scale=1);panel=Image.new('RGB',(w*2,vh),'white');draw=ImageDraw.Draw(panel)
                    for j,(key,pc) in enumerate(models.items()):
                        image=render_imp(c,pc,PIPE,bg,culling=masks[key])['render'].clamp(0,1)
                        im=Image.fromarray((image*255).round().byte().permute(1,2,0).cpu().numpy())
                        panel.paste(im,(j*w,36));draw.text((j*w+12,8),f'{key} | {kind} {row["index"]:02d}',fill='black',font=font)
                    process.stdin.write(np.asarray(panel).tobytes())
                    if row['index'] in [0,15,45]:panel.save(dest/f'{kind}-{row["index"]:02d}.png')
            finally:process.stdin.close()
            assert process.wait()==0
        print(kind,'complete',flush=True)
    a=summary['arms']['da3_guided']['evaluation'];b=summary['arms']['real_only']['evaluation']
    delta={region:{key:a[region][key]-b[region][key] for key in b[region]} for region in b}
    write_json(dest/'summary.json',dict(status='ready_for_visual_review',protocol=protocol,delta_vs_rgb_control=delta,
        accepted_as_best=False,human_ratings=0,frame_interpretation='A/B labels fixed for reproducibility; metadata reveals mapping, therefore no completed blinded human study is claimed.'))
    print(json.dumps(delta,indent=2),flush=True)


if __name__=='__main__':main()
