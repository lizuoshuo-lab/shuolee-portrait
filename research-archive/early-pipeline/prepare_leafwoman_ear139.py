"""Freeze source-only 022 ear panels before revealing any projections."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from official_smplifyx_entry123 import sha
from replay_leafwoman_arm138 import ARCHIVE, OUT as ARM, T
from update_leafwoman129_state import atomic, read


OUT = ARM.parent/'ear-region139'
MASK = T/'shuolee/current/novel-view-face/head-structure/private-runtime/models/flame-masks/FLAME_masks.pkl'
MAP = MASK.parents[1]/'smplx/correspondences/SMPL-X__FLAME_vertex_ids.npy'


def prepare():
    ear = ARM.parent/'ear-audit'
    manifest_path = ear/'render-manifest.json'
    visibility_path = ear/'ear-visibility-review.json'
    manifest = read(manifest_path)
    visibility = read(visibility_path)
    assert visibility['render_manifest_sha256'] == sha(manifest_path)
    assert manifest['input_sha256'][str(ARCHIVE)] == sha(ARCHIVE)
    assert not (OUT/'plan.json').exists()
    OUT.mkdir(exist_ok=True)
    rows = []
    font = ImageFont.truetype('consola.ttf', 13)
    for side in ('left','right'):
        visible = [r['frame'] for r in visibility['rows']
                   if r['ears'][side]['visibility'] == 'visible_external_ear']
        for frame in (visible[0], visible[len(visible)//2], visible[-1]):
            src, = (r for r in manifest['rows'] if r['frame'] == frame)
            path = Path(src['crop_path'])
            assert sha(path) == src['crop_sha256']
            with Image.open(path) as f:
                raw = f.convert('RGB')
            assert list(raw.size) == src['native_crop_wh']
            w,h = raw.size
            panel = Image.new('RGB', (2*w+46, 2*h+38), '#202530')
            ox, oy = 28, 26
            panel.paste(raw.resize((2*w, 2*h), Image.Resampling.NEAREST), (ox,oy))
            draw = ImageDraw.Draw(panel)
            for x in range(0,w,10):
                xp = ox+2*x
                draw.line((xp,oy-4,xp,oy-1),fill='#eee')
                if x%20==0: draw.text((xp-7,3), str(x), font=font, fill='#eee')
            for y in range(0,h,10):
                yp = oy+2*y
                draw.line((ox-4,yp,ox-1,yp),fill='#eee')
                if y%20==0: draw.text((0,yp-7),str(y),font=font,fill='#eee')
                # Small sparse markers on the duplicate display aid approximate annotation.
                for x in range(0,w,10):
                    xp = ox+2*x
                    draw.point((xp,yp),fill='#23f6fb')
                    draw.point((xp+1,yp),fill='#162430')
            dst = OUT/f'{frame}-source-only.png'
            panel.save(dst)
            rows.append(dict(frame=frame, side=side, original_crop=str(path), crop_sha256=sha(path),
                             crop_xyxy=src['crop_original_xyxy'], display_sha256=sha(dst), display=str(dst)))
    atomic(OUT/'plan.json',dict(frames=rows, archive_sha256=sha(ARCHIVE), ear_mask_sha256=sha(MASK),
        correspondence_sha256=sha(MAP), visibility_sha256=sha(visibility_path), manifest_sha256=sha(manifest_path),
        selection='For each visible side use chronological first, middle (floor(N/2)), last. No residual ranking.',
        annotation='On source only, mark rough visible pinna polygon without tracing hair or hidden portions, or mark unresolvable. Record uncertainty in crop pixels; add native integer crop offset. Freeze polygons before comparison.',
        comparison='Compare RTMW and pinned old SMPL-X ear vertex against source polygon. Use official FLAME pinna labels and FLAME-to-SMPLX index map to check the whole frozen ear region with mesh self-occlusion. No vertex identity or 3D ground truth claim.'))
    print(OUT)


if __name__=='__main__':
    prepare()
