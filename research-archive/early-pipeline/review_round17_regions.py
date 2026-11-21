"""Independent eye and hair ROIs traced on the R15/R16 native source RGB."""

from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Native integer-center coordinates. Eyes sample visible iris interiors, including
# through spectacles. Hair samples visible dark hair interiors, clear of the fringe.
POLYGONS = {
    '016': {
        'frame_000027': dict(eye=[[(1328,572),(1337,571),(1338,576),(1333,578),(1329,577)],
                                  [(1382,572),(1390,571),(1391,576),(1387,578),(1383,577)]],
                             hair=[[(1377,478),(1400,482),(1410,494),(1414,515),(1396,508),(1382,493)]]),
        'frame_000051': dict(eye=[[(1376,556),(1382,554),(1386,556),(1386,561),(1380,562),(1376,560)],
                                  [(1420,555),(1426,553),(1429,555),(1429,560),(1425,561),(1421,559)]],
                             hair=[[(1332,480),(1350,473),(1367,477),(1362,492),(1345,505),(1323,511),(1323,496)]]),
        'frame_000002': dict(eye=[[(1437,554),(1443,553),(1447,556),(1447,559),(1443,562),(1439,561)]],
                             hair=[[(1471,479),(1491,474),(1513,480),(1523,499),(1515,516),(1495,513),(1477,498)]]),
    },
    '006': {
        'frame_000032': dict(eye=[[(1345,491),(1350,489),(1353,491),(1352,496),(1349,498),(1345,496)],
                                  [(1399,492),(1405,491),(1407,494),(1406,498),(1402,500),(1399,497)]],
                             hair=[[(1314,429),(1325,420),(1335,422),(1337,431),(1328,439),(1318,440)]]),
        'frame_000041': dict(eye=[[(1368,445),(1373,442),(1377,444),(1377,449),(1373,452),(1369,449)]],
                             hair=[[(1294,395),(1307,392),(1322,396),(1328,408),(1325,419),(1311,423),(1299,413)]]),
        'frame_000049': dict(eye=[[(1401,416),(1406,412),(1410,413),(1411,417),(1407,420),(1403,420)]],
                             hair=[[(1296,347),(1309,343),(1321,349),(1328,360),(1322,370),(1307,366),(1299,358)]]),
    },
}


def regions(subject, pair, pin, sha, mask_digest, *, rgb=None):
    """Build foreground polygons from fixed source coordinates and check the independent mask."""
    frame = pair['frame']
    polygons = POLYGONS[subject][frame]
    source = pin(pair['source'],pair['source_sha256'])
    matte = pin(pair['score_mask'],pair['score_mask_sha256'])
    if rgb is None:
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB'))
    with Image.open(matte) as image:
        inside = np.asarray(image.convert('L')) >= 253
    wh = tuple(pair['camera']['wh'])
    if inside.shape != rgb.shape[:2] or inside.shape[::-1] != wh:
        raise ValueError('Native RGB and independent mask grids differ.')
    masks, observations = {}, {}
    for name, paths in polygons.items():
        canvas = Image.new('1',wh)
        draw = ImageDraw.Draw(canvas)
        for path in paths:
            draw.polygon(path,fill=1)
        mask = np.asarray(canvas)
        if mask.sum() < 32 or not inside[mask].all():
            raise ValueError(f'{subject} {frame} {name} annotation is outside reliable independent foreground.')
        masks[f'dark_source_{name}'] = mask
        light = np.asarray(rgb[mask],dtype=np.float64) @ [.2126,.7152,.0722] / 255.
        observations[name] = dict(pixels=int(mask.sum()),source_luma_p10_p50_p90=np.quantile(light,[.1,.5,.9]).tolist(),
            polygons=paths,sha256=mask_digest(mask))
    return masks, dict(frame=frame, source_sha256=sha(source), score_mask_sha256=sha(matte),
                       grid='native pixel integer centers', regions=observations)


def identity():
    path = Path(__file__)
    return dict(file=str(path),sha256=sha256(path.read_bytes()).hexdigest(),
                method='manual source RGB polygons pinned before R17 rendering; 253/255 independent foreground check',
                r17_model_render_seen_at_annotation=False)
