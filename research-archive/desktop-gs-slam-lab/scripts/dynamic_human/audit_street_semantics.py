"""Independently verify all saved street mask products; no accuracy claim."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parents[2]
OUT = LAB/'outputs/dynamic-human-street-005'
SCENE = LAB/'data/dynamic-human/argoverse2/processed/training/000'


def main():
    completion = json.loads((OUT/'semantic-completion.json').read_text())
    protocol = OUT/'semantic-protocol.json'
    assert completion['protocol_sha256'] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert completion['images'] == 1092 and completion['mask_files'] == 6552
    files = sorted((SCENE/'images').glob('*.jpg'))
    rows = {r['image']:r for r in completion['per_image']}
    assert set(rows) == {p.name for p in files}
    classes = np.zeros(19, np.int64)
    counts = []
    hashes = []
    groups = [('sky_masks', [10]), ('person_masks', [11]),
        ('fine_dynamic_masks/all', [11,12,13,14,15,17,18]),
        ('fine_dynamic_masks/human', [11,12,17,18]), ('fine_dynamic_masks/vehicle', [13,14,15])]
    for i, path in enumerate(files):
        label_path = SCENE/'semantic_labels'/f'{path.stem}.png'
        with Image.open(path) as rgb:
            size = rgb.size
        with Image.open(label_path) as source:
            assert source.mode == 'L' and source.size == size
            labels = np.array(source)
        assert labels.dtype == np.uint8 and labels.max() <= 18
        digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
        assert rows[path.name]['labels_sha256'] == digest
        classes += np.bincount(labels.ravel(), minlength=19)
        item = dict(image=path.name, labels_sha256=digest, masks={})
        for folder, values in groups:
            target = SCENE/folder/f'{path.stem}.png'
            with Image.open(target) as mask_image:
                assert mask_image.size == size and mask_image.mode == 'L'
                mask = np.asarray(mask_image)
            np.testing.assert_array_equal(mask, np.isin(labels, values).astype(np.uint8)*255)
            item['masks'][folder] = hashlib.sha256(target.read_bytes()).hexdigest()
        hashes.append(item)
        people = int((labels == 11).sum())
        sky = int((labels == 10).sum())
        assert people == rows[path.name]['person_pixels'] and sky == rows[path.name]['sky_pixels']
        counts.append(dict(image=path.name, person_pixels=people, sky_pixels=sky))
        if i % 100 == 0:
            print('VERIFIED MASK SET', i+1, '/', len(files), flush=True)
    expected_stems = {p.stem for p in files}
    for folder in ['semantic_labels']+[name for name,_ in groups]:
        assert {p.stem for p in (SCENE/folder).glob('*.png')} == expected_stems
    (OUT/'semantic-file-integrity.json').write_text(json.dumps(hashes, indent=2)+'\n')
    report = dict(status='all_saved_mask_products_verified_visual_review_separate', images=len(files),
        mask_files=len(files)*6, masks_pixelwise_equal_to_recorded_class_groups=True,
        native_dimensions_preserved=True, class_pixel_counts=classes.tolist(),
        images_with_person_pixels=sum(r['person_pixels']>0 for r in counts),
        completion_sha256=hashlib.sha256((OUT/'semantic-completion.json').read_bytes()).hexdigest(),
        file_integrity_sha256=hashlib.sha256((OUT/'semantic-file-integrity.json').read_bytes()).hexdigest(),
        limits='This validates saved outputs and class mapping, not semantic accuracy. Predictions are not instance masks or evaluation ground truth.')
    (OUT/'semantic-output-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
