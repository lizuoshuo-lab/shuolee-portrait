"""SegFormer B5 semantic preprocessing, preserving native pedestrian pixels.

Uses NVIDIA's HF checkpoint and the official 1024 crop / 768 stride pattern.
Local adapter uses native-resolution RGB tiles and FP16 inference; this is not
claimed to be bitwise-equivalent to the legacy MMSeg augmentation pipeline.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import SegformerForSemanticSegmentation

LAB = Path(__file__).resolve().parents[2]
OUT = LAB/'outputs/dynamic-human-street-005'
SCENE = LAB/'data/dynamic-human/argoverse2/processed/training/000'
MODEL = LAB/'models/segformer-b5-cityscapes'
CROP, STRIDE = 1024, 768


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(model, image):
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    sums = np.zeros((19, h, w), np.float32)
    counts = np.zeros((h, w), np.float32)
    mean = torch.tensor([.485, .456, .406], device='cuda').view(1,3,1,1)
    std = torch.tensor([.229, .224, .225], device='cuda').view(1,3,1,1)
    for iy in range(math.ceil(max(h-CROP, 0)/STRIDE)+1):
        y1 = min(iy*STRIDE+CROP, h); y0 = max(y1-CROP, 0)
        for ix in range(math.ceil(max(w-CROP, 0)/STRIDE)+1):
            x1 = min(ix*STRIDE+CROP, w); x0 = max(x1-CROP, 0)
            tensor = torch.from_numpy(rgb[y0:y1, x0:x1].copy()).permute(2,0,1)[None].cuda().float()/255
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(pixel_values=(tensor-mean)/std).logits
                logits = torch.nn.functional.interpolate(logits.float(), size=(y1-y0,x1-x0), mode='bilinear', align_corners=False)
            assert torch.isfinite(logits).all()
            sums[:, y0:y1, x0:x1] += logits[0].cpu().numpy()
            counts[y0:y1, x0:x1] += 1
            del tensor, logits
    assert counts.min() >= 1
    sums /= counts[None]
    return sums.argmax(0).astype(np.uint8)


def preview(image, labels, path):
    rgb = np.asarray(image.convert('RGB'))
    overlay = rgb.copy()
    for label, color in [(10, [40, 160, 255]), (11, [230, 80, 50])]:
        mask = labels == label
        overlay[mask] = (rgb[mask]*.45 + np.array(color)*.55).astype(np.uint8)
    panels = []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
    for source, title in [(image, 'Source image'), (Image.fromarray(overlay), 'Predicted person (orange) / sky (blue)')]:
        im = source.copy(); im.thumbnail((960, 760), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (980, 825), 'white');panel.paste(im, ((980-im.width)//2, 52))
        ImageDraw.Draw(panel).text((15, 13), title, font=font, fill='#182c36')
        panels.append(panel)
    canvas = Image.new('RGB', (1960, 825), 'white')
    for i, panel in enumerate(panels):canvas.paste(panel, (i*980,0))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--all', action='store_true');args=parser.parse_args()
    if args.all:
        deadline = time.monotonic()+3*3600
        while not all((LAB/'outputs/dynamic-human-color-continuity-004'/mode/'completion.json').exists() for mode in ['aliased','frozen']):
            if time.monotonic()>deadline:raise TimeoutError('Color study still unfinished; no extra large inference allocation')
            time.sleep(10)
    torch.set_num_threads(6)
    model_receipt = json.loads((OUT/'segformer-model-receipt.json').read_text())
    for member in model_receipt['members']:
        assert checksum(MODEL/member['file']) == member['sha256']
    protocol = dict(model=model_receipt['repository'], revision=model_receipt['revision'],
        implementation_sha256=checksum(Path(__file__)), crop_size=CROP, stride=STRIDE,
        input='native-resolution RGB, /255 then ImageNet mean/std; no image resizing',
        tile_merge='bilinear logits to native tile, arithmetic mean across overlapping tiles, per-pixel argmax',
        precision='FP16 autocast; FP32 logit interpolation and CPU accumulation',
        legacy_difference='HF SegFormer adapter, native-resolution tiles; no MMSeg MultiScaleFlipAug resize',
        role='Predicted training masks; not independent ground truth for evaluation',
        person_class=11, sky_class=10, official_human_group=[11,12,17,18], vehicle_group=[13,14,15])
    protocol_path = OUT/'semantic-protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text()) == protocol
    else:save(protocol_path, protocol)
    folders = ['semantic_labels', 'sky_masks', 'person_masks', 'fine_dynamic_masks/all',
               'fine_dynamic_masks/human', 'fine_dynamic_masks/vehicle']
    for folder in folders:(SCENE/folder).mkdir(parents=True, exist_ok=True)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    assert model.config.id2label[11] == 'person' and model.config.id2label[10] == 'sky'
    files = sorted((SCENE/'images').glob('*.jpg')) if args.all else [SCENE/'images/134_4.jpg']
    started = time.time(); rows=[]
    for index, path in enumerate(files):
        targets = [SCENE/folder/(path.stem+'.png') for folder in folders]
        image = Image.open(path).convert('RGB')
        if all(p.exists() for p in targets):
            labels = np.asarray(Image.open(targets[0]))
            assert labels.shape == (image.height, image.width) and labels.max() < 19
        else:
            labels = infer(model, image)
            masks = [labels, (labels==10)*255, (labels==11)*255,
                     np.isin(labels,[11,12,13,14,15,17,18])*255,
                     np.isin(labels,[11,12,17,18])*255, np.isin(labels,[13,14,15])*255]
            for target, mask in zip(targets, masks):
                Image.fromarray(mask.astype(np.uint8), mode='L').save(target)
        rows.append(dict(image=path.name, width=image.width, height=image.height,
            person_pixels=int((labels==11).sum()), sky_pixels=int((labels==10).sum()),
            labels_sha256=checksum(targets[0])))
        if path.stem in ['000_0','078_2','134_4','155_6']:
            preview(image, labels, OUT/f'assets/semantic-preview-{path.stem}.png')
        if index%10 == 0:
            print('SEMANTICS',index+1,'/',len(files), 'elapsed', round(time.time()-started), flush=True)
            save(OUT/'semantic-progress.json', dict(images_completed=index+1, images_total=len(files), last_image=path.name,
                role='full preprocessing' if args.all else 'single-image smoke test'))
    report = dict(status='full_scene_semantics_inferred' if args.all else 'single_image_semantics_inferred',
        protocol_sha256=checksum(protocol_path), images=len(files), mask_files=len(files)*len(folders),
        elapsed_seconds=time.time()-started, peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        quality='predictions await visual review; no ground-truth segmentation score claimed', per_image=rows)
    save(OUT/('semantic-completion.json' if args.all else 'semantic-smoke.json'), report)
    print('SEMANTICS COMPLETE', len(files), report['peak_cuda_allocated_gib'], flush=True)


if __name__ == '__main__':
    with torch.inference_mode():main()
