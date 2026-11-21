"""Read-only CPU input-cache replay of one completed Gaussian run.

No fitting, optimization, CUDA call, source modification or model output.
"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/'reviews/completed-cycle-io-audit.json'
    if output.exists():
        raise RuntimeError('The one-shot replay already has a result; inspect it instead of rerunning.')
    cfg = json.loads((root/'gaussian-run/config.json').read_text())
    source = root/'data/training-inputs.json'
    rows = json.loads(source.read_text())['records']
    assert len(rows) == 94 and cfg['image_cache_frames'] == 8
    assert cfg['training_height'] == 899 and cfg['seed'] == 6060
    size = (round(rows[0]['wh'][0]*899/rows[0]['wh'][1]), 899)
    assert all(r['wh'] == rows[0]['wh'] for r in rows)
    frame_bytes = size[0]*size[1]*4  # cached RGB uint8 + alpha uint8
    assert frame_bytes*len(rows) < 1024**3
    rng = np.random.default_rng(cfg['seed'])
    order = []
    while len(order) < cfg['iterations']:
        order.extend(rng.permutation(len(rows)).tolist()[::-1])
    order = order[:cfg['iterations']]
    def replay_counts(capacity):
        cache = OrderedDict([(0, None)])  # training preflight reads view0
        misses = 1
        for index in order:
            if index not in cache:
                misses += 1
                cache[index] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
            cache.move_to_end(index)
        return misses
    start = time.monotonic()
    for row in rows:
        assert sha(row['image']) == row['source_sha256']
        assert sha(row['mask']) == row['mask_sha256']
    arms = []
    observed = {}
    for capacity in (8, len(rows)):
        cache = OrderedDict()
        times, hits, digests = [], [], []
        for index in [0]+order[:188]:
            if time.monotonic()-start > 120:
                raise RuntimeError('Read-only audit CPU wall budget exceeded; do not expand automatically.')
            t0 = time.monotonic()
            hit = index in cache
            if not hit:
                row = rows[index]
                with Image.open(row['image']) as im:
                    rgb = torch.from_numpy(np.array(im.convert('RGB').resize(size, Image.Resampling.LANCZOS))).permute(2, 0, 1)
                with Image.open(row['mask']) as im:
                    alpha = torch.from_numpy(np.array(im.convert('L').resize(size, Image.Resampling.BILINEAR)))[None]
                cache[index] = rgb, alpha
                if len(cache) > capacity:
                    cache.popitem(last=False)
            cache.move_to_end(index)
            rgb, alpha = cache[index]
            times.append(time.monotonic()-t0)
            hits.append(hit)
            digest = hashlib.sha256(rgb.numpy().tobytes()+alpha.numpy().tobytes()).hexdigest()
            if index in observed:
                assert observed[index] == digest, 'Input pixels changed between cache capacities.'
            observed[index] = digest
            digests.append(digest)
        arms.append(dict(capacity=capacity, requests=len(times), cache_hits=sum(hits),
                         decoded_pairs=len(times)-sum(hits), input_loading_seconds=sum(times),
                         first_epoch_seconds=sum(times[1:95]), second_epoch_seconds=sum(times[95:]),
                         median_load_seconds=statistics.median(times),
                         stored_uint8_bytes=sum(x.numel()+y.numel() for x,y in cache.values()),
                         request_pixel_digest=hashlib.sha256(''.join(digests).encode()).hexdigest()))
        del cache, rgb, alpha
        gc.collect()
    assert arms[0]['request_pixel_digest'] == arms[1]['request_pixel_digest']
    fit = json.loads((root/'fit/official-fit-result.json').read_text())
    durations = [r['elapsed_seconds'] for r in fit['frame_results']]
    log = [json.loads(s) for s in (root/'gaussian-run/loss.jsonl').read_text().splitlines()]
    parse = datetime.fromisoformat
    utc_elapsed = (parse(log[-1]['updated'])-parse(log[0]['updated'])).total_seconds()
    mono_elapsed = log[-1]['seconds']-log[0]['seconds']
    result = dict(updated=datetime.now(timezone.utc).isoformat(), state='complete',
        mode='read_only_CPU_input_replay', cuda_used=False, source_sha256=sha(source),
        source_training_sha256=cfg['official_fit_provenance'].get('training_source_sha256') if isinstance(cfg.get('official_fit_provenance'),dict) else None,
        native_wh=rows[0]['wh'], training_wh=size, frame_bytes=frame_bytes,
        all94_uint8_cache_bytes=frame_bytes*len(rows), replay=arms,
        pixels_identical=True, quality_or_training_speedup_measured=False,
        full18000_order_simulation=dict(includes_initial_preflight=True,
            cache8_decoded_pairs=replay_counts(8), cache94_decoded_pairs=replay_counts(94),
            scope='Exact seeded access order simulation; not timed full training.'),
        fit_timing=dict(total_monotonic_seconds=fit['elapsed_seconds'],
            summed_frame_seconds=sum(durations), outer_overhead_seconds=fit['elapsed_seconds']-sum(durations),
            median_frame_seconds=statistics.median(durations), min_frame_seconds=min(durations),max_frame_seconds=max(durations)),
        gaussian_clock_check=dict(log_first_to_last_utc_seconds=utc_elapsed,
            log_first_to_last_monotonic_seconds=mono_elapsed,difference_seconds=utc_elapsed-mono_elapsed,
            explanation='Clock streams disagree. Do not combine wall-clock stage timings with monotonic timings or claim calibrated end-to-end speed.'),
        limitations=['CPU decode/resize/cache replay only; no optimizer or GPU render benchmark.',
            'Filesystem cache/order may affect loading times; no extrapolated training speedup.',
            'No change to model, K, masks, observations, optimizer, losses, fitting, or current delivery.'])
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
