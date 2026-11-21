"""Use only bit-identical bounded reference caches; reject sequential backward."""
import hashlib
import json
from pathlib import Path
import shutil
import time
ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-mip-001')


def main():
    source=RUNTIME/'repo-depth-memory';target=RUNTIME/'repo-depth-cache';assert not target.exists()
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    train=target/'msv2/train_depth_guided.py';text=train.read_text()
    start=text.index('        # The parameters are unchanged between these backward calls.')
    end=text.index('        if iteration % 100 == 0:',start)
    text=text[:start]+'''        loss = loss + depth_guidance(viewpoint_cam,gaussians,iteration,valid,gaussians._culling[:,viewpoint_cam.uid])
        loss.backward()
'''+text[end:]
    original=(RUNTIME/'repo-depthguide/msv2/train_depth_guided.py').read_text()
    block=original[original.index('        loss = (1.0 - opt.lambda_dssim)'):original.index('        iter_end.record()')].strip()
    assert block in text
    point='                gaussians.optimizer.zero_grad(set_to_none = True)'
    assert point in text
    text=text.replace(point,point+'''
                if iteration % 100 == 0 and torch.cuda.memory_reserved()-torch.cuda.memory_allocated()>256*2**20:
                    torch.cuda.empty_cache()
''')
    train.write_text(text)
    out=ROOT/'native-resolution-cache-runtime';out.mkdir(exist_ok=False)
    state=dict(status='prepared_pending_cache_validation',created=time.time(),repository=str(target),
        original_repository=str(RUNTIME/'repo-depthguide'),loss_and_backward_block_byte_identical=True,
        changes=['GPU reference cache bounded to4camera-resolution pairs; raw host tensors cached on CPU',
            'Release unused allocator blocks every100steps if unused reservation exceeds256MiB',
            'Memory telemetry only'],
        sequential_backward_rejected=True,rejection='Actual-scene gradient check showed1.50%relativeL2scale and0.284%rotation discrepancy; original joint backward retained.',
        rejection_evidence=str(ROOT/'native-resolution-memory-runtime/validation-attempt002.json'),
        files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [train,target/'native_depth_guidance.py']})
    (out/'manifest.json').write_text(json.dumps(state,indent=2));print(json.dumps(state))


if __name__=='__main__':main()
