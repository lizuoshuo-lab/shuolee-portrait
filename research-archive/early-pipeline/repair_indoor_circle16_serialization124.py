"""One deterministic FP32 inverse-exp encoding check; no image-guided search."""
import inspect
import diagnose_indoor_circle16_export124 as prior

def main():
    code=inspect.getsource(prior.main)
    code=code[:code.index('        delta=(native-portable).abs()')]
    code=code.replace("out=ROOT/'export-failure-diagnostic'","out=ROOT/'export-scale-encoding124'")
    code+='''        from plyfile import PlyData
        target=model.get_scaling.detach()
        original=plain._scaling.detach()
        # Invert exp in its actual float32 implementation. Neighbouring log
        # encodings are chosen only by parameter error, never rendered pixels.
        neg=torch.full_like(original,float('-inf'))
        pos=torch.full_like(original,float('inf'))
        encodings=torch.stack((original,torch.nextafter(original,neg),torch.nextafter(original,pos)))
        errors=(encodings.exp()-target[None]).abs()
        choice=errors.argmin(0)
        encoded=encodings.gather(0,choice[None])[0]
        assert torch.all((encoded.exp()-target).abs()<=(original.exp()-target).abs())
        ply=PlyData.read(snap/'world.ply')
        for j in range(3):ply['vertex'].data['scale_'+str(j)]=encoded[:,j].cpu().numpy()
        candidate=out/'world-nearest-log-f32.ply'
        ply.write(candidate)
        reread=official.GaussianModel(0);reread.load_ply(str(candidate))
        assert torch.equal(reread._scaling,encoded)
        delta=(native-render(cam,reread,pipe,bg)['render']).abs()
        old_delta=(native-portable).abs()
        parity=dict(mean_abs=float(delta.mean()),max_abs=float(delta.max()),
            passed=bool(delta.mean()<2e-6 and delta.max()<.005))
        save(out/'result.json',dict(updated=now(),source_sha256=sha(snap/'world.ply'),candidate_sha256=sha(candidate),
            source_checkpoint_sha256=sha(snap/'checkpoint.pt'),candidate=str(candidate),parity=parity,
            prior=dict(mean_abs=float(old_delta.mean()),max_abs=float(old_delta.max())),
            mechanism='Per-scalar float32 nearest inverse-exp encoding, one adjacent ULP each way; minimize scale error only; no image selection.',
            changed_log_components=int((encoded!=original).sum()),
            source_scale_max_error=float((original.exp()-target).abs().max()),
            candidate_scale_max_error=float((encoded.exp()-target).abs().max()),
            training_resumed=False,optimizer_updates_added=0,thresholds_unchanged=True))
        for path,digest in hashes.items():assert sha(path)==digest
        print('FP32_SCALE_ENCODING_RESULT',parity,flush=True)
'''
    ns=dict(prior.main.__globals__)
    exec(compile(code,__file__+':read-only-encoding-check','exec'),ns)
    ns['main']()

if __name__=='__main__':main()
