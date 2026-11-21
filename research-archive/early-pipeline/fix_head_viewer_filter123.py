import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now
ident='local-surface-123-demo-head-hat-s18000'
for folder in [a.V,a.S]:
    for path in [folder/f'model-{ident}.json',folder/'models'/ident/'model-context.json']:
        d=a.read(path);assert d['sha256']=='a5f56a4027cf937402b7bc538808c5ebed71fb4f579165ab2662220351c26cec'
        assert d['filter']==dict(preBlurAmount=0,blurAmount=.3)
        d['filter']=dict(preBlurAmount=.3,blurAmount=0)
        d['previewFilterNote']='Same noncompensated0.3 pixel variance as native GaussianAvatars lowpass and received scene-package import; viewport resolution still differs from172px native.'
        a.save(path,d)
    p=folder/'latest-result.json';d=a.read(p);assert d['modelId']==ident
    d.update(updated=now(),revision=d['sha256']+'-native-filter1');a.save(p,d)
print('ONLY_CURRENT_HEAD_PREVIEW_FILTER_FIXED')