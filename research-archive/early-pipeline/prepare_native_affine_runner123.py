from pathlib import Path
H=Path(__file__).parent
s=(H/'run_native_photometric123.py').read_text(encoding='utf-8')
s=s.replace('from roma116_coordinates import sample_direction','from roma116_coordinates import sample_direction\nfrom native_affine_patch123 import AffinePhotometry')
s=s.replace("O=D/'native-photometric-contrast'","O=D/'native-affine-contrast'")
s=s.replace("plan=read(O/'plan.json');p=", "assert read(O/'control-result.json')['passed']\n    frozen=read(D/'native-photometric-contrast/plan.json')\n    for path,h in frozen['source_sha256'].items():assert sha(local(path))==h,path\n    plan=read(O/'plan.json');p=")
s=s.replace('m=TranslationPhotometry(', 'm=AffinePhotometry(')
old="good=bool(opt.success and np.isfinite(opt.x).all());bound=bool(np.any(np.abs(opt.x[:2])>=2.999) or opt.x[2]<=.25001 or opt.x[2]>=3.99999 or abs(opt.x[3])>=.49999)"
new="affine_matrix=np.eye(2)+opt.x[2:6].reshape(2,2);det=float(np.linalg.det(affine_matrix));good=bool(opt.success and np.isfinite(opt.x).all() and det>1e-4);bound=bool(np.any(np.abs(opt.x[:2])>=2.999) or np.any(np.abs(opt.x[2:6])>=.24999) or opt.x[6]<=.25001 or opt.x[6]>=3.99999 or abs(opt.x[7])>=.49999)"
assert old in s;s=s.replace(old,new)
s=s.replace('m.fun([0,0,1,0])','m.fun([0,0,0,0,0,0,1,0])')
s=s.replace('translation_norm=float(np.linalg.norm(opt.x[:2])))','translation_norm=float(np.linalg.norm(opt.x[:2])),affine_determinant=det)')
s=s.replace("result=dict(updated=now()", "translation=read(D/'native-photometric-contrast/result.json');trans=np.array([x['refined']['eligible'] for x in translation['rows']])\n        result=dict(translation_only_pass=int(trans.sum()),gained_vs_translation=int((~trans&new).sum()),lost_vs_translation=int((trans&~new).sum()),updated=now()")
s=s.replace('Native local translation plus affine brightness only.', 'Native local affine geometry plus affine brightness. Same frozen query set is method-development evidence only.')
s=s.replace('One bounded real24-query photometric correspondence contrast, no camera optimization.','One controlled local-affine24-query contrast; no camera or training change.')
out=H/'run_native_affine123.py';assert not out.exists();out.write_text(s,encoding='utf-8');print(str(out))
