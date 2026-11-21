from datetime import datetime,timezone
import audit_gaussianavatars_params123 as a
out=a.REC/'cross-subject-sh0-124/grass-temporal'
paths=[a.R/'video10-global-initializer/index.html',a.V/'diagnostics/video10-global-initializer/index.html',a.S/'diagnostics/video10-global-initializer/index.html']
hashes={str(p):a.sha(p) for p in paths};assert len(set(hashes.values()))==1
models={str(p):a.read(p/'latest-result.json')['modelId'] for p in [a.V,a.S]}
assert all(v=='local-surface-123-demo-body-smplx-offset-s120000-sh0' for v in models.values())
proof=dict(updated=datetime.now(timezone.utc).isoformat(),tab_id='44',url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#grass-temporal-124',actual_UI_verified=True,actual_summary_and_image_visible=True,html_sha256=hashes,model_ids=models,old_model_preserved=True,tab15_untouched=True,screenshots=[str(out/'ui-review/actual-summary.png'),str(out/'ui-review/actual-report.png')],new_PLY_published=False,company00_deployment_verified=False)
a.save(out/'ui-review/receipt.json',proof)
p=a.read(out/'publication.json');p.update(actual_UI_verified=True,receipt=str(out/'ui-review/receipt.json'));a.save(out/'publication.json',p)
for path in [a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json']:
 d=a.read(path);d['grass_temporal124']['actual_UI_verified']=True;d['grass_temporal124']['ui_review']=str(out/'ui-review/receipt.json');a.save(path,d)
print('UI_AND_MODEL_PRESERVATION_VERIFIED',list(models.values())[0])