"""Freeze the source-selection experiment as a separately versioned script."""
from pathlib import Path
import hashlib,json
LAB=Path(__file__).resolve().parents[2];SRC=LAB/'scripts/dynamic_human/diagnose_4k4d_source_support.py'
dest=SRC.with_name('diagnose_4k4d_source_selection.py');assert not dest.exists()
text=SRC.read_text(encoding='utf8').replace('dynamic-human-4k4d-source-support-042','dynamic-human-4k4d-source-selection-043')
text=text.replace("variants=['original','source_foreground_gate','source_foreground_gate_unpremultiply']", "variants=['original','source_foreground_gate','nearest2_supported','temperature4_supported','hardmax_supported']")
text=text.replace("intervention='Reweight original learned blending weights by source-mask support>=0.5; second variant also unpremultiplies sampled sourceRGB by sampled mask. Original SH residual retained. Unsupported points fall back to original color.'", "intervention='Fixed source-support threshold0.5; compare all8sources against nearest2supported, normalized weight power4, and per-point hardmax. Same source colors and SH residual. Nearest2 without support falls back to all8supported; points with no support retain original color.'")
a=text.index('    restored=colors/support.clamp_min(.5)');b=text.index('    full=self.render_pytorch',a)
text=text[:a]+'''    def renormalize(w,fallback):
        d=w.sum(-3,keepdim=True)
        return torch.where(d>1e-7,w/d.clamp_min(1e-7),fallback)
    near=norm.clone();near[:,2:]=0
    near=renormalize(near,norm)
    sharp=renormalize(norm.pow(4),norm)
    hard=torch.zeros_like(norm).scatter(1,norm.argmax(1,keepdim=True),1.)
    hard=torch.where(usable,hard,weights)
    alternatives={'source_foreground_gate':norm,'nearest2_supported':near,'temperature4_supported':sharp,'hardmax_supported':hard}
    variants=[(name,((w*colors).sum(-3)+residual).clamp(0,1)) for name,w in alternatives.items()]
    assert all(torch.allclose(w.sum(-3),torch.ones_like(w.sum(-3)),atol=2e-6) for w in alternatives.values())
''' +text[b:]
a=text.index('    x0,y0,x1,y1=r[\'bbox\']');b=text.index('    rows.append(',a)
text=text[:a]+'''    x0,y0,x1,y1=r['bbox'];w=x1-x0;h=y1-y0
    for variant in ['source_foreground_gate','nearest2_supported','temperature4_supported','hardmax_supported']:
        panel=Image.new('RGB',(3*w,h+32),'white');d=ImageDraw.Draw(panel)
        for i,(label,array) in enumerate([('Original photograph',z['source']),('Original rendering',u8(pred['original'])),(variant,u8(pred[variant]))]):
            panel.paste(Image.fromarray(array[y0:y1,x0:x1]),(i*w,32));d.text((i*w+4,10),label,fill='black')
        panel.save(OUT/f'{prefix}-{variant}-comparison.png')
''' +text[b:]
text=text.replace('six_readonly_source_support_controls_complete_audit_and_review_pending','six_readonly_source_selection_controls_complete_audit_and_review_pending')
dest.write_text(text,encoding='utf8')
out=LAB/'outputs/dynamic-human-4k4d-source-selection-043';out.mkdir(exist_ok=True)
(out/'script-derivation.json').write_text(json.dumps(dict(source=str(SRC),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),derived=str(dest),derived_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),scope='Separate frozen-model color-weight experiment; existing042script/results unchanged.'),indent=2)+'\n')
print('043 fixed choices prepared')
