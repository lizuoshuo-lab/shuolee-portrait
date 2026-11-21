from pathlib import Path
H=Path(__file__).parent
s=(H/'update_deck_smplify_conditioned123.mjs').read_text(encoding='utf-8')
replacements={
    "tag='smplify-conditioned123'":"tag='smplify-observation123'",
    "b+'/smplify-corrected123-v2-deck-receipt.json'":"b+'/smplify-conditioned123-deck-receipt.json'",
    'smplifyx-reference-control/conditioned-reference':'smplifyx-reference-control/observation-contract',
    "replace('SMPLify-X修复后的三视角核验'":"replace('SMPLify-X数值修复与观测冲突'",
    "replace('侧面改善，另两视角的求解试探步失稳'":"replace('三视角完成拟合；重复手腕目标仍不一致'",
    "replace('侧面身体中位20.26→5.69px'":"replace('面部中位1.53/0.97/0.94px'",
    "replace('三组依次为视角0、31、62；每组左原图'":"replace('三组左原图、右网格；左右腕重复检测差距中位'",
    '02_有限视角重建优化路线_123数值修复与观测冲突.pptx':'02_有限视角重建优化路线_123跨视角一致性.pptx',
    'preserve_deck_smplify_conditioned123.py':'preserve_deck_observation123.py'
}
for a,b in replacements.items():
    assert a in s,a
    s=s.replace(a,b)
(H/'update_deck_observation123.mjs').write_text(s,encoding='utf-8')
