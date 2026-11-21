from pathlib import Path
H=Path(__file__).parent
s=(H/'update_deck_smplify_corrected123_v2.mjs').read_text(encoding='utf-8')
replacements={
 "tag='smplify-corrected123-v2'":"tag='smplify-conditioned123'",
 "b+'/smplify-ref123-deck-receipt.json'":"b+'/smplify-corrected123-v2-deck-receipt.json'",
 "smplifyx-reference-control/corrected-reference":"smplifyx-reference-control/conditioned-reference",
 "replace('参考适配暴露置信度传递错误'":"replace('SMPLify-X修复后的三视角核验'",
 "replace('原始检测分数被错误地全部截成1'":"replace('侧面改善，另两视角的求解试探步失稳'",
 "replace('三视角试跑无效。分数接口已修正'":"replace('侧面身体中位20.26→5.69px'",
 "replace('三组依次为视角0、31、62。每组左为原图'":"replace('三组依次为视角0、31、62；每组左原图'",
 "02_有限视角重建优化路线_123修复后实拍核验v2.pptx":"02_有限视角重建优化路线_123数值修复与观测冲突.pptx",
 "preserve_deck_smplify_corrected123_v2.py":"preserve_deck_smplify_conditioned123.py"}
for a,b in replacements.items():
    assert a in s,a
    s=s.replace(a,b)
(H/'update_deck_smplify_conditioned123.mjs').write_text(s,encoding='utf-8')
