from pathlib import Path
build=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
source=(build/'close_template038.ps1').read_text(encoding='utf-8-sig')
updates=[
 ('smplx-nlf-template-fresh-038','smplx-nlf-visible-projection-fresh-039'),
 ('template038','visible039'),('template_family_effect_verified_alignment_not_accepted','visible_projection_no_clear_geometry_gain'),
 ("@('neutral','male')","@('surface3d','visible2d')"),('Reviewed rejected038 model','Reviewed rejected039 model'),
 ('nlf037-deck-receipt.json','template038-deck-receipt.json'),
 ('038两组模型','039两组模型'),('smplx_nlf_template_038','smplx_nlf_visible_projection_039')]
for a,b in updates:
    assert a in source,a
    source=source.replace(a,b)
source += '''
$publicPath = Join-Path ([IO.Path]::GetFullPath((Join-Path $researchRoot '..\\viewer\\body-fitting-method\\nlf-prior'))) 'visible-conclusion.json'
$public=Get-Content -LiteralPath $publicPath -Raw | ConvertFrom-Json
$public | Add-Member -NotePropertyName cleanup -NotePropertyValue $conclusion.cleanup -Force
$public | Add-Member -NotePropertyName model_files_retained -NotePropertyValue $false -Force
$public | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $publicPath -Encoding utf8
'''
(build/'close_visible039.ps1').write_text(source,encoding='utf-8-sig')
print('Prepared bounded exact-file cleanup, no recursive removal.')
