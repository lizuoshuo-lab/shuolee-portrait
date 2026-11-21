from pathlib import Path
H=Path(__file__).parent
s=(H/'update_deck_field_material123.mjs').read_text(encoding='utf-8')
s=s.replace("tag='field-material123'","tag='direct-smplx123'")
s=s.replace('local-clothed-surface-123/field-preflight/native-material/topology-review','local-clothed-surface-123/direct-smplx-bindings')
s=s.replace("root+'/actual-review.json'","root+'/visual-review.json'")
s=s.replace("root+'/native-review/'+review.slide.figure","root+'/'+review.slide.figure")
s=s.replace('02_有限视角重建优化路线_123田地原生材料.pptx','02_有限视角重建优化路线_123二维三维拟合.pptx')
s=s.replace('preserve_deck_field_material123.py','preserve_deck_direct_smplx123.py')
p=H/'update_deck_direct_smplx123.mjs';assert not p.exists();p.write_text(s,encoding='utf-8')
print(p)
