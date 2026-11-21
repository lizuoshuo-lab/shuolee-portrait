from pathlib import Path
folder=Path(__file__).resolve().parent
text=(folder/'audit_4k4d_source_support.py').read_text(encoding='utf8').replace('dynamic-human-4k4d-source-support-042','dynamic-human-4k4d-source-selection-043')
a=text.index("    x0,y0,x1,y1=row['bbox']");b=text.index('    checks.append(',a)
text=text[:a]+'''    x0,y0,x1,y1=row['bbox'];w=x1-x0
    for key in row['metrics']:
        if key=='original':continue
        panel=np.array(Image.open(OUT/f'{prefix}-{key}-comparison.png'))
        for i,array in enumerate([ref['source'],u8(pred['original']),u8(pred[key])]):assert np.array_equal(panel[32:,i*w:(i+1)*w],array[y0:y1,x0:x1])
''' +text[b:]
text=text.replace('native_pngs=3,full_person_panel=1','native_pngs=5,full_person_panel=4').replace('18nativePNGs_6panels','30nativePNGs_24panels').replace('042 passed:18nativePNGs,6panels','043 passed:30nativePNGs,24panels')
target=folder/'audit_4k4d_source_selection.py';assert not target.exists();target.write_text(text,encoding='utf8')
