import compare_transfer_077 as m
m.A=m.B/'person-static-observed-correspondence-fresh-077/review/correspondence'
m.R=m.B/'person-static-surface-consistency-fresh-079'
if __name__=='__main__':
    m.main()
    p=m.R/'review/paired-transfer-comparison.json';r=m.read(p)
    r['units']='Full rectified image pixels, same175 camera manifest in077/079; not1536trainingpixels'
    r.update(reference_round='077',candidate_round='079')
    p.write_text(m.json.dumps(r,indent=2)+'\n',encoding='utf8')
