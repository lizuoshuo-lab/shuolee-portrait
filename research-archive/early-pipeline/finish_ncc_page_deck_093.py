from pathlib import Path
import json,hashlib,shutil
from html.parser import HTMLParser
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V=B.parent/'viewer';W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

class Balance(HTMLParser):
    def __init__(self):super().__init__();self.details=0;self.minimum=0
    def handle_starttag(self,t,a):
        if t=='details':self.details+=1
    def handle_endtag(self,t):
        if t=='details':self.details-=1;self.minimum=min(self.minimum,self.details)

def main():
    page=V/'body-smplx.html';s=page.read_text(encoding='utf8')
    for suffix in ['control','off']:
        old=f'href="?model=whole-body-static-ncc-{suffix}"';assert s.count(old)==1
        s=s.replace(old,f'href="./?model=whole-body-static-ncc-{suffix}"')
    marker='<!-- STATIC_SURFACE_085_END -->';i=s.index(marker)+len(marker);j=s.rindex('</main>')
    p=Balance();p.feed(s[:i]);assert p.details==0
    h=Balance();h.feed(s[i:j]);assert h.details==0 and h.minimum==0
    s=s[:i]+'\n<details id="pre093-history"><summary>此前人体拟合与重建记录（历史）</summary><p class="notice">以下保留各轮当时的结论与证据。旧段落中的“当前”和“下一步”只指该轮，最新状态以上方093对照为准。</p>'+s[i:j]+'</details>\n'+s[j:]
    page.write_text(s,encoding='utf8')
    # The exported slide52 PNG has been visually inspected and first51 renders verified unchanged.
    path=W/'human093-deck-receipt.json';r=read(path)
    src=Path(r['source']);final=Path(r['finalPath'])
    assert not r['canonical_replaced'] and sha(src)==r['source_sha256'] and sha(final)==r['final_sha256']
    assert r['changed_slides']==[52] and all(not t['changed'] for t in r['renders'][:51])
    shutil.copy2(final,src);assert sha(src)==r['final_sha256']
    r.update(canonical_replaced=True,final_slide_visual_review=True,first51_visual_render_unchanged=True)
    path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(ppt=str(src),sha256=r['final_sha256'],history_collapsed=True)))

if __name__=='__main__':main()
