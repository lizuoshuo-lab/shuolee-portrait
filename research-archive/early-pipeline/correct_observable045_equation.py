from pathlib import Path
B=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
pub=B/'publish_observable045.py'
page=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer/body-smplx.html')
formula='''<math display="block"><mrow><msub><mi>e</mi><mrow><mi>t</mi><mi>j</mi><mi>a</mi></mrow></msub><mo>=</mo><msub><mrow><mo>[</mo><msub><mi>π</mi><mi>t</mi></msub><mo>(</mo><msub><mi>X</mi><mrow><mi>t</mi><mi>j</mi></mrow></msub><mo>)</mo><mo>]</mo></mrow><mi>a</mi></msub><mo>−</mo><msub><mrow><mo>[</mo><msub><mi>u</mi><mrow><mi>t</mi><mi>j</mi></mrow></msub><mo>]</mo></mrow><mi>a</mi></msub></mrow></math>
<math display="block"><mrow><msub><mi>ℒ</mi><mtext>image</mtext></msub><mo>=</mo><mfrac><mrow><munder><mo>∑</mo><mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></munder><msub><mi>v</mi><mrow><mi>t</mi><mi>j</mi></mrow></msub><msub><mi>w</mi><mrow><mi>t</mi><mi>j</mi></mrow></msub><munder><mo>∑</mo><mi>a</mi></munder><mi>ρ</mi><mo>(</mo><mfrac><msub><mi>e</mi><mrow><mi>t</mi><mi>j</mi><mi>a</mi></mrow></msub><mi>σ</mi></mfrac><mo>)</mo></mrow><mrow><munder><mo>∑</mo><mrow><mi>t</mi><mo>,</mo><mi>j</mi></mrow></munder><msub><mi>w</mi><mrow><mi>t</mi><mi>j</mi></mrow></msub></mrow></mfrac></mrow></math>
'''
for p in [pub,page]:
    t=p.read_text(encoding='utf-8');start=t.index('<math display="block">',t.index('<h5>可见性筛选怎样进入目标函数</h5>'))
    end=t.index('<p><var>t</var>为帧',start)
    t=t[:start]+formula+t[end:]
    t=t.replace('<var>t</var>为帧，<var>j</var>为对应点；','<var>t</var>为帧，<var>j</var>为对应点，<var>a</var>分别取横、纵两个像素坐标；')
    t=t.replace('身体项尺度为20像素，稳健函数与前述相同。','身体项尺度为20像素，对横纵残差分别计算稳健函数后求和，稳健函数与前述相同。')
    p.write_text(t,encoding='utf-8')
print('Equation now matches coordinatewise robust loss in observable_landmark_constraint.py.')
