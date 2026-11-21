from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral'})
fig=plt.figure(figsize=(10,5),dpi=180)
lines=[r'$\widehat{I}_b=\widehat{F}+(1-\widehat{A})b$',
       r'$I_b^*=F^*+(1-A^*)b,\quad b\in\{0,1\}$',
       r'$\mathcal{L}_{\rm pair}=\frac{1}{2}\sum_{b\in\{0,1\}}\ell(\widehat{I}_b,I_b^*)$',
       r'$\mathcal{L}_\alpha=\frac{1}{2}\langle|\widehat{A}-A^*|\rangle_\Omega+\frac{1}{2}\langle|\widehat{A}-A^*|\rangle_\Gamma$',
       r'$F^*=\mathcal{R}(A_0I_0),\quad A^*=\mathcal{R}(A_0)$']
for i,s in enumerate(lines):fig.text(.03,.87-i*.18,s,fontsize=23 if i==3 else 27,color='#183543')
out=Path(__file__).parent/'alpha-formula.png';fig.savefig(out,bbox_inches='tight',pad_inches=.15,facecolor='white');plt.close(fig)
