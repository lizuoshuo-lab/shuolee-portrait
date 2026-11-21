"""Typeset the exact training objective for the existing technical report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'DejaVu Serif'})
out=Path('/mnt/c/Users/Administrator/Documents/New project/shuolee-slide-update/mask-formula.png')
fig=plt.figure(figsize=(7.0,4.0),dpi=230,facecolor='white')
lines=[
    (r'$\Omega_F=\{p:A^*(p)>0.99\}$',.88),
    (r'$\Omega_S=\{p:\mathcal{N}_{11\times11}(p)\subseteq\Omega_F\}$',.69),
    (r'$\mathcal{L}_1=\frac{1}{3|\Omega_F|}\sum_{p\in\Omega_F}\|\widehat{C}(p)-C^*(p)\|_1$',.49),
    (r'$\mathcal{L}_S=\frac{1}{|\Omega_S|}\sum_{p\in\Omega_S}(1-\mathrm{SSIM}_p)$',.27),
    (r'$\mathcal{L}_{\mathrm{RGB}}=0.8\mathcal{L}_1+0.2\mathcal{L}_S$',.08)]
for txt,y in lines:fig.text(.025,y,txt,color='#173345',fontsize=23,va='center')
fig.savefig(out,dpi=230,bbox_inches=None,pad_inches=0);plt.close(fig)
print(out)
