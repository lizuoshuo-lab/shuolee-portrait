from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent/'static-formula.png'
plt.rcParams.update({'mathtext.fontset':'stix','font.family':'STIXGeneral'})
fig=plt.figure(figsize=(16,1.1),dpi=180)
fig.text(.01,.4,r'$\mathcal{L}=\mathcal{L}_{\mathrm{RGB}}+0.1\mathcal{L}_{\alpha}+100\mathcal{L}_{s}+0.015\mathcal{L}_{N}+0.03\mathcal{L}_{G}+0.15\mathcal{L}_{P}$',fontsize=29,color='#183543')
fig.savefig(out,bbox_inches='tight',pad_inches=.05,facecolor='white');plt.close(fig)
print(out)
