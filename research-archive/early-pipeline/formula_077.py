from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset']='stix'
p=Path(__file__).parent
f=plt.figure(figsize=(16,.88),dpi=220);f.patch.set_alpha(0)
f.text(.006,.45,r'$\mathcal{L}=\mathcal{L}_{\mathrm{RGB}}+0.1\mathcal{L}_{\alpha}+100\mathcal{L}_{s}+0.015\mathcal{L}_{N}+0.03\mathcal{L}_{G}+0.15\mathcal{L}_{P}+0.002\,w_k\mathcal{L}_{C}$',
       va='center',ha='left',fontsize=39,color='#183444')
f.savefig(p/'077-objective.png',transparent=True);plt.close(f)
