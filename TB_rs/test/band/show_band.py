import numpy as np
import os
import matplotlib.pyplot as plt
band_data=np.loadtxt('BAND.dat')
fig, ax = plt.subplots()
knode_file=os.path.join('.','BAND_klabel.dat')
with open(knode_file,'r',encoding='utf-8') as f:
    lines=f.readlines()
knodes=[]
for i in range(len(lines)):
    knodes.append(str.split(lines[i]))
    knodes[i][1]=float(knodes[i][1])
    ax.axvline(x=knodes[i][1],linewidth=0.5,color='k')
ax.axhline(y=0,linewidth=0.5,color='g',ls='--')
knodes=list(map(list, zip(*knodes)))
ax.set_xticks(knodes[1])
ax.set_xlim(knodes[1][0],knodes[1][-1])
for i,a in enumerate(knodes[0]):
  if a=='GAMMA':
    knodes[0][i]='$\Gamma$'
ax.set_xticklabels(knodes[0])
ax.set_ylabel(r'E-E$_f$ (eV)')
ax.set_xlabel(r'kpoints')
ax.set_ylim(-2,2)
ax.plot(band_data[:,0],band_data[:,1:],color='k',linewidth=0.5)
fig.savefig('bandstructure.pdf')
