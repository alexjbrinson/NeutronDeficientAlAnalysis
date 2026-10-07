"""
Figure 9 — Mirror charge radii differences vs isospin asymmetry
Reads data from fig9_data.csv

Usage: python3 fig9_mirror_shift.py
"""
import csv
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# -------- Load data ----------------------------------------------------------
data = {}
with open('fig9_data.csv') as f:
    reader = csv.DictReader(filter(lambda r: not r.lstrip().startswith('#'), f))
    for row in reader:
        data.setdefault(row['dataset'], []).append({
            'label': row['label'],
            'I':     float(row['I']),
            'value': float(row['value']),
            'sigma': float(row['sigma']),
        })

def cols(name, key):
    return [d[key] for d in data[name]]

# -------- Plot ---------------------------------------------------------------
c_exp = '#000000'

s, f = 1.5, 1.1
fig, ax1 = plt.subplots(figsize=(13, 8))

# CC/AFDMC band
I_lin = np.linspace(0, 0.22, 200)
ax1.fill_between(I_lin, 1.574*I_lin - 0.021, 1.574*I_lin + 0.021,
                 alpha=0.10, color='#43A047', zorder=0)

# Literature
ax1.errorbar(cols('literature', 'I'), cols('literature', 'value'),
             yerr=cols('literature', 'sigma'),
             fmt='s', color='#78909C',
             markersize=7*s, capsize=0, markeredgecolor='k', markeredgewidth=0.4*s,
             markerfacecolor='#78909C', elinewidth=0.8*s, zorder=3, alpha=0.6)

# This work
I_e = cols('mirror_exp', 'I')
v_e = cols('mirror_exp', 'value')
s_e = cols('mirror_exp', 'sigma')
ax1.errorbar(I_e, v_e, yerr=s_e, fmt='s:', color=c_exp,
             markersize=10*s, capsize=0, markeredgecolor='k', markeredgewidth=0.6*s,
             markerfacecolor=c_exp, lw=1.5*s, elinewidth=1.2*s, zorder=6)

# Predicted 22F
I_p = data['mirror_pred'][0]['I']
v_p = data['mirror_pred'][0]['value']
s_p = data['mirror_pred'][0]['sigma']
ax1.errorbar([I_e[-1], I_p], [v_e[-1], v_p], fmt=':', color=c_exp,
             lw=1.5*s, zorder=5)
ax1.errorbar(I_p, v_p, yerr=s_p, fmt='s', color=c_exp,
             markersize=10*s, capsize=0, markeredgecolor='k', markeredgewidth=0.6*s,
             markerfacecolor='white', lw=1.5*s, elinewidth=1.2*s, zorder=6)

# 17Ne halo benchmark (red star)
I_h = data['halo'][0]['I']
v_h = data['halo'][0]['value']
s_h = data['halo'][0]['sigma']
ax1.errorbar(I_h, v_h, yerr=s_h, fmt='*', color='#C62828',
             markersize=18*s, capsize=0, markeredgecolor='#8B0000', markeredgewidth=0.8*s,
             markerfacecolor='#C62828', elinewidth=1.5*s, zorder=7)

# Annotations
fs_lab = 12*s*f
labels = ['$^{25}$Al–$^{25}$Mg', '$^{24}$Al–$^{24}$Na', '$^{23}$Al–$^{23}$Ne']
offsets = [(14, -12), (10, -12), (10, -24)]
for d, lbl, (dx, dy) in zip(data['mirror_exp'], labels, offsets):
    ax1.annotate(lbl, (d['I'], d['value']),
                 textcoords="offset points",
                 xytext=(dx*s, dy*s), fontsize=fs_lab,
                 fontweight='bold', color='#333')
ax1.annotate('$^{22}$Al(meas.)\n– $^{22}$F(pred.)', (I_p, v_p),
             textcoords="offset points", xytext=(8*s, -36*s),
             fontsize=11*s*f, fontweight='bold', color='#555', fontstyle='italic')
ax1.annotate('$^{17}$Ne–$^{17}$N', (I_h, v_h),
             textcoords="offset points", xytext=(-18*s, 4*s),
             fontsize=fs_lab, fontweight='bold', color='#8B0000', ha='right')

# Axes
fs_ax = 16*s*f
fs_tl = 16*s*f
ax1.set_xlabel(r'Isospin asymmetry $I = |N-Z|/A$',
               fontsize=fs_ax)
ax1.set_xticks([ii*0.04 for ii in range(6)])
ax1.set_ylabel(r'$\Delta R_{\rm ch}^{\rm mirror}$ (fm)',
               fontsize=fs_ax)
ax1.set_xlim(0.02, 0.215)
ax1.set_ylim(0, 0.68)
ax1.tick_params(labelsize=fs_tl, length=6*s, width=1*s, direction="in")

# Legend
fs_leg = 12*s*f
legend_elements = [
    Line2D([0],[0], marker='s', color=c_exp, markerfacecolor=c_exp,
           markeredgecolor='k', markersize=10*s, ls=':',
           label=r'This work ($\Delta R_{\rm ch}^{\rm mirror}$)'),
    Line2D([0],[0], marker='s', color='#78909C', markerfacecolor='#78909C',
           markeredgecolor='k', markersize=7*s, ls='', alpha=0.6,
           label=r'Literature'),
    plt.Rectangle((0,0),1,1, fc='#43A047', alpha=0.12, label=r'CC/AFDMC'),
    Line2D([0],[0], marker='s', color=c_exp, markerfacecolor='white',
           markeredgecolor='k', markersize=10*s, ls=':',
           label=r'Predicted $^{22}$F'),
    Line2D([0],[0], marker='*', color='#C62828', markerfacecolor='#C62828',
           markeredgecolor='#8B0000', markersize=18*s, ls='',
           label=r'$^{17}$Ne (two-proton halo)'),
]
leg = ax1.legend(handles=legend_elements, fontsize=fs_leg, handletextpad=0.4, borderaxespad=0.5,
                 labelspacing=0.3, frameon=True, facecolor='white', edgecolor='black',
                 loc='upper left', framealpha=0.5, bbox_to_anchor=(0.01,0.99),fancybox=False)
leg.get_frame().set_linewidth(2.0) 


# Spines and tick fonts
for sp in ax1.spines.values():
    sp.set_linewidth(1.2*s)

plt.tight_layout()
# plt.show(); quit()
plt.savefig('fig9_mirror_shift.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('fig9_mirror_shift.png', dpi=200, bbox_inches='tight')
print("Saved fig9_mirror_shift.pdf and .png")
