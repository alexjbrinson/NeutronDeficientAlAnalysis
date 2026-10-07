"""
Figure 3 — Mirror charge radii and proton skin
Reads data from fig3_data.csv

Usage: python3 fig3_mirror_proton_skin.py
"""
import csv
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# -------- Load data ----------------------------------------------------------
data = {}
with open('fig3_data.csv') as f:
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
c_exp     = '#000000'
c_nleft   = '#D32F2F'
c_frame   = '#42A5F5'

s, f = 1.5, 1.1
fig, ax1 = plt.subplots(figsize=(12, 8))

# CC/AFDMC band
I_lin = np.linspace(0, 0.22, 200)
ax1.fill_between(I_lin, 1.574*I_lin - 0.021, 1.574*I_lin + 0.021,
                 alpha=0.10, color='#43A047', zorder=0)

# Mirror experimental
I_e   = cols('mirror_exp', 'I')
v_e   = cols('mirror_exp', 'value')
s_e   = cols('mirror_exp', 'sigma')
ax1.errorbar(I_e, v_e, yerr=s_e, fmt='s:', color=c_exp,
             markersize=10*s, capsize=0, markeredgecolor='k',
             markeredgewidth=0.6*s, markerfacecolor=c_exp,
             lw=1.5*s, elinewidth=1.2*s, zorder=6)

# Predicted 22F
I_p = data['mirror_pred'][0]['I']
v_p = data['mirror_pred'][0]['value']
s_p = data['mirror_pred'][0]['sigma']
ax1.errorbar([I_e[-1], I_p], [v_e[-1], v_p], fmt=':', color=c_exp,
             lw=1.5*s, zorder=5)
ax1.errorbar(I_p, v_p, yerr=s_p, fmt='s', color=c_exp,
             markersize=10*s, capsize=0, markeredgecolor='k',
             markeredgewidth=0.6*s, markerfacecolor='white',
             lw=1.5*s, elinewidth=1.2*s, zorder=6)

# NLEFT
I_n = np.array(cols('nleft_skin', 'I')) - 0.003
v_n = cols('nleft_skin', 'value')
s_n = cols('nleft_skin', 'sigma')
ax1.errorbar(I_n, v_n, yerr=s_n, fmt='o:', color=c_nleft,
             markersize=10*s, capsize=0, markeredgecolor=c_nleft,
             markeredgewidth=1.5*s, markerfacecolor='white',
             lw=1.3*s, elinewidth=1.2*s, zorder=4)

# FRAME
I_f = np.array(cols('frame_skin', 'I')) + 0.003
v_f = cols('frame_skin', 'value')
s_f = cols('frame_skin', 'sigma')
ax1.errorbar(I_f, v_f, yerr=s_f, fmt='D:', color=c_frame,
             markersize=9*s, capsize=0, markeredgecolor=c_frame,
             markeredgewidth=1.5*s, markerfacecolor='white',
             lw=1.3*s, elinewidth=1.2*s, zorder=4)

# Annotations
fs_lab = 10*s*f
labels = ['$^{25}$Al–$^{25}$Mg', '$^{24}$Al–$^{24}$Na', '$^{23}$Al–$^{23}$Ne']
offsets = [(14, -12), (10, -12), (10, -24)]
for d, lbl, (dx, dy) in zip(data['mirror_exp'], labels, offsets):
    ax1.annotate(lbl, (d['I'], d['value']),
                 textcoords="offset points",
                 xytext=(dx*s, dy*s), fontsize=fs_lab,
                 fontweight='bold', color='#333')
ax1.annotate('$^{22}$Al(meas.)\n –$^{22}$F(pred.)\n', (I_p, v_p),
             textcoords="offset points", xytext=(8*s, -36*s),
             fontsize=9*s*f, fontweight='bold', color='#555', fontstyle='italic')

# Axes
fs_tl = 16*s*f
fs_ax = 16*s*f
ax1.set_xlabel(r'Isospin asymmetry $I = |N-Z|/A$',#r'${Isospin\ asymmetry}\ I = |N-Z|/A$',
               fontsize=fs_ax)
ax1.set_ylabel(r'$\Delta R_{\rm ch}^{\rm mirror}$ (fm)',#$\mathbf{(fm)}$',
               fontsize=fs_ax)
ax1.set_yticks([i/10 for i in range(5)],[fr'${i/10}$' for i in range(5)])
ax1.set_xlim(0.017, 0.217)#(0.02, 0.215)
ax1.set_ylim(0, 0.4)#(-0.005, 0.40)
ax1.tick_params(labelsize=fs_tl, length=6*s, width=1*s, direction='in')
# ax1.set_xticks([(13-n)/(13+n) for n in range(9,13)],[f'${(13-n)}/{(13+n)}$' for n in range(9,13)])
ax1.set_xticks([(13-n)/(13+n) for n in range(9,13)],[r'%.3f'%((13-n)/(13+n)) for n in range(9,13)])

# Right axis (proton skin scale)
ax2 = ax1.twinx()
ax2.set_yticks([i/10 for i in range(5)],[fr'{i/10}' for i in range(5)])
ax2.set_ylim(ax1.get_ylim())
ax2.set_ylabel(r'$\Delta R_{pn}$ of Al isotopes (fm)',#r'$\Delta R_{pn}$ $\mathbf{of\ Al\ isotopes\ (fm)}$',
               fontsize=fs_ax, color='#555',
               rotation=270, labelpad=30*s)
ax2.tick_params(labelsize=fs_tl, length=6*s, width=1*s, direction='in')


# Top axis (mass number)
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xticks([1/25, 2/24, 3/23, 4/22])
ax3.set_xticklabels([fr'{s}' for s in ['25', '24', '23', '22']],
                    fontsize=11*s*f)
ax3.set_xlabel(r'Mass number $A$',#r'$\mathbf{Mass\ number}\ A$',
               fontsize=fs_ax, labelpad=10*s)
ax3.tick_params(labelsize=fs_tl, length=6*s, width=1*s, direction='in')

# Legend
fs_leg = 10.5*s*f
legend_elements = [
    Line2D([0],[0], marker='s', color=c_exp, markerfacecolor=c_exp,
           markeredgecolor='k', markersize=10*s, ls=':',
           label=r'This work ($\Delta R_{\rm ch}^{\rm mirror}$)'),
    Line2D([0],[0], marker='o', color=c_nleft, markerfacecolor='white',
           markeredgecolor=c_nleft, markeredgewidth=1.5*s,
           markersize=10*s, ls=':',
           label=r'NLEFT ($r_{pp} - r_{nn}$)'),
    Line2D([0],[0], marker='D', color=c_frame, markerfacecolor='white',
           markeredgecolor=c_frame, markeredgewidth=1.5*s,
           markersize=9*s, ls=':',
           label=r'FRAME ($r_{pp} - r_{nn}$)'),
    plt.Rectangle((0,0),1,1, fc='#43A047', alpha=0.12, label=r'CC/AFDMC'),
    Line2D([0],[0], marker='s', color=c_exp, markerfacecolor='white',
           markeredgecolor='k', markersize=10*s, ls=':',
           label=r'Predicted $^{22}$F'),
]
leg = ax1.legend(handles=legend_elements, fontsize=fs_leg, handletextpad=0.4, borderaxespad=0.5,
                 labelspacing=0.3, frameon=True, facecolor='white', edgecolor='black',
                 loc='upper left', framealpha=0.5, bbox_to_anchor=(0.01,0.99),fancybox=False)

leg.get_frame().set_linewidth(2.0) 
leg.get_frame().set_edgecolor('black')
# for t in leg.get_texts():
#     t.set_fontweight('bold')

# Spines and tick fonts
for ax in [ax1, ax2, ax3]:
    for sp in ax.spines.values():
        sp.set_linewidth(1.2*s)
# for ax in [ax1, ax2]:
#     for lab in ax.get_xticklabels() + ax.get_yticklabels():
#         lab.set_fontweight('bold')

plt.tight_layout()
# plt.show(); quit()
plt.savefig('fig3_mirror_proton_skin.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('fig3_mirror_proton_skin.png', dpi=200, bbox_inches='tight')
print("Saved fig3_mirror_proton_skin.pdf and .png")
