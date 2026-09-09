"""APEX architecture figure. Boxes are sized to measured text and checked for overlap."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

plt.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Liberation Serif","DejaVu Serif"],
 "mathtext.fontset":"stix","pdf.fonttype":42,"ps.fonttype":42,
 "savefig.bbox":"tight","savefig.pad_inches":0.03})

fig = plt.figure(figsize=(7.16, 1.84))
ax  = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1000); ax.set_ylim(0,428); ax.axis('off')
fig.canvas.draw(); REND = fig.canvas.get_renderer()
inv = ax.transData.inverted()

FS, PADX, PADY = 7.0, 11.0, 7.0
NODES = {}

def measure(t):
    bb = t.get_window_extent(renderer=REND)
    p  = Bbox(inv.transform(bb.get_points()))
    return p.width, p.height

def node(key, cx, cy, label, fc, ec, fs=FS, padx=PADX, pady=PADY, lw=1.0, ls='-', bold=False):
    t = ax.text(cx, cy, label, ha='center', va='center', fontsize=fs, zorder=6,
                linespacing=1.42, fontweight='bold' if bold else 'normal')
    w, h = measure(t)
    x0, y0, W, H = cx-w/2-padx, cy-h/2-pady, w+2*padx, h+2*pady
    ax.add_patch(FancyBboxPatch((x0,y0), W, H, boxstyle="round,pad=0,rounding_size=6",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=5))
    NODES[key] = dict(cx=cx, cy=cy, x0=x0, y0=y0, x1=x0+W, y1=y0+H)
    return NODES[key]

def free(cx, cy, label, fs=6.6, color='#555', ha='center', va='center', style='normal', weight='normal'):
    ax.text(cx, cy, label, ha=ha, va=va, fontsize=fs, color=color, zorder=6,
            linespacing=1.55, style=style, fontweight=weight)

def arrow(p, q, c='#4A4A4A', lw=.95, ls='-', z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=8,
                 color=c, lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0))

def between(a, b, side, c='#4A4A4A', lw=.95, pad=3):
    A, B = NODES[a], NODES[b]
    if side=='rl': arrow((A['x1']+pad, A['cy']), (B['x0']-pad, B['cy']), c, lw)
    if side=='tb': arrow((A['cx'], A['y0']-pad), (B['cx'], B['y1']+pad), c, lw)

# ---------------- stage 1 ----------------
node('img',  152, 358, 'Medical image  $X_i$',                              '#FFFFFF', '#9A9A9A')
node('bb',   152, 288, 'Pre-trained network\nViT / ConvNeXt / MobileNetV3', '#FFFFFF', '#4F4F4F', fs=6.7)
node('z',     86, 196, 'latent\n$\\mathbf{z}_i\\in\\mathbb{R}^{d}$',        '#FFF3DC', '#E69F00', fs=6.7)
node('s',    222, 196, 'logits\n$\\mathbf{s}_i$',                           '#E2EFF8', '#0072B2', fs=6.7)
node('m',    152,  86, 'geometric descriptors  $\\mathbf{m}_i\\in\\mathbb{R}^{62}$\n'
                       'colour · GLCM · LBP · shape · wavelet',              '#FFF3DC', '#E69F00', fs=6.4, ls=(0,(2.8,1.8)))
between('img','bb','tb')
arrow((120, NODES['bb']['y0']-3), (NODES['z']['cx']+10, NODES['z']['y1']+3))
arrow((184, NODES['bb']['y0']-3), (NODES['s']['cx']-10, NODES['s']['y1']+3))

g = [NODES[k] for k in ('img','bb','z','s','m')]
gx0,gx1 = min(n['x0'] for n in g)-14, max(n['x1'] for n in g)+14
gy0,gy1 = min(n['y0'] for n in g)-12, max(n['y1'] for n in g)+12
ax.add_patch(mpatches.Rectangle((gx0,gy0), gx1-gx0, gy1-gy0, fc='#F4F4F4', ec='#BFBFBF',
             lw=.9, ls=(0,(4,2.2)), zorder=0))
free((gx0+gx1)/2, gy1+13, 'STAGE 1  ·  FROZEN BACKBONE', fs=7.0, color='#6E6E6E', va='bottom', weight='bold')

# ---------------- stage 2 ----------------
node('cat',  404, 150, 'fuse\n$\\mathbf{w}_i=[\\mathbf{z}_i;\\mathbf{m}_i]$',            '#FFFFFF', '#D55E00', fs=6.7)
node('prj',  404, 244, 'truncated SVD\n$\\Phi:\\mathbb{R}^{d+62}\\!\\to\\!\\mathbb{R}^{64}$', '#FFFFFF', '#A8A8A8', fs=6.4)
node('rbf',  596, 300, 'RBF SVM   $\\mathbf{p}^{\\mathrm{rbf}}$',  '#E9F6F0', '#009E73', fs=6.7)
node('poly', 596, 228, 'cubic SVM   $\\mathbf{p}^{\\mathrm{poly}}$', '#E9F6F0', '#009E73', fs=6.7)
node('knn',  596, 156, '$k$-NN   $\\mathbf{p}^{\\mathrm{knn}}$',   '#E9F6F0', '#009E73', fs=6.7)
node('gate', 768, 228, 'confidence gate\n$\\max_k p^{\\mathrm{b}}_{ik} < \\tau$ ?', '#F8E9F3', '#CC79A7', fs=6.6)
node('out',  920, 228, 'calibrated\n$\\hat{\\mathbf{p}}_i$', '#FFFFFF', '#2A2A2A', fs=7.4, lw=1.3, bold=True)

arrow((NODES['z']['x1']+3, NODES['z']['cy']-5), (NODES['cat']['x0']-3, NODES['cat']['cy']+5))
arrow((NODES['m']['x1']+3, NODES['m']['cy']),   (NODES['cat']['x0']-3, NODES['cat']['cy']-12))
between('cat','prj','tb'); NODES['cat'],NODES['prj']=NODES['cat'],NODES['prj']
arrow((NODES['cat']['cx'], NODES['cat']['y1']+3), (NODES['prj']['cx'], NODES['prj']['y0']-3))
for k in ('rbf','poly','knn'):
    arrow((NODES['prj']['x1']+3, NODES['prj']['cy']), (NODES[k]['x0']-3, NODES[k]['cy']))
    arrow((NODES[k]['x1']+3, NODES[k]['cy']), (NODES['gate']['x0']-3, NODES['gate']['cy']))
between('gate','out','rl', c='#2A2A2A', lw=1.15)

yTop = 372
px = NODES['s']['x1']+3
for seg in ([(px,326)],):
    pass
ax.plot([px, 318], [NODES['s']['cy']]*2, color='#0072B2', lw=1.0, ls=(0,(3.4,2)), zorder=3)
ax.plot([318, 318], [NODES['s']['cy'], yTop], color='#0072B2', lw=1.0, ls=(0,(3.4,2)), zorder=3)
ax.plot([318, NODES['gate']['cx']], [yTop]*2, color='#0072B2', lw=1.0, ls=(0,(3.4,2)), zorder=3)
arrow((NODES['gate']['cx'], yTop), (NODES['gate']['cx'], NODES['gate']['y1']+3),
      c='#0072B2', lw=1.0, ls=(0,(3.4,2)), z=3)
free(400, yTop-5, '$\\mathbf{p}^{\\mathrm{b}}_i$', fs=7.4, color='#0072B2', va='top')

h = [NODES[k] for k in ('cat','prj','rbf','poly','knn','gate')]
hx0,hx1 = min(n['x0'] for n in h)-16, max(n['x1'] for n in h)+16
hy0,hy1 = min(n['y0'] for n in h)-40, yTop+24
ax.add_patch(mpatches.Rectangle((hx0,hy0), hx1-hx0, hy1-hy0, fc='none', ec='#D55E00',
             lw=1.2, ls=(0,(5.5,2.8)), zorder=1))
free((hx0+hx1)/2, hy1+13, 'STAGE 2  ·  APEX POST-HOC FUSION LAYER', fs=7.0, color='#D55E00',
     va='bottom', weight='bold')
free(596, 344, 'kernel experts, fitted on $\\mathcal{D}_{\\mathrm{fit}}$', fs=6.5, color='#009E73', style='italic')
free(NODES['gate']['cx'], NODES['gate']['y0']-9, 'weights fitted on $\\mathcal{D}_{\\mathrm{cal}}$',
     fs=6.3, color='#CC79A7', style='italic', ha='center', va='top')
free((hx0+hx1)/2, hy0+19,
     '$\\hat{\\mathbf{p}}_i \\;=\\; \\tilde{w}_1(\\mathbf{p}^{\\mathrm{b}}_i)\\,\\mathbf{p}^{\\mathrm{b}}_i'
     '\\;+\\; w_2\\,\\mathbf{p}^{\\mathrm{rbf}}_i \\;+\\; w_3\\,\\mathbf{p}^{\\mathrm{poly}}_i'
     '\\;+\\; w_4\\,\\mathbf{p}^{\\mathrm{knn}}_i$', fs=7.9, color='#333333')

# ---- overlap check ----
def ov(a,b,m=2):
    return not (a['x1']+m<=b['x0'] or b['x1']+m<=a['x0'] or a['y1']+m<=b['y0'] or b['y1']+m<=a['y0'])
ks=list(NODES); bad=[(ks[i],ks[j]) for i in range(len(ks)) for j in range(i+1,len(ks)) if ov(NODES[ks[i]],NODES[ks[j]])]
print("overlapping node pairs:", bad if bad else "none")

plt.savefig('plots/fig1_architecture.pdf'); plt.savefig('plots/fig1_architecture.png', dpi=340)
plt.close()
