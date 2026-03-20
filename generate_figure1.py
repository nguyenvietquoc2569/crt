"""
Generate Figure 1: Mimi-VC System Architecture Diagram
For IEEE conference paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================================================
# Configuration
# ============================================================
FIG_W, FIG_H = 7.5, 3.6
DPI = 300

# Colors
C_AUDIO    = '#D6DCE4'
C_ENCODER  = '#4472C4'
C_SEMANTIC = '#548235'
C_ACOUSTIC = '#C55A11'
C_TRANSF   = '#7030A0'
C_RETRIEV  = '#9966CC'
C_TRANSFRM = '#B366CC'
C_TARGET   = '#BF8F00'
C_ARROW    = '#333333'
C_KEEP     = '#548235'
C_REPLACE  = '#C55A11'

FONT = 'serif'
plt.rcParams['font.family'] = FONT
plt.rcParams['font.size'] = 8


# ============================================================
# Helpers
# ============================================================
def box(ax, x, y, w, h, label, fc, tc='white', fs=8, fw='bold',
        ec='#333333', lw=0.8, sub=None, sub_fs=6.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    if sub:
        ax.text(x + w/2, y + h/2 + 0.12, label, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=fw, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.15, sub, ha='center', va='center',
                fontsize=sub_fs, color=tc, fontstyle='italic', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fs, color=tc, fontweight=fw, zorder=3)


def arr(ax, x1, y1, x2, y2, c=C_ARROW, lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                shrinkA=2, shrinkB=2), zorder=1)


def carr(ax, x1, y1, x2, y2, c=C_ARROW, lw=1.2, rad=0.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=2, shrinkB=2), zorder=1)


def lbl(ax, x, y, txt, fs=6.5, c='#555555', ha='center'):
    ax.text(x, y, txt, ha=ha, va='center', fontsize=fs, color=c,
            fontstyle='italic', zorder=4)


# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 15.5)
ax.set_ylim(-0.1, 4.3)
ax.axis('off')

BH = 0.55   # box height
SW = 1.3    # small width
MW = 1.5    # medium width
LW = 1.8    # large width

# Y positions
Y1 = 3.1    # main top row (source audio, encoder, CB0, merge, decoder, output)
Y2 = 1.9    # middle row (CB1-7, transformation)
Y3 = 0.5    # bottom row (target path)

# ============================================================
# SOURCE PATH - Top Row
# ============================================================
# Source Audio
box(ax, 0.5, Y1, SW, BH, 'Source', C_AUDIO, tc='#333333', sub='Audio')
arr(ax, 0.5+SW, Y1+BH/2, 2.2, Y1+BH/2)

# Mimi Encoder
box(ax, 2.25, Y1, LW, BH, 'Mimi', C_ENCODER, sub='Encoder')

# Split arrow to CB0
arr(ax, 2.25+LW, Y1+BH/2, 4.6, Y1+BH/2, c=C_KEEP)

# CB0 Semantic
box(ax, 4.65, Y1, LW, BH, 'CB0', C_SEMANTIC, sub='Semantic')

# Split arrow down to CB1-7
arr(ax, 2.25+LW, Y1+BH/2-0.05, 4.6, Y2+BH/2, c=C_REPLACE)

# CB1-7 Acoustic
box(ax, 4.65, Y2, LW, BH, 'CB1\u20137', C_ACOUSTIC, sub='Acoustic')

# ============================================================
# SEMANTIC PRESERVE PATH (dashed green)
# ============================================================
# Long arrow from CB0 to Combined box
arr(ax, 4.65+LW, Y1+BH/2, 10.4, Y1+BH/2, c=C_KEEP, lw=1.5)
lbl(ax, 8.5, Y1+BH/2+0.18, 'preserve', c=C_SEMANTIC, fs=7)

# Dashed outline for preserve path
dash = FancyBboxPatch(
    (6.55, Y1-0.06), 3.7, BH+0.12, boxstyle="round,pad=0.05",
    facecolor='none', edgecolor=C_SEMANTIC, linewidth=0.8,
    linestyle='--', zorder=1
)
ax.add_patch(dash)

# ============================================================
# ACOUSTIC TRANSFORM PATH
# ============================================================
# Arrow from CB1-7 to Token Translator
arr(ax, 4.65+LW, Y2+BH/2, 7.0, Y2+BH/2, c=C_REPLACE)

# Token Translator block
tw, th = 2.6, 0.85
ty = Y2 - 0.15
box(ax, 7.05, ty, tw, th, '', C_TRANSF)
ax.text(7.05+tw/2, ty+th-0.12, 'Token Translator', ha='center', va='center',
        fontsize=7.5, color='white', fontweight='bold', zorder=3)

# Sub-boxes: Retrieval and Learned
sbw, sbh = 1.1, 0.35
sby = ty + 0.08
box(ax, 7.2, sby, sbw, sbh, 'Retrieval', C_RETRIEV, fs=6.5, sub='(FAISS)', sub_fs=5.5)
box(ax, 8.4, sby, sbw, sbh, 'Learned', C_TRANSFRM, fs=6.5, sub='(Transformer)', sub_fs=5.5)
ax.text(8.35, sby+sbh/2, '/', ha='center', va='center', fontsize=8,
        color='white', fontweight='bold', zorder=4)

# Arrow from translator to Combined
arr(ax, 7.05+tw, Y2+BH/2, 10.4, Y2+BH/2, c=C_REPLACE)
lbl(ax, 10.0, Y2+BH/2+0.18, 'replace', c=C_ACOUSTIC, fs=7)

# ============================================================
# COMBINED TOKENS
# ============================================================
cw, ch = 1.5, 1.5
cy = Y2 - 0.12
box(ax, 10.45, cy, cw, ch, '', '#2E5090', ec='#1B3A6B')
ax.text(10.45+cw/2, cy+ch-0.18, 'Combined', ha='center', va='center',
        fontsize=7, color='white', fontweight='bold', zorder=3)
ax.text(10.45+cw/2, cy+ch-0.4, 'Tokens', ha='center', va='center',
        fontsize=7, color='white', fontweight='bold', zorder=3)

# Small CB0 and CB1-7' inside
smw, smh = 1.1, 0.28
box(ax, 10.45+(cw-smw)/2, cy+0.55, smw, smh, 'CB0', C_SEMANTIC, fs=6.5, lw=0.5)
box(ax, 10.45+(cw-smw)/2, cy+0.14, smw, smh, "CB1\u20137'", C_ACOUSTIC, fs=6.5, lw=0.5)

# ============================================================
# DECODER + OUTPUT
# ============================================================
# Arrow from Combined to Decoder
mid_y = cy + ch/2
arr(ax, 10.45+cw, mid_y, 12.35, mid_y)

# Mimi Decoder
box(ax, 12.4, mid_y-BH/2, LW, BH, 'Mimi', C_ENCODER, sub='Decoder')

# Arrow to Converted Audio
arr(ax, 12.4+LW, mid_y, 14.5, mid_y)

# Converted Audio - align vertically with decoder
box(ax, 14.0, mid_y-BH/2, SW+0.15, BH, 'Converted', C_AUDIO,
    tc='#333333', sub='Audio')

# ============================================================
# TARGET PATH - Bottom Row
# ============================================================
# Target Audio
box(ax, 0.5, Y3, SW, BH, 'Target', C_TARGET, tc='#333333', sub='Audio')
arr(ax, 0.5+SW, Y3+BH/2, 2.2, Y3+BH/2, c=C_TARGET)

# Mimi Encoder (target)
box(ax, 2.25, Y3, LW, BH, 'Mimi', C_ENCODER, sub='Encoder')
arr(ax, 2.25+LW, Y3+BH/2, 4.6, Y3+BH/2, c=C_TARGET)

# FAISS Index / Training Data
iw = 2.3
box(ax, 4.65, Y3, iw, BH, 'FAISS Index /', '#8B6914', fs=7,
    sub='Training Data', sub_fs=6)

# Curved arrow from index up to translator
carr(ax, 4.65+iw/2, Y3+BH, 7.05+tw/2, ty, c=C_TARGET, lw=1.2, rad=-0.25)
lbl(ax, 5.3, Y2-0.55, 'target speaker\nreference', fs=6, c='#8B6914')

# ============================================================
# SEPARATOR & PATH LABELS
# ============================================================
ax.axhline(y=1.35, xmin=0.02, xmax=0.65, color='#CCCCCC',
           linestyle=':', linewidth=0.8, zorder=0)

ax.text(0.15, Y1+BH/2, 'Source\nPath', ha='center', va='center',
        fontsize=6, color='#888888', fontweight='bold', fontstyle='italic')
ax.text(0.15, Y3+BH/2, 'Target\nPath', ha='center', va='center',
        fontsize=6, color='#888888', fontweight='bold', fontstyle='italic')

# ============================================================
# Save
# ============================================================
plt.tight_layout(pad=0.2)
out = '/Users/nguyenquoc/duplex/figure1_architecture.png'
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')

# Also save as high-res for paper
out_hires = '/Users/nguyenquoc/duplex/figure1_architecture_hires.png'
fig.savefig(out_hires, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.close()
print(f'Saved: {out} ({DPI} DPI)')
print(f'Saved: {out_hires} (600 DPI)')
