#!/usr/bin/env python3
"""Generate professional charts for the CRT paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

outdir = '/Users/nguyenquoc/duplex/paper'

# ── Figure 1: MCD Comparison Bar Chart ──
fig, ax = plt.subplots(figsize=(4.5, 3.2))

methods = ['CRT\nconf>0.4', 'FAISS\nbaseline', 'kNN-VC', 'FreeVC', 'CRT\nargmax']
mcd_vals = [102.45, 106.61, 115.99, 118.08, 127.66]
colors = ['#2166AC', '#67A9CF', '#D6604D', '#B2182B', '#FDDBC7']

bars = ax.bar(methods, mcd_vals, color=colors, edgecolor='#333333', linewidth=0.5, width=0.65)

# Add value labels
for bar, val in zip(bars, mcd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add improvement annotations
ax.annotate('', xy=(0, 102.45), xytext=(1, 106.61),
            arrowprops=dict(arrowstyle='<->', color='#2166AC', lw=1.5))
ax.text(0.5, 104.2, '-3.9%', ha='center', fontsize=7, color='#2166AC', fontweight='bold')

ax.annotate('', xy=(0, 102.45), xytext=(2, 115.99),
            arrowprops=dict(arrowstyle='<->', color='#D6604D', lw=1.2))
ax.text(1.0, 108.5, '-11.7%', ha='center', fontsize=7, color='#D6604D', fontweight='bold')

ax.set_ylabel('Mel Cepstral Distortion (MCD) ↓')
ax.set_title('MCD Comparison Across Methods')
ax.set_ylim(95, 135)
ax.axhline(y=106.61, color='#67A9CF', linestyle='--', alpha=0.5, linewidth=0.8, label='FAISS baseline')
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
fig.savefig(f'{outdir}/fig1_mcd_comparison.png', dpi=300)
plt.close()
print('Saved fig1_mcd_comparison.png')

# ── Figure 2: Confidence Threshold vs MCD ──
fig, ax1 = plt.subplots(figsize=(4.5, 3.2))

tau = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mcd = [109.29, 102.45, 109.40, 109.02, 109.40, 109.51, 109.29]
override = [26.2, 17.8, 10.0, 6.1, 3.2, 2.6, 1.3]

color_mcd = '#2166AC'
color_ovr = '#D6604D'

ax1.plot(tau, mcd, 'o-', color=color_mcd, linewidth=2, markersize=6, label='MCD', zorder=3)
ax1.scatter([0.4], [102.45], s=120, color=color_mcd, zorder=4, edgecolors='black', linewidth=1.5)
ax1.axhline(y=106.61, color='#999999', linestyle='--', alpha=0.7, linewidth=0.8, label='FAISS baseline')
ax1.set_xlabel('Confidence Threshold (τ)')
ax1.set_ylabel('MCD ↓', color=color_mcd)
ax1.tick_params(axis='y', labelcolor=color_mcd)
ax1.set_ylim(99, 112)

ax2 = ax1.twinx()
ax2.bar(tau, override, alpha=0.25, color=color_ovr, width=0.06, label='Override %')
ax2.set_ylabel('Override % →', color=color_ovr)
ax2.tick_params(axis='y', labelcolor=color_ovr)
ax2.set_ylim(0, 35)

# Annotate optimal point
ax1.annotate('τ = 0.4\nMCD = 102.45\n17.8% override',
            xy=(0.4, 102.45), xytext=(0.55, 101.0),
            fontsize=7, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F0FE', edgecolor='#2166AC', alpha=0.9))

ax1.set_title('Confidence Threshold vs. MCD and Override Rate')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

plt.tight_layout()
fig.savefig(f'{outdir}/fig2_confidence_threshold.png', dpi=300)
plt.close()
print('Saved fig2_confidence_threshold.png')

# ── Figure 3: SECS vs MCD scatter ──
fig, ax = plt.subplots(figsize=(4.5, 3.2))

# All methods
all_methods = {
    'CRT conf>0.4': (0.816, 102.45),
    'FAISS baseline': (0.817, 106.61),
    'CRT conf>0.6': (0.827, 109.02),
    'CRT conf>0.9': (0.815, 109.29),
    'kNN-VC': (0.900, 115.99),
    'FreeVC': (0.738, 118.08),
    'CRT argmax': (0.706, 127.66),
}

for name, (secs, mcd) in all_methods.items():
    if 'CRT conf>0.4' in name:
        ax.scatter(secs, mcd, s=120, c='#2166AC', marker='*', zorder=5, edgecolors='black', linewidth=0.8)
        ax.annotate(name, (secs, mcd), xytext=(-40, -15), textcoords='offset points', fontsize=7, fontweight='bold', color='#2166AC')
    elif 'kNN-VC' in name:
        ax.scatter(secs, mcd, s=80, c='#D6604D', marker='D', zorder=4, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (secs, mcd), xytext=(5, 5), textcoords='offset points', fontsize=7, color='#D6604D')
    elif 'FreeVC' in name:
        ax.scatter(secs, mcd, s=80, c='#B2182B', marker='s', zorder=4, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (secs, mcd), xytext=(5, 5), textcoords='offset points', fontsize=7, color='#B2182B')
    elif 'FAISS' in name:
        ax.scatter(secs, mcd, s=80, c='#67A9CF', marker='^', zorder=4, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (secs, mcd), xytext=(5, -12), textcoords='offset points', fontsize=7, color='#67A9CF')
    elif 'argmax' in name:
        ax.scatter(secs, mcd, s=60, c='#CCCCCC', marker='x', zorder=3)
        ax.annotate(name, (secs, mcd), xytext=(5, 5), textcoords='offset points', fontsize=7, color='#999999')
    else:
        ax.scatter(secs, mcd, s=50, c='#92C5DE', marker='o', zorder=3, alpha=0.7)

# Ideal region
ax.annotate('Ideal\n(↑SECS, ↓MCD)', xy=(0.92, 98), fontsize=7, fontstyle='italic', color='#666666',
            ha='center')

ax.set_xlabel('SECS (Speaker Similarity) →')
ax.set_ylabel('MCD (Spectral Distortion) ↓')
ax.set_title('SECS vs. MCD Trade-off')
ax.set_xlim(0.68, 0.94)
ax.set_ylim(96, 132)

plt.tight_layout()
fig.savefig(f'{outdir}/fig3_secs_mcd_tradeoff.png', dpi=300)
plt.close()
print('Saved fig3_secs_mcd_tradeoff.png')

# ── Figure 4: Architecture diagram (clean linear flow) ──
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Color palette — muted, professional
C_ENC    = ('#EAF2FB', '#2166AC', '#1A3F6B')  # fill, edge, text
C_RETR   = ('#FFF1E0', '#CC6A00', '#7A3F00')
C_CRT    = ('#E8F5E9', '#2E7D32', '#1B4D20')
C_DEC    = ('#F3E8F7', '#7B1FA2', '#4A1363')
C_COND   = ('#FCE4EC', '#C62828', '#7F1818')
C_GATE   = ('#FFFDE7', '#AF8B00', '#5F4C00')

def rounded_box(ax, x, y, w, h, text, fill, edge, textcolor, fontsize=8, bold=True):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            color=textcolor, zorder=3)

def arrow(ax, x1, y1, x2, y2, color='#444', lw=1.3, style='-|>'):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw, zorder=1
    )
    ax.add_patch(a)

# Main horizontal flow: Source → Encode → Retrieve → CRT → Decode → Output
# All main boxes on y=3.0 with height=1.3, centered vertically
Y_MAIN = 3.0
H_BOX = 1.3

# Box 1: Source Audio (input label, not in pipeline box)
ax.text(0.5, Y_MAIN + H_BOX/2, 'Source\naudio',
        ha='center', va='center', fontsize=8,
        fontweight='bold', color='#333')

# Box 2: Mimi Encoder
rounded_box(ax, 1.4, Y_MAIN, 1.9, H_BOX, 'Mimi\nEncoder', *C_ENC, fontsize=8)

# Tensor label between encoder and retrieval
ax.text(3.7, Y_MAIN + H_BOX + 0.15, 'tokens',
        ha='center', va='bottom', fontsize=6.5, style='italic', color='#666')
ax.text(3.7, Y_MAIN - 0.15, '[32 × T]',
        ha='center', va='top', fontsize=6, family='monospace', color='#888')

# Box 3: FAISS Retrieval
rounded_box(ax, 4.1, Y_MAIN, 2.0, H_BOX, 'FAISS\nRetrieval', *C_RETR, fontsize=8)

ax.text(6.45, Y_MAIN + H_BOX + 0.15, 'noisy tokens',
        ha='center', va='bottom', fontsize=6.5, style='italic', color='#666')
ax.text(6.45, Y_MAIN - 0.15, '[19 × T]',
        ha='center', va='top', fontsize=6, family='monospace', color='#888')

# Box 4: CRT (highlighted — this is the contribution)
rounded_box(ax, 6.8, Y_MAIN - 0.15, 2.4, H_BOX + 0.3,
            'CRT\n(Conformer)\n707M', *C_CRT, fontsize=8)

ax.text(9.55, Y_MAIN + H_BOX + 0.15, 'refined tokens',
        ha='center', va='bottom', fontsize=6.5, style='italic', color='#666')
ax.text(9.55, Y_MAIN - 0.15, '[19 × T]',
        ha='center', va='top', fontsize=6, family='monospace', color='#888')

# Box 5: Mimi Decoder
rounded_box(ax, 9.9, Y_MAIN, 2.0, H_BOX, 'Mimi\nDecoder', *C_DEC, fontsize=8)

# Output label
ax.text(13.1, Y_MAIN + H_BOX/2, 'Converted\naudio',
        ha='center', va='center', fontsize=8,
        fontweight='bold', color='#333')

# Main flow arrows (horizontal)
arrow(ax, 1.1, Y_MAIN + H_BOX/2, 1.4, Y_MAIN + H_BOX/2)
arrow(ax, 3.35, Y_MAIN + H_BOX/2, 4.1, Y_MAIN + H_BOX/2)
arrow(ax, 6.15, Y_MAIN + H_BOX/2, 6.8, Y_MAIN + H_BOX/2)
arrow(ax, 9.2, Y_MAIN + H_BOX/2, 9.9, Y_MAIN + H_BOX/2)
arrow(ax, 11.95, Y_MAIN + H_BOX/2, 12.65, Y_MAIN + H_BOX/2)

# Side branch 1 (top): Confidence gating feedback on CRT
rounded_box(ax, 6.8, 5.0, 2.4, 0.95,
            'Confidence gating  τ=0.4', *C_GATE, fontsize=7)
# Vertical arrow from CRT up to gate and back
arrow(ax, 8.0, Y_MAIN + H_BOX + 0.15, 8.0, 5.0,
      color='#AF8B00', lw=1.1)
ax.text(8.15, 4.85, 'logits', ha='left', va='top',
        fontsize=6, style='italic', color='#AF8B00')

# Side branch 2 (bottom): Target speaker reference feeding both retrieval and CRT
rounded_box(ax, 4.5, 0.5, 3.2, 1.1,
            'Target speaker reference\n→ FAISS index + embedding',
            *C_COND, fontsize=7, bold=False)

# Arrows from target reference up to retrieval and CRT
arrow(ax, 5.3, 1.6, 5.3, Y_MAIN,
      color='#C62828', lw=1.0)
arrow(ax, 7.2, 1.6, 7.8, Y_MAIN - 0.15,
      color='#C62828', lw=1.0)

# Stage labels at the top
ax.text(2.35, 6.3, 'Stage 1: Encode', ha='center', va='center',
        fontsize=7, style='italic', color='#555')
ax.text(5.1, 6.3, 'Stage 2: Retrieve', ha='center', va='center',
        fontsize=7, style='italic', color='#555')
ax.text(8.0, 6.3, 'Stage 3: Refine', ha='center', va='center',
        fontsize=7, style='italic', color='#2E7D32', fontweight='bold')
ax.text(10.9, 6.3, 'Stage 4: Decode', ha='center', va='center',
        fontsize=7, style='italic', color='#555')

# Highlight bracket for the CRT stage (our contribution)
from matplotlib.patches import FancyArrow
ax.plot([6.75, 9.25], [6.05, 6.05], '-', color='#2E7D32', lw=1.4, zorder=1)
ax.plot([6.75, 6.75], [6.0, 6.1], '-', color='#2E7D32', lw=1.4, zorder=1)
ax.plot([9.25, 9.25], [6.0, 6.1], '-', color='#2E7D32', lw=1.4, zorder=1)

plt.tight_layout()
fig.savefig(f'{outdir}/fig4_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved fig4_architecture.png')

print('All charts generated.')
