"""Generate clean confusion matrix and training curves for the technical report."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

matplotlib.use('Agg')

# ── Data from actual training output ──
epochs = list(range(1, 11))
loss =     [0.1471, 0.1006, 0.0873, 0.0729, 0.0547, 0.0417, 0.0320, 0.0282, 0.0227, 0.0176]
top1_acc = [0.978,  0.970,  0.983,  0.979,  0.980,  0.982,  0.979,  0.985,  0.988,  0.991]

# Confusion matrix values (from YOLO output, without background)
#                    Predicted
#                  e-waste   non-ewaste
# True e-waste       372        7
# True non-ewaste      0      379
cm = np.array([[372,   7],
               [  0, 379]])

cm_norm = np.array([[0.98, 0.02],
                    [0.00, 1.00]])

classes = ['e-waste', 'non-ewaste']

# LR schedule
lr0 = 1e-3
lrf = 0.01
warmup_epochs = 3
lr_schedule = []
for e in epochs:
    if e <= warmup_epochs:
        lr = lr0 * (e / warmup_epochs)
    else:
        progress = (e - warmup_epochs) / (10 - warmup_epochs)
        lr = lr0 * (lrf + 0.5 * (1 - lrf) * (1 + math.cos(math.pi * progress)))
    lr_schedule.append(lr)

# ── Style ──
colors = {
    'loss': '#ef4444',
    'acc': '#10b981',
    'lr': '#6c63ff',
    'bg': '#0f1117',
    'surface': '#1a1b26',
    'grid': '#2a2a3a',
    'text': '#e4e4ed',
    'subtext': '#8888a0',
}

def style_ax(ax):
    ax.set_facecolor(colors['surface'])
    ax.grid(True, alpha=0.15, color=colors['grid'])
    ax.tick_params(colors=colors['subtext'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(colors['grid'])
        spine.set_linewidth(0.5)

# ═══════════════════════════════════════════════════════
# FIGURE 1: Training Curves (3 panels)
# ═══════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig1.patch.set_facecolor(colors['bg'])

for ax in axes:
    style_ax(ax)

# ── Loss ──
ax1 = axes[0]
ax1.plot(epochs, loss, color=colors['loss'], linewidth=2.5, marker='o', markersize=7,
         markerfacecolor='white', markeredgecolor=colors['loss'], markeredgewidth=2)
ax1.fill_between(epochs, loss, alpha=0.12, color=colors['loss'])
ax1.set_xlabel('Epoch', color=colors['text'], fontsize=11)
ax1.set_ylabel('Loss', color=colors['text'], fontsize=11)
ax1.set_title('Training Loss', color=colors['text'], fontsize=14, fontweight='bold', pad=12)
ax1.set_xticks(epochs)
for i, v in enumerate(loss):
    ax1.annotate(f'{v:.4f}', (epochs[i], loss[i]), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=7.5, color='#fca5a5')

# ── Accuracy ──
ax2 = axes[1]
acc_pct = [a * 100 for a in top1_acc]
ax2.plot(epochs, acc_pct, color=colors['acc'], linewidth=2.5, marker='s', markersize=7,
         markerfacecolor='white', markeredgecolor=colors['acc'], markeredgewidth=2)
ax2.fill_between(epochs, acc_pct, alpha=0.12, color=colors['acc'])
ax2.set_xlabel('Epoch', color=colors['text'], fontsize=11)
ax2.set_ylabel('Accuracy (%)', color=colors['text'], fontsize=11)
ax2.set_title('Top-1 Validation Accuracy', color=colors['text'], fontsize=14, fontweight='bold', pad=12)
ax2.set_xticks(epochs)
ax2.set_ylim(96, 100)
for i, v in enumerate(acc_pct):
    ax2.annotate(f'{v:.1f}%', (epochs[i], acc_pct[i]), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=7.5, color='#6ee7b7')

# ── LR ──
ax3 = axes[2]
lr_display = [lr * 1000 for lr in lr_schedule]
ax3.plot(epochs, lr_display, color=colors['lr'], linewidth=2.5, marker='D', markersize=7,
         markerfacecolor='white', markeredgecolor=colors['lr'], markeredgewidth=2)
ax3.fill_between(epochs, lr_display, alpha=0.12, color=colors['lr'])
ax3.set_xlabel('Epoch', color=colors['text'], fontsize=11)
ax3.set_ylabel('Learning Rate (×10⁻³)', color=colors['text'], fontsize=11)
ax3.set_title('Learning Rate Schedule', color=colors['text'], fontsize=14, fontweight='bold', pad=12)
ax3.set_xticks(epochs)

plt.tight_layout(pad=2.0)
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight', facecolor=colors['bg'])
plt.close()
print("✅ Saved: training_curves.png")

# ═══════════════════════════════════════════════════════
# FIGURE 2: Confusion Matrix (2 panels — counts + normalized)
# ═══════════════════════════════════════════════════════
fig2, (ax_cm, ax_norm) = plt.subplots(1, 2, figsize=(14, 5.5))
fig2.patch.set_facecolor(colors['bg'])

for ax, data, title, fmt, cmap_name in [
    (ax_cm,   cm,      'Confusion Matrix (Counts)',     'd',   'Blues'),
    (ax_norm, cm_norm, 'Confusion Matrix (Normalized)', '.2f', 'Blues'),
]:
    ax.set_facecolor(colors['surface'])

    # Custom colormap
    im = ax.imshow(data, interpolation='nearest', cmap=cmap_name, aspect='auto')

    # Add text annotations
    thresh = data.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = format(data[i, j], fmt)
            text_color = 'white' if data[i, j] > thresh else '#1a1b26'
            ax.text(j, i, val, ha='center', va='center',
                    fontsize=20, fontweight='bold', color=text_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, color=colors['text'])
    ax.set_yticklabels(classes, fontsize=11, color=colors['text'])
    ax.set_xlabel('Predicted', fontsize=12, color=colors['text'], labelpad=10)
    ax.set_ylabel('Actual', fontsize=12, color=colors['text'], labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', color=colors['text'], pad=12)
    ax.tick_params(colors=colors['subtext'])

    for spine in ax.spines.values():
        spine.set_color(colors['grid'])

    cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=colors['subtext'])

plt.tight_layout(pad=2.5)
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor=colors['bg'])
plt.close()
print("✅ Saved: confusion_matrix.png")

# ── Print metrics ──
TP = cm[0, 0]  # 372
FN = cm[0, 1]  # 7
FP = cm[1, 0]  # 0
TN = cm[1, 1]  # 379

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"\n📊 Classification Metrics:")
print(f"   TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   Accuracy:  {accuracy:.4f}")
