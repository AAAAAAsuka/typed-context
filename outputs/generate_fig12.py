"""
Generate Figure 12: Conceptual diagram comparing Standard Transformer vs Typed Context Transformer.

Produces: fig12_conceptual.png

This script uses only matplotlib and numpy (no external dependencies).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np


def draw_rounded_box(ax, xy, width, height, text, facecolor, edgecolor="black",
                     fontsize=9, text_color="white", lw=1.5, alpha=1.0, fontstyle="normal",
                     fontweight="normal", zorder=2):
    """Draw a rounded rectangle with centered text."""
    box = mpatches.FancyBboxPatch(
        xy, width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.05),
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=fontweight, fontstyle=fontstyle, zorder=zorder + 1)
    return box


def draw_arrow(ax, start, end, color="black", lw=1.5, style="->", connectionstyle="arc3,rad=0.0",
               zorder=1):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=zorder,
    )


def draw_standard_transformer(ax):
    """Draw the LEFT panel: Standard Transformer."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel title
    ax.text(5, 11.5, "Standard Transformer", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333")

    # --- Input tokens row ---
    ax.text(5, 10.3, "Input Tokens", ha="center", va="center",
            fontsize=10, fontstyle="italic", color="#555555")

    # System tokens (blue)
    sys_color = "#3B82F6"
    draw_rounded_box(ax, (0.8, 9.0), 3.5, 0.8, "System Tokens", facecolor=sys_color,
                     fontsize=9, fontweight="bold")

    # "+" sign
    ax.text(4.95, 9.4, "+", ha="center", va="center", fontsize=16, fontweight="bold",
            color="#333333")

    # User tokens (red)
    usr_color = "#EF4444"
    draw_rounded_box(ax, (5.6, 9.0), 3.5, 0.8, "User Tokens", facecolor=usr_color,
                     fontsize=9, fontweight="bold")

    # Note: all tokens treated equally
    ax.text(5, 8.45, "(all tokens treated equally regardless of source)",
            ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#888888")

    # Arrow down to encoding
    draw_arrow(ax, (5, 8.2), (5, 7.65), color="#555555", lw=2, style="-|>")

    # Position Encoding box
    enc_color = "#8B5CF6"
    draw_rounded_box(ax, (1.5, 6.8), 7, 0.8, "Position Encoding (RoPE only)",
                     facecolor=enc_color, fontsize=10, fontweight="bold")

    # Arrow down to attention
    draw_arrow(ax, (5, 6.7), (5, 6.15), color="#555555", lw=2, style="-|>")

    # Attention block
    attn_color = "#F59E0B"
    draw_rounded_box(ax, (1.5, 5.0), 7, 1.1, "Self-Attention\n(no type awareness)",
                     facecolor=attn_color, fontsize=10, fontweight="bold", text_color="white")

    # Arrow down to output
    draw_arrow(ax, (5, 4.9), (5, 4.35), color="#555555", lw=2, style="-|>")

    # Output box
    out_color = "#10B981"
    draw_rounded_box(ax, (2.0, 3.4), 6, 0.8, "Output", facecolor=out_color,
                     fontsize=11, fontweight="bold")

    # Limitation callout
    draw_rounded_box(ax, (1.0, 1.3), 8, 1.6,
                     "",
                     facecolor="#FEF3C7", edgecolor="#F59E0B", text_color="#92400E",
                     fontsize=8, lw=2, alpha=0.9)
    ax.text(5, 2.35, "Limitation", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#92400E")
    ax.text(5, 1.85, "Attention treats all tokens identically.", ha="center", va="center",
            fontsize=8, color="#92400E")
    ax.text(5, 1.5, "No mechanism to distinguish trusted vs. untrusted.", ha="center", va="center",
            fontsize=8, color="#92400E")

    # Small icons showing homogeneous attention
    for i, x_pos in enumerate([2.5, 4.0, 5.5, 7.0]):
        c = sys_color if i < 2 else usr_color
        circle = plt.Circle((x_pos, 7.95), 0.15, color=c, alpha=0.5, zorder=3)
        ax.add_patch(circle)
        draw_arrow(ax, (x_pos, 7.78), (x_pos, 7.65), color=c, lw=1.0, style="-|>")


def draw_typed_context_transformer(ax):
    """Draw the RIGHT panel: Typed Context Transformer."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel title
    ax.text(5, 11.5, "Typed Context Transformer", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333")

    # --- Input tokens row ---
    ax.text(5, 10.3, "Input Tokens (with Type Tags)", ha="center", va="center",
            fontsize=10, fontstyle="italic", color="#555555")

    # System tokens (blue) with type label
    sys_color = "#3B82F6"
    draw_rounded_box(ax, (0.5, 9.0), 3.8, 0.8, "System Tokens", facecolor=sys_color,
                     fontsize=9, fontweight="bold")
    # Type badge
    draw_rounded_box(ax, (1.3, 8.5), 1.8, 0.45, "type = 0", facecolor="#1D4ED8",
                     edgecolor="#1E40AF", fontsize=7.5, fontweight="bold", lw=1)

    # "+" sign
    ax.text(4.95, 9.4, "+", ha="center", va="center", fontsize=16, fontweight="bold",
            color="#333333")

    # User tokens (red) with type label
    usr_color = "#EF4444"
    draw_rounded_box(ax, (5.6, 9.0), 3.8, 0.8, "User Tokens", facecolor=usr_color,
                     fontsize=9, fontweight="bold")
    # Type badge
    draw_rounded_box(ax, (6.5, 8.5), 1.8, 0.45, "type = 1", facecolor="#B91C1C",
                     edgecolor="#991B1B", fontsize=7.5, fontweight="bold", lw=1)

    # Arrow down to encoding
    draw_arrow(ax, (5, 8.2), (5, 7.65), color="#555555", lw=2, style="-|>")

    # Position + Type Encoding box (split visual)
    enc_color_pos = "#8B5CF6"
    enc_color_type = "#EC4899"

    # Combined encoding box background
    draw_rounded_box(ax, (0.8, 6.55), 8.4, 1.05, "",
                     facecolor="#F3E8FF", edgecolor="#7C3AED", fontsize=10,
                     fontweight="bold", lw=2, text_color="#333333")

    # Position RoPE sub-box
    draw_rounded_box(ax, (1.0, 6.7), 3.6, 0.7, "Position RoPE", facecolor=enc_color_pos,
                     fontsize=9, fontweight="bold")

    # "+" in encoding
    ax.text(5.0, 7.05, "+", ha="center", va="center", fontsize=14, fontweight="bold",
            color="#7C3AED")

    # Type rotation sub-box
    draw_rounded_box(ax, (5.4, 6.7), 3.6, 0.7, "Type Rotation", facecolor=enc_color_type,
                     fontsize=9, fontweight="bold")

    # Small annotation
    ax.text(5, 6.35, "rotation applied in dedicated subspace dimensions",
            ha="center", va="center", fontsize=7, fontstyle="italic", color="#7C3AED")

    # Arrow down to attention
    draw_arrow(ax, (5, 6.25), (5, 5.7), color="#555555", lw=2, style="-|>")

    # Attention block (type-aware)
    attn_color = "#F59E0B"
    draw_rounded_box(ax, (1.0, 4.55), 8, 1.1, "Self-Attention\n(type-aware via typed RoPE)",
                     facecolor=attn_color, fontsize=10, fontweight="bold", text_color="white")

    # Arrow down to output
    draw_arrow(ax, (5, 4.45), (5, 3.9), color="#555555", lw=2, style="-|>")

    # Output box
    out_color = "#10B981"
    draw_rounded_box(ax, (2.0, 3.0), 6, 0.8, "Type-Aware Output", facecolor=out_color,
                     fontsize=11, fontweight="bold")

    # Advantage callout
    draw_rounded_box(ax, (0.6, 1.0), 8.8, 1.6,
                     "",
                     facecolor="#D1FAE5", edgecolor="#10B981", text_color="#065F46",
                     fontsize=8, lw=2, alpha=0.9)
    ax.text(5, 2.05, "Advantage", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#065F46")
    ax.text(5, 1.55, "Attention natively distinguishes token types.", ha="center", va="center",
            fontsize=8, color="#065F46")
    ax.text(5, 1.2, "Persistent rotation encodes provenance in geometry.", ha="center", va="center",
            fontsize=8, color="#065F46")

    # Colored circles showing type-differentiated attention
    for i, x_pos in enumerate([2.2, 3.7, 6.0, 7.5]):
        c = sys_color if i < 2 else usr_color
        circle = plt.Circle((x_pos, 7.95), 0.15, color=c, alpha=0.7, zorder=3)
        ax.add_patch(circle)
        # Arrows with matching color to show type-awareness
        draw_arrow(ax, (x_pos, 7.78), (x_pos, 7.65), color=c, lw=1.2, style="-|>")


def main():
    fig = plt.figure(figsize=(18, 10), facecolor="white")

    # Suptitle
    fig.suptitle("Typed Context via Persistent Rotation",
                 fontsize=20, fontweight="bold", color="#1E293B", y=0.97)

    # Subtitle
    fig.text(0.5, 0.935,
             "How typed RoPE augments standard positional encoding to make attention type-aware",
             ha="center", fontsize=12, fontstyle="italic", color="#64748B")

    # Create two side-by-side subplots
    ax_left = fig.add_axes([0.02, 0.04, 0.46, 0.85])
    ax_right = fig.add_axes([0.52, 0.04, 0.46, 0.85])

    # Draw vertical separator
    sep_ax = fig.add_axes([0.485, 0.06, 0.03, 0.83])
    sep_ax.set_xlim(0, 1)
    sep_ax.set_ylim(0, 1)
    sep_ax.axis("off")
    sep_ax.plot([0.5, 0.5], [0.0, 1.0], color="#CBD5E1", lw=2, ls="--", zorder=0)
    sep_ax.text(0.5, 0.5, "vs", ha="center", va="center",
                fontsize=16, fontweight="bold", color="#94A3B8",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", lw=1.5))

    # Draw the two panels
    draw_standard_transformer(ax_left)
    draw_typed_context_transformer(ax_right)

    # Legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor="#3B82F6", edgecolor="black", label="System / Trusted Tokens (type=0)"),
        mpatches.Patch(facecolor="#EF4444", edgecolor="black", label="User / Untrusted Tokens (type=1)"),
        mpatches.Patch(facecolor="#8B5CF6", edgecolor="black", label="Position Encoding (RoPE)"),
        mpatches.Patch(facecolor="#EC4899", edgecolor="black", label="Type Rotation (Typed RoPE)"),
        mpatches.Patch(facecolor="#F59E0B", edgecolor="black", label="Self-Attention"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
               frameon=True, fancybox=True, shadow=False, edgecolor="#CBD5E1",
               bbox_to_anchor=(0.5, 0.005))

    # Save
    output_path = "/data/bochuan/typed_context/outputs/fig12_conceptual.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
