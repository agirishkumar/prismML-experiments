#!/usr/bin/env python3
"""Static PNG charts from results/ for the README and social posts (light theme)."""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib.common import RESULTS  # noqa: E402

SURFACE, GRID, INK, INK2 = "#fcfcfb", "#e8e7e3", "#0b0b0b", "#52514e"
FAMILY = {"bonsai2": ("Bonsai 2 (ternary)", "#2a78d6"), "qwen3.5": ("Qwen3.5 (Q8_0)", "#eb6834")}
MODELS = [("bonsai2-2b", "bonsai2", "Bonsai 2B"), ("bonsai2-4b", "bonsai2", "Bonsai 4B"),
          ("bonsai2-9b", "bonsai2", "Bonsai 9B"), ("qwen35-2b", "qwen3.5", "Qwen3.5 2B"),
          ("qwen35-4b", "qwen3.5", "Qwen3.5 4B")]
FOOT = "RTX 3060 Laptop 6 GB · Prism llama.cpp b10777 · greedy, thinking off · IFEval n=541"


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _data():
    rows = []
    for mid, fam, label in MODELS:
        e1 = {(r["ctx"], r["kv_type"]): r for r in csv.DictReader(open(RESULTS / mid / "e1_memory.csv"))}
        ife = json.loads((RESULTS / mid / "ifeval_summary.json").read_text())
        rows.append({"label": label, "fam": fam, "vram_gib": int(e1[("4096", "f16")]["proc_vram_mib"]) / 1024,
                     "ifeval": ife["strict_prompt"] * 100, "j_correct": ife["energy"]["j_net_per_correct"]})
    return rows


def quality_vs_memory(rows):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    _style(ax, fig)
    for fam, (name, color) in FAMILY.items():
        pts = [r for r in rows if r["fam"] == fam]
        ax.scatter([r["vram_gib"] for r in pts], [r["ifeval"] for r in pts], s=110, color=color,
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=name)
    offsets = {"Bonsai 2B": (8, -14), "Bonsai 4B": (8, 6), "Bonsai 9B": (8, 6),
               "Qwen3.5 2B": (8, -14), "Qwen3.5 4B": (-8, 8)}
    for r in rows:
        dx, dy = offsets[r["label"]]
        ax.annotate(f"{r['label']}  {r['ifeval']:.1f}%", (r["vram_gib"], r["ifeval"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10, color=INK, ha="right" if dx < 0 else "left")
    ax.set_xlim(0, 5)
    ax.set_ylim(60, 90)
    ax.set_xlabel("GPU memory at 4K context (GiB)", color=INK2, fontsize=10)
    ax.set_ylabel("IFEval strict prompt accuracy (%)", color=INK2, fontsize=10)
    ax.set_title("Instruction following vs memory footprint", loc="left", color=INK, fontsize=14, pad=26)
    ax.text(0, 1.03, "A ternary 4B in 1.4 GiB beats an 8-bit 2B in 2.1 GiB", transform=ax.transAxes,
            color=INK2, fontsize=10)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.01, 0.01, FOOT, color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "chart_quality_vs_memory.png", facecolor=SURFACE)


def energy_per_correct(rows):
    rows = sorted(rows, key=lambda r: r["j_correct"])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    _style(ax, fig)
    ax.grid(axis="y", visible=False)
    h = 0.42  # ≈ 22px at this size; leaves air in each band
    for i, r in enumerate(rows):
        color = FAMILY[r["fam"]][1]
        ax.barh(i, r["j_correct"], height=h, color=color, zorder=3)
        ax.text(r["j_correct"] + 8, i, f"{r['j_correct']:.0f} J", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], color=INK)
    ax.set_xlim(0, max(r["j_correct"] for r in rows) * 1.15)
    ax.set_xlabel("GPU energy per correct IFEval answer (joules above idle)", color=INK2, fontsize=10)
    ax.set_title("Energy per correct answer", loc="left", color=INK, fontsize=14, pad=26)
    ax.text(0, 1.04, "Lower is better · GPU joules above idle, divided by prompts that pass every instruction",
            transform=ax.transAxes, color=INK2, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in FAMILY.values()]
    leg = ax.legend(handles, [n for n, _ in FAMILY.values()], loc="lower right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.01, 0.01, FOOT, color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "chart_energy_per_correct.png", facecolor=SURFACE)


if __name__ == "__main__":
    data = _data()
    quality_vs_memory(data)
    energy_per_correct(data)
    print("wrote", RESULTS / "chart_quality_vs_memory.png", RESULTS / "chart_energy_per_correct.png")
