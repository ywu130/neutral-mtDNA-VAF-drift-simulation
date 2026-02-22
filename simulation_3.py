#!/usr/bin/env python3
"""
Neutral-selection (genetic drift) simulation for mtDNA heteroplasmy (VAF) across generations.

This version is customized to produce a publication-friendly "Panel A" layout:
  - A 2x3 grid:
      Top row: trajectories across generations (scatter clouds)
      Bottom row: final-generation VAF distributions (histograms)
  - Exactly three initial VAFs by default: 0.03, 0.15, 0.45
  - Optional dashed horizontal line for a functional threshold (e.g., 0.6 or 0.8)

Model (per cell, per generation):
  - Each cell has N mtDNA copies.
  - If current mutant fraction is p_t, then:
        k_{t+1} ~ Binomial(N, p_t)
    so p_{t+1} = k_{t+1} / N

Example:
  python neutral_mtdna_drift_panelA.py --cells 1000 --copies 1000 --gens 13 --seed 0 --save panelA.png
  python neutral_mtdna_drift_panelA.py --threshold 0.7 --save panelA_thr07.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def simulate_neutral_drift(
    n_cells: int,
    n_copies: int,
    n_gens: int,
    init_vaf: float,
    rng: np.random.Generator,
    randomize_init: bool = False,
) -> np.ndarray:
    """
    Returns vaf matrix of shape (n_gens+1, n_cells).
    vaf[t, i] is mutant fraction in cell i at generation t.
    """
    if not (0.0 <= init_vaf <= 1.0):
        raise ValueError(f"init_vaf must be in [0,1], got {init_vaf}")
    if n_cells <= 0 or n_copies <= 0 or n_gens < 0:
        raise ValueError("cells/copies must be >0 and gens must be >=0")

    vaf = np.empty((n_gens + 1, n_cells), dtype=np.float64)

    if randomize_init:
        k0 = rng.binomial(n_copies, init_vaf, size=n_cells)
    else:
        k0 = np.full(n_cells, int(round(init_vaf * n_copies)), dtype=np.int64)

    vaf[0, :] = k0 / n_copies

    k = k0.copy()
    for t in range(1, n_gens + 1):
        p = k / n_copies
        k = rng.binomial(n_copies, p)  # vectorized across cells
        vaf[t, :] = k / n_copies

    return vaf


def make_panel_A_2x3(
    init_vafs,
    vaf_mats,
    bins: int,
    threshold: float | None,
    show_kde: bool,
    seed_for_jitter: int,
):
    """
    Creates a single 2x3 figure:
      Row 1: trajectories (scatter clouds per generation)
      Row 2: final histograms
    """
    assert len(init_vafs) == 3, "This layout expects exactly 3 initial VAFs."

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), constrained_layout=True)

    # make jitter reproducible and independent from simulation RNG
    jitter_rng = np.random.default_rng(seed_for_jitter)

    for j, init in enumerate(init_vafs):
        vaf = vaf_mats[j]  # (T+1, cells)
        T = vaf.shape[0] - 1
        n_cells = vaf.shape[1]

        # ---------- Top row: trajectories ----------
        ax = axes[0, j]

        # Color by generation using a colormap; no hardcoded colors.
        for t in range(T + 1):
            x = np.full(n_cells, t, dtype=float)
            x += (jitter_rng.random(n_cells) - 0.5) * 0.12  # small horizontal jitter
            ax.scatter(
                x,
                vaf[t],
                s=6,
                alpha=0.65,
                c=np.full(n_cells, t),
                cmap="viridis",
                vmin=0,
                vmax=T,
                edgecolors="none",
            )

        if threshold is not None:
            ax.axhline(threshold, linestyle="--", linewidth=1.2)

        ax.set_title(f"Initial VAF = {init:.2f}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Cell VAF")
        ax.set_xlim(-0.6, T + 0.6)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(0, T + 1))

        # ---------- Bottom row: final distribution ----------
        ax2 = axes[1, j]
        final = vaf[-1]

        ax2.hist(final, bins=bins, density=True, alpha=0.75)

        if show_kde:
            # Simple Gaussian overlay (not a true KDE; keeps dependencies minimal)
            mu = float(np.mean(final))
            sigma = float(np.std(final) + 1e-12)
            xs = np.linspace(0, 1, 400)
            ys = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
            ax2.plot(xs, ys, linewidth=1.6)

        if threshold is not None:
            ax2.axvline(threshold, linestyle="--", linewidth=1.2)

        #ax2.set_xlabel("Final VAF (Gen T)")
        ax2.set_xlabel(f"Final VAF (Generation {T})")
        ax2.set_ylabel("Density")
        ax2.set_xlim(0.0, 1.0)

    # Panel label "A" for the whole block (useful when you combine with an illustration as Panel B)
    #fig.text(0.01, 0.99, "A", fontsize=18, fontweight="bold", va="top")

    return fig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=int, default=1000, help="number of cells to simulate")
    p.add_argument("--copies", type=int, default=1000, help="mtDNA copies per cell (N)")
    p.add_argument("--gens", type=int, default=13, help="number of generations")
    p.add_argument(
        "--init",
        type=float,
        nargs="*",
        default=[0.03, 0.15, 0.45],
        help="exactly 3 initial VAFs for the 2x3 layout (default: 0.03 0.15 0.45)",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed for simulation")
    p.add_argument("--jitter-seed", type=int, default=123, help="random seed for plotting jitter")
    p.add_argument("--bins", type=int, default=40, help="bins for histograms")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="optional functional-threshold VAF (e.g., 0.6 or 0.8); draws dashed line",
    )
    p.add_argument(
        "--randomize-init",
        action="store_true",
        help="if set, initial mutant counts are drawn as Binomial(N, init_vaf) per cell",
    )
    p.add_argument(
        "--no-kde",
        action="store_true",
        help="if set, do not overlay Gaussian curve on histograms",
    )
    p.add_argument("--save", type=str, default="", help="path to save a PNG/PDF (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    init_vafs = list(args.init)
    if len(init_vafs) != 3:
        raise ValueError(
            f"This script expects exactly 3 initial VAFs for the 2x3 layout; got {len(init_vafs)}: {init_vafs}"
        )

    rng = np.random.default_rng(args.seed)

    vaf_mats = []
    for init in init_vafs:
        vaf_mats.append(
            simulate_neutral_drift(
                n_cells=args.cells,
                n_copies=args.copies,
                n_gens=args.gens,
                init_vaf=init,
                rng=rng,
                randomize_init=args.randomize_init,
            )
        )

    fig = make_panel_A_2x3(
        init_vafs=init_vafs,
        vaf_mats=vaf_mats,
        bins=args.bins,
        threshold=args.threshold,
        show_kde=not args.no_kde,
        seed_for_jitter=args.jitter_seed,
    )

    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()