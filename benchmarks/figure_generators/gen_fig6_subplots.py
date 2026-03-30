"""Generate individual subplot PNGs for Figure 6 (Biological Validation — Heatmaps).

Produces:
  Panel A — Workflow
  Panel B — Gene importance heatmaps (datasets × models)

GO enrichment dot plots are moved to Figure 9.

Output: benchmarks/paper_figures/{series}/subplots/fig6/

Usage:
    python -m benchmarks.figure_generators.gen_fig6_subplots --series dpmm
"""

import argparse
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.figure_generators.common import (
    BIO_RESULTS,
    MODEL_SHORT_NAMES,
    PRIOR_MODELS_DPMM,
    REPRESENTATIVE_DATASETS,
    compute_umap,
)
from benchmarks.figure_generators.data_loaders import load_cross_latent
from benchmarks.figure_generators.subplot_style import (
    FIGSIZE_ENRICHMENT,
    FIGSIZE_HEATMAP,
    FIGSIZE_UMAP,
    FONTSIZE_ANNOTATION,
    FONTSIZE_TICK,
    FONTSIZE_TITLE,
    LINE_WIDTH_SPINE,
    SCATTER_SIZE_UMAP,
    apply_subplot_style,
    build_manifest,
)
from src.visualization import bind_figure_region, save_with_vcd, style_axes

_COMP_PALETTE = [
    "#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1",
    "#9C755F", "#BAB0AC", "#86BCB6", "#A0CBE8", "#CFCFCF",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Bio data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_bio_paths(model, dataset, results_dir):
    tag = f"{model}_{dataset}"
    candidates = []
    model_dir = results_dir / model.replace("/", "_")
    if model_dir.is_dir():
        candidates.append(model_dir)
    candidates.append(results_dir)
    for d in candidates:
        lp = d / f"{tag}_latent_data.npz"
        ip = d / f"{tag}_importance.npz"
        if lp.exists() and ip.exists():
            return lp, ip, d
    return None, None, None


def _load_dataset_bio(models, dataset, results_dir):
    model_data = {}
    for model in models:
        latent_path, importance_path, bio_dir = _resolve_bio_paths(
            model, dataset, results_dir)
        tag = f"{model}_{dataset}"
        if latent_path is None:
            continue
        try:
            ld = np.load(latent_path, allow_pickle=True)
            imp = np.load(importance_path, allow_pickle=True)
            latent = ld["latent"]
            components = ld.get("components")
            importance = imp["importance"]
            gene_names = imp.get("gene_names")
            K_total = (components.shape[1] if components is not None
                       else importance.shape[0])
            enrichments = {}
            for k in range(K_total):
                csv_path = bio_dir / f"{tag}_enrichment_comp{k}.csv"
                if csv_path.exists():
                    edf = pd.read_csv(csv_path)
                    if len(edf) > 0:
                        enrichments[k] = edf
            model_data[model] = {
                "latent": latent, "components": components,
                "importance": importance, "gene_names": gene_names,
                "enrichments": enrichments,
            }
        except Exception:
            pass
    return model_data if model_data else None


def _discover_bio_datasets(models, results_dir):
    found = {}
    for ds in REPRESENTATIVE_DATASETS:
        md = _load_dataset_bio(models, ds, results_dir)
        if md:
            found[ds] = md
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Subplot generators
# ═══════════════════════════════════════════════════════════════════════════════

def gen_bio_umap(model_name, datasets, out_path, show_legend=False):
    """Generate one cross-dataset UMAP subplot for bio models."""
    import matplotlib as mpl
    blocks, ds_labels = [], []
    for ds in datasets:
        lat = load_cross_latent(model_name, ds)
        if lat is None or len(lat) == 0:
            continue
        blocks.append(lat)
        ds_labels.extend([ds] * len(lat))
    if not blocks:
        fig = plt.figure(figsize=FIGSIZE_UMAP)
        layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
        ax = layout.add_axes(fig)
        style_axes(ax)
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=FONTSIZE_TITLE)
        save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)
        return
    X = np.vstack(blocks)
    emb = compute_umap(X)
    ds_labels = np.array(ds_labels)
    cmap = mpl.colormaps.get("tab20", mpl.colormaps["tab20"])
    color_map = {ds: cmap(i / max(len(datasets) - 1, 1))
                 for i, ds in enumerate(datasets)}

    fig = plt.figure(figsize=FIGSIZE_UMAP)

    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))

    ax = layout.add_axes(fig)

    style_axes(ax)
    for ds in datasets:
        mask = ds_labels == ds
        if not np.any(mask):
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=SCATTER_SIZE_UMAP,
                   alpha=0.60, color=[color_map.get(ds, "gray")],
                   label=ds, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    short = MODEL_SHORT_NAMES.get(model_name, model_name)
    ax.set_title(short, fontsize=FONTSIZE_TITLE, pad=2,
                 loc="left", fontweight="normal")
    for sp in ax.spines.values():
        sp.set_linewidth(LINE_WIDTH_SPINE)
    if show_legend:
        ax.legend(loc="lower right", fontsize=FONTSIZE_TITLE,
                  framealpha=0.85, markerscale=1.2, handletextpad=0.1,
                  borderpad=0.3, labelspacing=0.2, ncol=2,
                  columnspacing=0.5)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_heatmap(md, comp_prefix, ds_name, model, out_path, top_genes=30):
    """Generate one gene importance heatmap subplot PNG.

    The raw perturbation importance is z-scored column-wise (across
    components per gene) so that universally important genes get z≈0
    while component-specific genes stand out.  A diverging
    colormap (``RdBu_r``) shows positive (component-specific) vs. negative
    (suppressed) z-scores.

    Gene selection: top genes are chosen based on the *maximum z-score*
    across components, which naturally favours component-specific genes.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    importance = md["importance"]
    gene_names = md["gene_names"]
    n_comp = min(importance.shape[0], 10)

    # Column-wise z-score: for each gene, subtract mean across components
    # and divide by std.  Genes equally important in all components → z ≈ 0.
    imp = importance[:n_comp, :]
    mu = imp.mean(axis=0, keepdims=True)
    sd = imp.std(axis=0, keepdims=True) + 1e-12
    imp_z = (imp - mu) / sd

    # Select top genes by max z-score (component-specific importance)
    top_idx = set()
    for k in range(n_comp):
        top_idx.update(np.argsort(imp_z[k])[::-1][:top_genes])
    # First rank by global max to get the top candidates
    top_idx_ranked = sorted(top_idx,
                            key=lambda i: -imp_z.max(axis=0)[i])[:top_genes]
    # Re-sort by dominant component (argmax) then by magnitude within that
    # component → creates a block-diagonal pattern that directly reveals
    # which genes characterise each component.
    def _sort_key_imp(i):
        dom = int(np.argmax(imp_z[:, i]))
        return (dom, -imp_z[dom, i])
    top_idx = sorted(top_idx_ranked, key=_sort_key_imp)
    sub_imp = imp_z[:, top_idx]
    sub_genes = ([str(gene_names[i]) for i in top_idx]
                 if gene_names is not None else [f"g{i}" for i in top_idx])

    # Wider figure: use 2-column width to fit more genes
    fig_w = FIGSIZE_HEATMAP[0] * 2.5
    fig_h = FIGSIZE_HEATMAP[1] * 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)
    # Symmetric color limits centred at 0
    vlim = max(abs(sub_imp.min()), abs(sub_imp.max()), 1.0)
    im = ax.imshow(sub_imp, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-vlim, vmax=vlim)
    ax.set_yticks(range(sub_imp.shape[0]))
    ax.set_yticklabels([f"{comp_prefix}{k+1}" for k in range(sub_imp.shape[0])],
                       fontsize=FONTSIZE_TITLE + 1)
    ax.set_xticks(range(len(sub_genes)))
    # Truncate long gene names to 8 chars to reduce overlap
    display_genes = [g[:8] for g in sub_genes]
    ax.set_xticklabels(display_genes, rotation=90, ha="center",
                       fontsize=max(FONTSIZE_TICK - 1, 12))
    ax.set_title(f"{short} \u2014 {ds_name} (z-scored)",
                 fontsize=FONTSIZE_TITLE + 1, loc="left",
                 fontweight="normal")
    # Colorbar placed OUTSIDE the heatmap to avoid overlapping data
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.12)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=FONTSIZE_TICK, length=2, pad=1)
    cb.set_ticks([-vlim, 0, vlim])
    cb.set_ticklabels([f"{-vlim:.1f}", "0", f"{vlim:.1f}"])
    cb.outline.set_linewidth(0.4)
    # Check for text overlaps before saving
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_correlation_heatmap(corr, gene_names, comp_prefix, ds_name, model,
                            out_path, top_genes=30):
    """Generate a latent–gene Pearson correlation heatmap.

    Selects the top correlated genes per component (by absolute
    correlation) and displays as a diverging heatmap.  Both positive
    and negative correlations are informative.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    n_comp = min(corr.shape[0], 10)
    corr_sub = corr[:n_comp, :]

    # Select top genes per component by |r|
    top_idx = set()
    for k in range(n_comp):
        top_idx.update(np.argsort(np.abs(corr_sub[k]))[::-1][:top_genes])
    # First candidate set ranked by max |r|
    top_idx_ranked = sorted(top_idx,
                            key=lambda i: -np.abs(corr_sub[:, i]).max())[:top_genes]
    # Re-sort by dominant component based on max |r|, then magnitude within
    # → block-diagonal: genes most correlated with comp-1 first, etc.
    def _sort_key_corr(i):
        abs_col = np.abs(corr_sub[:, i])
        dom = int(np.argmax(abs_col))
        return (dom, -abs_col[dom])
    top_idx = sorted(top_idx_ranked, key=_sort_key_corr)
    sub_corr = corr_sub[:, top_idx]
    sub_genes = ([str(gene_names[i]) for i in top_idx]
                 if gene_names is not None
                 else [f"g{i}" for i in top_idx])

    fig_w = FIGSIZE_HEATMAP[0] * 2.5
    fig_h = FIGSIZE_HEATMAP[1] * 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)
    vlim = max(abs(sub_corr.min()), abs(sub_corr.max()), 0.2)
    im = ax.imshow(sub_corr, aspect="auto", cmap="coolwarm",
                   interpolation="nearest", vmin=-vlim, vmax=vlim)
    ax.set_yticks(range(n_comp))
    ax.set_yticklabels([f"{comp_prefix}{k+1}" for k in range(n_comp)],
                       fontsize=FONTSIZE_TITLE + 1)
    ax.set_xticks(range(len(sub_genes)))
    # Truncate long gene names to 8 chars to reduce overlap
    display_genes = [g[:8] for g in sub_genes]
    ax.set_xticklabels(display_genes, rotation=90, ha="center",
                       fontsize=max(FONTSIZE_TICK - 1, 12))
    ax.set_title(f"{short} \u2014 {ds_name}  (latent\u2013gene corr.)",
                 fontsize=FONTSIZE_TITLE + 1, loc="left",
                 fontweight="normal")
    # Colorbar placed OUTSIDE the heatmap to avoid overlapping data
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.12)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=FONTSIZE_TICK, length=2, pad=1)
    cb.set_ticks([-vlim, 0, vlim])
    cb.set_ticklabels([f"{-vlim:.2f}", "0", f"{vlim:.2f}"])
    cb.outline.set_linewidth(0.4)
    cb.set_label("Pearson r", fontsize=FONTSIZE_ANNOTATION, labelpad=3)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_component_umap(umap_emb, latent, labels, comp_prefix, ds_name,
                       model, out_path, n_comp=4):
    """Generate UMAP panels colored by latent component intensity.

    For each of the first *n_comp* components, creates a UMAP scatter
    where each cell is colored by its value in that latent dimension.
    All components are laid out in a 2×2 grid.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    K = min(latent.shape[1], n_comp)

    fig = plt.figure(figsize=(FIGSIZE_UMAP[0] * K, FIGSIZE_UMAP[1]))

    _root = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))

    _grid = _root.grid(1, 1, row_gap=0.04, col_gap=0.04)

    axes = [[_grid[r][c].add_axes(fig) for c in range(1)] for r in range(1)]

    axes = np.array(axes).reshape(1, 1)

    if 1 == 1 and 1 > 1: axes = axes.flatten()

    for _ax in np.atleast_1d(axes).flat: style_axes(_ax)
    axes = axes.ravel()

    for k in range(K):
        ax = axes[k]
        vals = latent[:, k]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=vals, s=SCATTER_SIZE_UMAP, alpha=0.70,
                        cmap="viridis", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{comp_prefix}{k+1}", fontsize=FONTSIZE_TITLE,
                     pad=2, fontweight="normal")
        for sp in ax.spines.values():
            sp.set_linewidth(LINE_WIDTH_SPINE)
        # Mini colorbar placed outside each subplot via make_axes_locatable
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=FONTSIZE_ANNOTATION, length=1.5)
        cb.outline.set_linewidth(0.3)

    fig.suptitle(f"{short} — {ds_name}  (component intensity)",
                 fontsize=FONTSIZE_TITLE, y=1.02, fontweight="normal")
    fig.tight_layout(pad=0.5)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_gene_umap(umap_emb, top_gene_expr, top_gene_names, labels,
                  comp_prefix, ds_name, model, out_path, n_comp=4):
    """Generate UMAP panels colored by top-correlated gene expression.

    For each of the first *n_comp* components, shows a UMAP where cells
    are colored by the expression of the gene most correlated with that
    component.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    K = min(top_gene_expr.shape[1], n_comp)

    fig = plt.figure(figsize=(FIGSIZE_UMAP[0] * K, FIGSIZE_UMAP[1]))

    _root = bind_figure_region(fig, (0.05, 0.05, 0.95, 0.92))

    _grid = _root.grid(1, 1, row_gap=0.04, col_gap=0.04)

    axes = [[_grid[r][c].add_axes(fig) for c in range(1)] for r in range(1)]

    axes = np.array(axes).reshape(1, 1)

    if 1 == 1 and 1 > 1: axes = axes.flatten()

    for _ax in np.atleast_1d(axes).flat: style_axes(_ax)
    axes = axes.ravel()

    for k in range(K):
        ax = axes[k]
        expr = top_gene_expr[:, k]
        gene_name = str(top_gene_names[k])[:12]
        sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                        c=expr, s=SCATTER_SIZE_UMAP, alpha=0.70,
                        cmap="magma", rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{comp_prefix}{k+1}: {gene_name}",
                     fontsize=FONTSIZE_TITLE, pad=2, fontweight="normal")
        for sp in ax.spines.values():
            sp.set_linewidth(LINE_WIDTH_SPINE)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(labelsize=FONTSIZE_ANNOTATION, length=1.5)
        cb.outline.set_linewidth(0.3)

    fig.suptitle(f"{short} — {ds_name}  (top-correlated gene)",
                 fontsize=FONTSIZE_TITLE, y=1.02, fontweight="normal")
    fig.tight_layout(pad=0.5)
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


def gen_enrichment_dotplot(md, comp_prefix, ds_name, model, out_path,
                           top_n=10):
    """Generate one GO enrichment dot plot subplot PNG.

    Uses full-width figsize for vertical layout.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    enrichments = md.get("enrichments", {})

    all_terms = []
    for k, edf in enrichments.items():
        if "Adjusted P-value" not in edf.columns or "Term" not in edf.columns:
            continue
        for _, row in edf.head(6).iterrows():
            pval = row["Adjusted P-value"]
            if pval <= 0 or pval >= 0.05:
                continue
            term_raw = str(row["Term"])
            term_short = term_raw.split("(GO:")[0].strip()
            if len(term_short) > 38:
                term_short = term_short[:35] + "..."
            overlap_str = str(row.get("Overlap", "0/1"))
            try:
                gene_count = int(overlap_str.split("/")[0])
            except (ValueError, IndexError):
                gene_count = 1
            all_terms.append({
                "term": term_short, "comp": k,
                "nlp": -np.log10(max(pval, 1e-30)),
                "pval": pval, "gene_count": gene_count,
            })

    fig = plt.figure(figsize=FIGSIZE_ENRICHMENT)

    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))

    ax = layout.add_axes(fig)

    style_axes(ax)
    if not all_terms:
        ax.text(0.5, 0.5, "No significant terms",
                transform=ax.transAxes, ha="center",
                fontsize=FONTSIZE_TITLE)
        ax.set_title(f"{short} \u2014 {ds_name}", fontsize=FONTSIZE_TITLE,
                     loc="left", fontweight="normal")
        save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)
        return

    best = {}
    for t in all_terms:
        key = t["term"]
        if key not in best or t["nlp"] > best[key]["nlp"]:
            best[key] = t
    term_list = sorted(best.values(), key=lambda x: -x["nlp"])[:top_n]

    y_pos = np.arange(len(term_list))
    nlps = [t["nlp"] for t in term_list]
    gcs = [t["gene_count"] for t in term_list]
    comp_ids = [t["comp"] for t in term_list]
    comp_colors = [_COMP_PALETTE[c % len(_COMP_PALETTE)] for c in comp_ids]
    labels = [f"{t['term']}" for t in term_list]

    mn, mx = max(min(gcs), 1), max(gcs)
    sizes = ([20 + 120 * (g - mn) / (mx - mn) for g in gcs]
             if mx > mn else [60] * len(gcs))

    ax.scatter(nlps, y_pos, s=sizes, c=comp_colors,
               edgecolors="white", linewidth=0.4, alpha=0.88, zorder=3)
    # Use ytick labels instead of ax.text so tight_layout accounts for them
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONTSIZE_TITLE,
                       color="#333333")
    ax.tick_params(axis='y', length=0, pad=2)  # no tick marks
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()
    ax.set_xlabel("$-\\log_{10}$(adj. p)", fontsize=FONTSIZE_TITLE)
    ax.set_title(f"{short} \u2014 {ds_name}", fontsize=FONTSIZE_TITLE,
                 loc="left", fontweight="normal")
    ax.tick_params(axis='x', labelsize=FONTSIZE_TITLE)
    ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.grid(axis="x", alpha=0.2, linewidth=0.4)
    # Check for text overlaps before saving
    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate(series, out_dir):
    """Generate all subplot PNGs for Figure 6."""
    print(f"\n  Figure 6 subplots ({series})")
    sub_dir = out_dir / "fig6"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    models = PRIOR_MODELS_DPMM
    comp_prefix = "Dim"
    results_dir = BIO_RESULTS

    bio_datasets = _discover_bio_datasets(models, results_dir)
    if not bio_datasets:
        print("    No bio data found.")
        return build_manifest(sub_dir, {})

    # Panel A removed — duplicate of Figure 2 Panel A.
    avail_models = [m for m in models
                    if any(m in bio_datasets.get(ds, {}) for ds in bio_datasets)]

    # Panel B — heatmaps
    heatmap_files = {}
    for ds_name in bio_datasets:
        ds_heat = []
        for model in avail_models:
            md = bio_datasets[ds_name].get(model)
            if md is None:
                continue
            safe_m = model.replace("/", "_")
            fname = f"heat_{ds_name}_{safe_m}.png"
            gen_heatmap(md, comp_prefix, ds_name, model, sub_dir / fname)
            ds_heat.append({"file": fname, "model": model})
        heatmap_files[ds_name] = ds_heat

    manifest = build_manifest(sub_dir, {
        "panelA": heatmap_files,
        "models": avail_models,
        # Keep canonical order: setty, endo, dentate
        "datasets": [d for d in REPRESENTATIVE_DATASETS if d in bio_datasets],
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 6 subplots")
    parser.add_argument("--series", required=True, choices=["dpmm"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / args.series / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(args.series, out)
