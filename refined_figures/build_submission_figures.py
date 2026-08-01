#!/usr/bin/env python3
"""Build the ten four-panel PanODE-DPMM submission figures from saved assets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "article" / "dpmm" / "figures" / "submission"
COLORS = {"Pure-AE": "#4E79A7", "DPMM-Base": "#F28E2B", "DPMM-FM": "#5C6BC0"}
LETTERS = "ABCD"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
    })


def canvas(title: str):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for letter, ax in zip(LETTERS, axes.flat):
        ax.text(-0.11, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom")
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    return fig, axes.flat


def save(fig, number: int, slug: str, sources: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.003, f"Source: {sources}", fontsize=5.5, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"F{number:02d}_{slug}.{suffix}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    assert len(fig.axes) == 4
    plt.close(fig)


def text_panel(ax, title: str, lines: list[str], color="#2F4B7C") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.88, transform=ax.transAxes,
                               facecolor="#F7F8FA", edgecolor=color, linewidth=1.2))
    ax.text(0.07, 0.86, "\n".join(lines), transform=ax.transAxes, va="top",
            linespacing=1.65, fontsize=8.2)


def bars(ax, labels, series: dict[str, list[float]], title, ylabel="", rotation=20):
    x = np.arange(len(labels)); width = 0.78 / max(len(series), 1)
    for j, (name, values) in enumerate(series.items()):
        ax.bar(x + (j - (len(series)-1)/2)*width, values, width*0.92,
               label=name, color=list(COLORS.values())[j % 3], alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_ylabel(ylabel)
    if len(series) > 1: ax.legend(frameon=False)


def load_core():
    return pd.read_csv(ROOT / "experiments/results/dpmm_fm_core/summary.csv")


def fig1():
    fig, ax = canvas("F1 | Adaptive-prior architecture and operating hypotheses")
    text_panel(ax[0], "Shared autoencoder", ["3,000 HVGs → encoder [256,128]", "10-D latent representation", "reconstruction objective shared", "across all three variants"])
    text_panel(ax[1], "Adaptive DPMM prior", ["90% reconstruction warmup", "Bayesian mixture refit", "every 10 epochs", "cluster count inferred online"], "#F28E2B")
    text_panel(ax[2], "Conditional flow refinement", ["transport toward DPMM centres", "flow weight λFM = 0.1", "noise scale = 0.5", "projection-fidelity hypothesis"], "#5C6BC0")
    text_panel(ax[3], "Pre-specified operating points", ["Pure-AE: label concordance", "DPMM-Base: compact geometry", "DPMM-FM: projection fidelity", "No universal-best assumption"], "#2E7D32")
    save(fig, 1, "architecture", "models/dpmm_base.py; models/dpmm_flow_matching.py")


def fig2():
    d = load_core().groupby("model", as_index=False).mean(numeric_only=True)
    order = [m for m in ["Pure-AE", "DPMM-Base", "DPMM-FM"] if m in set(d.model)]
    q = d.set_index("model").loc[order]
    fig, ax = canvas("F2 | Geometry–concordance trade-off")
    bars(ax[0], order, {"NMI": q.NMI.tolist(), "ARI": q.ARI.tolist()}, "Label concordance", "score")
    bars(ax[1], order, {"ASW": q.ASW.tolist()}, "Cluster compactness", "ASW")
    bars(ax[2], order, {"DRE-UMAP": q.DRE_umap_overall_quality.tolist(), "DRE-tSNE": q.DRE_tsne_overall_quality.tolist()}, "Projection fidelity", "quality")
    ax[3].scatter(q.NMI, q.ASW, s=90, c=[COLORS[m] for m in order])
    for m in order: ax[3].annotate(m, (q.loc[m,"NMI"], q.loc[m,"ASW"]), xytext=(4,4), textcoords="offset points")
    ax[3].set(xlabel="NMI ↑", ylabel="ASW ↑", title="Pareto, not dominance"); ax[3].grid(alpha=.2)
    save(fig, 2, "tradeoff", "experiments/results/dpmm_fm_core/summary.csv")


def fig3():
    metrics=["NMI","ARI","ASW","DAV"]; wins=[2,4,12,12]; losses=[10,8,0,0]
    p=[.999,.924,2.44e-4,2.44e-4]; delta=[-.639,-.431,1,1]
    fig, ax=canvas("F3 | Core statistical evidence (12 datasets)")
    bars(ax[0],metrics,{"wins":wins,"losses":losses},"Directional outcomes","datasets",0)
    ax[1].bar(metrics,-np.log10(p),color="#4E79A7"); ax[1].axhline(-math.log10(.05),color="#C44E52",ls="--"); ax[1].set(title="Wilcoxon evidence",ylabel="−log10(p)")
    ax[2].bar(metrics,delta,color=["#C44E52" if v<0 else "#55A868" for v in delta]); ax[2].axhline(0,color="black",lw=.7); ax[2].set(title="Cliff's δ",ylabel="effect")
    text_panel(ax[3],"Interpretation",["ASW and DAV: 12/12 wins", "p = 2.44×10⁻⁴; δ = 1.0", "NMI/ARI losses are directional", "but not signed-rank significant"])
    save(fig,3,"statistics","article/dpmm/tables/wilcoxon_core.tex")


def fig4():
    d=pd.read_csv(ROOT/"experiments/results/dpmm_fm_sensitivity_core/summary.csv")
    fig,ax=canvas("F4 | DPMM-FM sensitivity across eight datasets")
    specs=[("flow_weight","NMI","Flow weight: NMI"),("flow_weight","DRE_umap_overall_quality","Flow weight: DRE-UMAP"),("flow_noise_scale","ARI","Noise scale: ARI"),("flow_noise_scale","LSE_overall_quality","Noise scale: LSE")]
    for a,(sw,m,title) in zip(ax,specs):
        g=d[d.Sweep==sw].groupby("SweepVal")[m].agg(["mean","std"])
        a.errorbar(g.index,g["mean"],yerr=g["std"],marker="o",capsize=3,color="#5C6BC0"); a.set(title=title,xlabel=sw.replace("_"," "),ylabel=m.split("_")[0])
    save(fig,4,"sensitivity","experiments/results/dpmm_fm_sensitivity_core/summary.csv")


def fig5():
    p=ROOT/"benchmarks/training_dynamics_results/DPMM-Base_setty_history.json"; h=json.load(p.open())
    fig,ax=canvas("F5 | Saved training dynamics and cost")
    ax[0].plot(h["train_loss"],color="#4E79A7"); ax[0].set(title="Total loss",xlabel="epoch",ylabel="loss")
    ax[1].plot(h["recon_loss"],color="#55A868"); ax[1].set(title="Reconstruction loss",xlabel="epoch",ylabel="loss")
    ax[2].plot(h["dpmm_loss"],color="#F28E2B"); ax[2].axvline(540,color="#333",ls="--",lw=.8); ax[2].set(title="DPMM loss after warmup",xlabel="epoch",ylabel="loss")
    snaps=pd.DataFrame(h["snapshots"]); ax[3].plot(snaps.epoch,snaps.NMI,label="NMI"); ax[3].plot(snaps.epoch,snaps.ARI,label="ARI"); ax[3].plot(snaps.epoch,snaps.ASW,label="ASW"); ax[3].legend(frameon=False); ax[3].set(title="Snapshot metrics",xlabel="epoch",ylabel="score")
    save(fig,5,"training","benchmarks/training_dynamics_results/DPMM-Base_setty_history.json")


def fig6():
    fig,ax=canvas("F6 | Biological exploration on saved representations")
    for a,ds in zip(ax[:3],["setty","endo","dentate"]):
        p=ROOT/f"benchmarks/biological_validation/results/DPMM-Base_{ds}_umap_data.npz"; z=np.load(p,allow_pickle=True); emb=z["umap_emb"]; labels=z["labels"]
        a.scatter(emb[:,0],emb[:,1],c=labels,s=3,cmap="tab20",alpha=.65); a.set(title=f"{ds}: saved UMAP",xticks=[],yticks=[])
    files=sorted((ROOT/"benchmarks/biological_validation/results/DPMM-Base").glob("DPMM-Base_dentate_enrichment_comp*.csv"))
    terms=[]; scores=[]
    for p in files[:6]:
        d=pd.read_csv(p); row=d.sort_values("Adjusted P-value").iloc[0]; terms.append(str(row.Term)[:28]); scores.append(-math.log10(max(float(row["Adjusted P-value"]),1e-300)))
    ax[3].barh(terms[::-1],scores[::-1],color="#55A868"); ax[3].set(title="Dentate GO exploration",xlabel="−log10 adjusted p")
    save(fig,6,"biological","benchmarks/biological_validation/results/DPMM-Base_*_umap_data.npz; DPMM-Base/*enrichment*.csv")


def fig7():
    d=pd.read_csv(ROOT/"article/dpmm/data/per_variant_external_summary.csv"); d=d[d.Model.isin(["DPMM-Base","Pure-AE"])].set_index("Model")
    fig,ax=canvas("F7 | External benchmark: full screen vs selected view")
    bars(ax[0],list(d.index),{"all significant":d.Significant_all.tolist()},"Full-screen significant wins","tests")
    bars(ax[1],list(d.index),{"core significant":d.Core_significant.tolist()},"Core significant wins","of 44")
    bars(ax[2],list(d.index),{"core win rate":d.Core_win_rate.tolist()},"Core win rate","fraction")
    text_panel(ax[3],"Selection and provenance",["Statistics: 56 datasets × 18 baselines", "Display: 12 complete core datasets", "11 representative baseline methods", "pairwise upstream exports: PENDING"])
    save(fig,7,"external_selection","article/dpmm/data/per_variant_external_summary.csv; article/dpmm/tables/external_winrate.tex; PROVENANCE-GAP.md")


def fig8():
    d=pd.read_csv(ROOT/"article/dpmm/data/per_variant_external_summary.csv").sort_values("Core_win_rate",ascending=False).head(8)
    fig,ax=canvas("F8 | External evidence decomposition and provenance boundary")
    ax[0].barh(d.Model[::-1],d.Core_win_rate[::-1],color="#4E79A7"); ax[0].set(title="Top summary win rates",xlabel="core win rate")
    ax[1].scatter(d.Significant_all,d.Core_significant,c=d.Core_win_rate,cmap="viridis",s=55); ax[1].set(title="All vs core significant wins",xlabel="all significant",ylabel="core significant")
    ax[2].bar(d.Model.str.replace("Pure-","P-").str.replace("DPMM-","D-").str.replace("Topic-","T-").head(6),d.Total_tests.head(6),color="#9E9E9E"); ax[2].tick_params(axis="x",rotation=35); ax[2].set(title="Test-count coverage",ylabel="tests")
    text_panel(ax[3],"PENDING-PROVENANCE",["Metric-family attribution cannot", "be regenerated in this repository:", "pairwise external exports missing.", "No ASW/DAV split is fabricated."] ,"#C44E52")
    save(fig,8,"external_provenance","article/dpmm/data/per_variant_external_summary.csv; article/dpmm/PROVENANCE-GAP.md")


def fig9():
    fig,ax=canvas("F9 | Downstream utility and computational cost")
    bars(ax[0],["Pure-AE","DPMM-Base"],{"accuracy":[.784,.725]},"kNN accuracy","score")
    bars(ax[1],["Pure-AE","DPMM-Base"],{"macro-F1":[.572,.485]},"kNN macro-F1","score")
    bars(ax[2],["Pure-AE","DPMM-Base"],{"time (s)":[46.4,50.5]},"Mean training time","seconds")
    bars(ax[3],["Pure-AE","DPMM-Base"],{"parameters":[1615430,1615430]},"Parameter count","parameters")
    save(fig,9,"utility_cost","article/dpmm/tables/knn.tex; article/dpmm/tables/runtime.tex")


def fig10():
    fig,ax=canvas("F10 | Task-conditional decision map")
    text_panel(ax[0],"Label recovery",["Choose Pure-AE", "NMI 0.609; ARI 0.406", "kNN accuracy 0.784", "No prior-induced boundary shift"],"#4E79A7")
    text_panel(ax[1],"Compact geometry",["Choose DPMM-Base", "ASW 0.374; DAV 0.868", "12/12 ASW and DAV wins", "Accept concordance cost"],"#F28E2B")
    text_panel(ax[2],"Projection fidelity",["Choose DPMM-FM", "DRE/LSE/DREX operating point", "Flow smooths global manifold", "Not a clustering optimum"],"#5C6BC0")
    labels=["labels","geometry","projection","classification"]; mat=np.array([[3,1,1,3],[1,3,2,1],[0,2,3,0]])
    im=ax[3].imshow(mat,cmap="Blues",vmin=0,vmax=3); ax[3].set_xticks(range(4),labels,rotation=25,ha="right"); ax[3].set_yticks(range(3),["Pure-AE","DPMM-Base","DPMM-FM"]); ax[3].set_title("Evidence-based suitability",loc="left",fontweight="bold")
    for i in range(3):
        for j in range(4): ax[3].text(j,i,str(mat[i,j]),ha="center",va="center",color="white" if mat[i,j]>1 else "black")
    save(fig,10,"decision_map","article/dpmm/tables/*.tex; experiments/results/dpmm_fm_core/summary.csv")


def main():
    style()
    required=[ROOT/"experiments/results/dpmm_fm_core/summary.csv",ROOT/"experiments/results/dpmm_fm_sensitivity_core/summary.csv",ROOT/"article/dpmm/data/per_variant_external_summary.csv"]
    for p in required:
        if not p.exists(): raise FileNotFoundError(p)
    for fn in (fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10): fn()
    print(f"Built 10 four-panel figures in {OUT}")


if __name__ == "__main__": main()
