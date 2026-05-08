"""
Graphes et tableaux a partir des fichiers de resultats.
Necessite d'avoir lance :
  - tests/test_recognition.py  -> tests/results/recognition_results.json
  - tests/test_placement.py    -> tests/results/placement_results.json
  - tests/benchmark_timing.py  -> tests/results/timing_results.json

Usage :
  python tests/plot_results.py                     # tout
  python tests/plot_results.py --only recog
  python tests/plot_results.py --only placement
  python tests/plot_results.py --only timing
"""
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
RECOG_PATH     = os.path.join(RESULTS_DIR, "recognition_results.json")
PLACEMENT_PATH = os.path.join(RESULTS_DIR, "placement_results.json")
TIMING_PATH    = os.path.join(RESULTS_DIR, "timing_results.json")

COLORS = {
    "V1 (gpt-4o)":         "#4C72B0",
    "V1.1 (gpt-4o)":       "#DD8452",
    "V2 (gpt-4o)":         "#C44E52",
    "V1 (llama3.1)":       "#8172B2",
    "V1.1 (llama3.1)":     "#937860",
    "V2 (llama3.1)":       "#DA8BC3",
    "V1.1.1 (phi3-scene)": "#55A868",
}

_FALLBACK_COLORS = [
    "#CCB974", "#6ACC65", "#4878D0", "#EE854A", "#D65F5F",
    "#956CB4", "#8C8C8C", "#B5BD61", "#17BECF", "#E377C2",
]
_fallback_seen = {}


def version_color(name):
    if name in COLORS:
        return COLORS[name]
    if name not in _fallback_seen:
        _fallback_seen[name] = _FALLBACK_COLORS[len(_fallback_seen) % len(_FALLBACK_COLORS)]
    return _fallback_seen[name]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> {path}")


def avg(values):
    return sum(values) / len(values) if values else 0.0


def print_table(title, headers, rows, footer=None):
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep   = "  " + "-" * (sum(col_w) + 3 * len(col_w))
    line  = lambda cells: "  " + "   ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells))
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(line(headers))
    print(sep)
    for row in rows:
        print(line(row))
    if footer:
        print(sep)
        print(line(footer))
    print(f"{'='*65}")


# ---------------------------------------------------------------
# RECOGNITION
# ---------------------------------------------------------------

def plot_recognition(data):
    versions = list(data["versions"].keys())
    ids      = [e["id"] for e in data["versions"][versions[0]]]
    n_runs   = data["n_runs"]

    # tableau
    rows = []
    gstats = {}
    for v in versions:
        entries = data["versions"][v]
        prec = avg([e["aggregated"]["precision_mean"] for e in entries])
        rec  = avg([e["aggregated"]["recall_mean"]    for e in entries])
        f1   = avg([e["aggregated"]["f1_mean"]        for e in entries])
        t    = avg([e["time_mean_s"]                  for e in entries])
        gstats[v] = dict(prec=prec, rec=rec, f1=f1, t=t)
        rows.append([v, f"{prec*100:.1f}%", f"{rec*100:.1f}%", f"{f1*100:.1f}%", f"{t:.2f}s"])

    print_table(
        f"RECONNAISSANCE  ({n_runs} runs/prompt)",
        ["Version", "Precision", "Recall", "F1", "Temps moy"],
        rows,
    )

    if data.get("synonym_results"):
        syn = data["synonym_results"]
        ok  = sum(1 for r in syn if r["matched"])
        print(f"  Synonymes : {ok}/{len(syn)} ({ok/len(syn)*100:.0f}%)")

    if data.get("threshold_results"):
        thr = data["threshold_results"]
        ok  = sum(1 for r in thr if r["correct"])
        print(f"  Seuil     : {ok}/{len(thr)} corrects ({ok/len(thr)*100:.0f}%)")

    # fig 1 : barres precision / recall / f1
    x_pos = np.arange(len(versions))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f"Reconnaissance — moyenne sur {len(ids)} prompts ({n_runs} runs)",
                 fontsize=12, fontweight="bold")
    for ax, metric, label in zip(axes, ["prec", "rec", "f1"], ["Precision", "Recall", "F1"]):
        vals = [gstats[v][metric] * 100 for v in versions]
        bars = ax.bar(x_pos, vals, color=[version_color(v) for v in versions],
                      width=0.5, alpha=0.85)
        ax.set_xlim(-0.6, len(versions) - 0.4)
        ax.set_xticks([])
        ax.set_ylim(0, 115)
        ax.set_ylabel("%")
        ax.set_title(label)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    handles = [Patch(facecolor=version_color(v), alpha=0.85, label=v) for v in versions]
    fig.legend(handles=handles, loc="lower center", ncol=len(versions),
               bbox_to_anchor=(0.5, 0.0), fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    savefig(fig, "recog_global.png")

    # fig 2 : F1 par prompt
    x = np.arange(len(ids))
    w = 0.8 / len(versions)
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    for i, v in enumerate(versions):
        f1s = [e["aggregated"]["f1_mean"] for e in data["versions"][v]]
        ax2.bar(x + i * w - 0.4 + w / 2, f1s, width=w,
                label=v, color=version_color(v), alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("F1 score")
    ax2.set_title("F1 reconnaissance par prompt et par version")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    ax2.legend()
    plt.tight_layout()
    savefig(fig2, "recog_f1_per_form.png")

    # fig 3 : temps par prompt
    all_times = [e["time_mean_s"] for v in versions for e in data["versions"][v]]
    ylim_top = np.percentile(all_times, 95) * 1.4

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    for v in versions:
        times = [e["time_mean_s"] for e in data["versions"][v]]
        ax3.plot(ids, times, marker="o", label=v, color=version_color(v),
                 linewidth=1.5, markersize=4)
        for xi, t in enumerate(times):
            if t > ylim_top:
                ax3.text(xi, ylim_top * 0.97, f"^{t:.0f}s",
                         ha="center", va="top", fontsize=7,
                         color=version_color(v), fontweight="bold")
    ax3.set_ylim(0, ylim_top)
    ax3.set_xticks(range(len(ids)))
    ax3.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Temps (s)")
    ax3.set_title("Temps de reconnaissance par prompt")
    ax3.legend()
    plt.tight_layout()
    savefig(fig3, "recog_time_per_form.png")


# ---------------------------------------------------------------
# PLACEMENT (relations + orientations)
# ---------------------------------------------------------------

def plot_placement(data):
    n_runs = data["n_runs"]

    # detection dynamique par structure des entrees
    rel_versions     = {k: v for k, v in data.items()
                        if k != "n_runs" and isinstance(v, list) and v and "rel_f1_mean" in v[0]}
    spatial_versions = {k: v for k, v in data.items()
                        if k != "n_runs" and isinstance(v, list) and v and "quat_valid_rate" in v[0]}

    ids = [e["id"] for entries in (list(rel_versions.values()) or list(spatial_versions.values()))[:1]
           for e in entries]

    # tableaux relations + orientations
    rows_rel, rows_ori = [], []
    rel_stats, ori_stats = {}, {}
    for label, entries in rel_versions.items():
        rel_f1  = avg([e["rel_f1_mean"]   for e in entries])
        rel_pre = avg([e["rel_precision"]  for e in entries])
        rel_rec = avg([e["rel_recall"]     for e in entries])
        ori_f1  = avg([e["ori_f1_mean"]   for e in entries])
        ori_pre = avg([e["ori_precision"]  for e in entries])
        ori_rec = avg([e["ori_recall"]     for e in entries])
        rel_stats[label] = dict(f1=rel_f1, pre=rel_pre, rec=rel_rec)
        ori_stats[label] = dict(f1=ori_f1, pre=ori_pre, rec=ori_rec)
        rows_rel.append([label, f"{rel_pre*100:.1f}%", f"{rel_rec*100:.1f}%", f"{rel_f1*100:.1f}%"])
        rows_ori.append([label, f"{ori_pre*100:.1f}%", f"{ori_rec*100:.1f}%", f"{ori_f1*100:.1f}%"])

    if rows_rel:
        print_table(f"PLACEMENT — RELATIONS  ({n_runs} runs/prompt)",
                    ["Version", "Precision", "Recall", "F1"], rows_rel)
    if rows_ori:
        print_table(f"PLACEMENT — ORIENTATIONS  ({n_runs} runs/prompt)",
                    ["Version", "Precision", "Recall", "F1"], rows_ori)

    # tableaux coherence spatiale
    for label, entries in spatial_versions.items():
        quat  = avg([e["quat_valid_rate"]    for e in entries])
        z_ok  = avg([e["pos_plausible_rate"] for e in entries])
        order = avg([e["rel_order_acc_mean"] for e in entries])
        print_table(
            f"PLACEMENT — {label} COHERENCE SPATIALE  ({n_runs} runs/prompt)",
            ["Quat valides", "Z >= 0", "Ordre vertical"],
            [[f"{quat*100:.1f}%", f"{z_ok*100:.1f}%", f"{order*100:.1f}%"]],
        )

    # fig 1 : F1 relations par prompt
    if rel_versions:
        x = np.arange(len(ids))
        w = 0.8 / len(rel_versions)
        fig, ax = plt.subplots(figsize=(16, 5))
        for i, (label, entries) in enumerate(rel_versions.items()):
            ax.bar(x + i * w - 0.4 + w / 2, [e["rel_f1_mean"] for e in entries],
                   width=w, label=label, color=version_color(label), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("F1 relations")
        ax.set_title("F1 relations par prompt et par version")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
        ax.legend()
        plt.tight_layout()
        savefig(fig, "placement_rel_f1_per_form.png")

    # fig 2 : barres globales relations + orientations
    if rel_stats:
        vs_all = list(rel_stats.keys())
        x_pos  = np.arange(len(vs_all))
        fig2, axes = plt.subplots(1, 2, figsize=(max(8, len(rel_stats) * 2 + 4), 5))
        fig2.suptitle("Placement — relations et orientations (moyennes globales)",
                      fontsize=12, fontweight="bold")
        for ax, stats, title in zip(axes, [rel_stats, ori_stats], ["Relations F1", "Orientations F1"]):
            vs   = list(stats.keys())
            vals = [stats[v]["f1"] * 100 for v in vs]
            bars = ax.bar(x_pos[:len(vs)], vals, color=[version_color(v) for v in vs],
                          width=0.4, alpha=0.85)
            ax.set_xlim(-0.6, len(vs) - 0.4)
            ax.set_xticks([])
            ax.set_ylim(0, 115)
            ax.set_ylabel("%")
            ax.set_title(title)
            ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        handles = [Patch(facecolor=version_color(v), alpha=0.85, label=v) for v in vs_all]
        fig2.legend(handles=handles, loc="lower center", ncol=len(vs_all),
                    bbox_to_anchor=(0.5, 0.0), fontsize=9)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        savefig(fig2, "placement_global.png")

    # fig 3 : coherence spatiale par prompt
    for label, entries in spatial_versions.items():
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        ax3.plot(ids, [e["quat_valid_rate"]    for e in entries],
                 marker="o", label="Quat valides",  linewidth=1.5, markersize=4)
        ax3.plot(ids, [e["pos_plausible_rate"] for e in entries],
                 marker="s", label="Z >= 0",         linewidth=1.5, markersize=4)
        ax3.plot(ids, [e["rel_order_acc_mean"] for e in entries],
                 marker="^", label="Ordre vertical", linewidth=1.5, markersize=4)
        ax3.set_xticks(range(len(ids)))
        ax3.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
        ax3.set_ylim(-0.05, 1.15)
        ax3.set_ylabel("Taux")
        ax3.set_title(f"{label} — coherence spatiale par prompt")
        ax3.legend()
        plt.tight_layout()
        savefig(fig3, f"placement_spatial_{label.replace(' ', '_').replace('/', '-')}.png")


# ---------------------------------------------------------------
# TIMING
# ---------------------------------------------------------------

def plot_timing(data):
    versions = list(data["versions"].keys())
    ids      = [e["id"] for e in data["versions"][versions[0]]]
    n_runs   = data["n_runs"]

    rows = []
    gstats = {}
    for v in versions:
        entries = data["versions"][v]
        gm   = avg([e["mean_s"] for e in entries])
        gsd  = avg([e["std_s"]  for e in entries])
        gmin = min(e["min_s"] for e in entries)
        gmax = max(e["max_s"] for e in entries)
        gstats[v] = dict(mean=gm, std=gsd, mn=gmin, mx=gmax)
        rows.append([v, f"{gm:.2f}s", f"{gsd:.2f}s", f"{gmin:.2f}s", f"{gmax:.2f}s"])

    print_table(
        f"PIPELINE COMPLET — TEMPS  ({n_runs} runs/prompt)",
        ["Version", "Moy", "Std", "Min", "Max"],
        rows,
    )

    # fig 1 : bar moyen + errorbar
    fig, ax = plt.subplots(figsize=(7, 5))
    means_ = [gstats[v]["mean"] for v in versions]
    stds_  = [gstats[v]["std"]  for v in versions]
    bars   = ax.bar(versions, means_, yerr=stds_, capsize=6,
                    color=[version_color(v) for v in versions],
                    width=0.5, alpha=0.85)
    ax.set_ylabel("Temps (s)")
    ax.set_title("Temps moyen — pipeline complet (avec Genesis si applicable)")
    for bar, m, s in zip(bars, means_, stds_):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.2,
                f"{m:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "timing_global.png")

    # fig 2 : temps par prompt avec errorbar
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for v in versions:
        t_means = [e["mean_s"] for e in data["versions"][v]]
        t_stds  = [e["std_s"]  for e in data["versions"][v]]
        ax2.errorbar(ids, t_means, yerr=t_stds, label=v,
                     color=version_color(v), marker="o",
                     linewidth=1.5, markersize=4, capsize=3)
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Temps (s)")
    ax2.set_title("Temps pipeline complet par prompt")
    ax2.legend()
    plt.tight_layout()
    savefig(fig2, "timing_per_form.png")

    # fig 3 : boxplot distribution
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    all_runs = [[t for e in data["versions"][v] for t in e["runs"]] for v in versions]
    bp = ax3.boxplot(all_runs, labels=versions, patch_artist=True)
    for patch, v in zip(bp["boxes"], versions):
        patch.set_facecolor(version_color(v))
        patch.set_alpha(0.7)
    ax3.set_ylabel("Temps (s)")
    ax3.set_title("Distribution des temps — pipeline complet")
    plt.tight_layout()
    savefig(fig3, "timing_boxplot.png")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["recog", "placement", "timing"], default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    tasks = {
        "recog":     (RECOG_PATH,     plot_recognition, "python tests/test_recognition.py"),
        "placement": (PLACEMENT_PATH, plot_placement,   "python tests/test_placement.py"),
        "timing":    (TIMING_PATH,    plot_timing,      "python tests/benchmark_timing.py"),
    }

    to_run = [args.only] if args.only else list(tasks.keys())

    for key in to_run:
        path, fn, hint = tasks[key]
        if not os.path.exists(path):
            print(f"[{key}] Fichier introuvable : {path}")
            print(f"         Lance d'abord : {hint}")
        else:
            fn(load(path))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
