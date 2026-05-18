"""
Graphes et tableaux LaTeX a partir des fichiers de resultats.
Necessite d'avoir lance :
  - tests/test_recognition.py  -> tests/results/recognition_results.json
  - tests/test_placement.py    -> tests/results/placement_results.json
  - tests/benchmark_timing.py  -> tests/results/timing_results.json

Usage :
  python tests/plot_results.py                       # tout
  python tests/plot_results.py --only recog
  python tests/plot_results.py --only placement
  python tests/plot_results.py --only timing
  python tests/plot_results.py --latex               # affiche aussi les blocs LaTeX
  python tests/plot_results.py --no-show             # sauvegarde sans ouvrir
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
VISION_V3_PATH    = os.path.join(RESULTS_DIR, "vision_results_v3.json")
VISION_V31_PATH   = os.path.join(RESULTS_DIR, "vision_results_v3_1.json")

COLORS = {
    "V3":                          "#937860",
    "V3.1":                        "#DA8BC3",
    "V1 (gpt-4o)":                 "#C44E52",
    "V1.1 (gpt-4o)":               "#4C72B0",
    "V1.1.1 (gpt-4o)":             "#DD8452",
    "V1 (llama3.1)":               "#DA8BC3",
    "V1.1 (llama3.1)":             "#8172B2",
    "V1.1.1 (llama3.1)":           "#937860",
    "V2 (phi3-scene)":             "#55A868",
    "V1 (gpt-4o) + boucle":        "#E87E82",
    "V1.1 (gpt-4o) + boucle":      "#7A9EC8",
    "V1.1.1 (gpt-4o) + boucle":    "#EEA47A",
    "V1 (llama3.1) + boucle":      "#EAB0D3",
    "V1.1 (llama3.1) + boucle":    "#A192C2",
    "V1.1.1 (llama3.1) + boucle":  "#B39A80",
    "V2 (phi3-scene) + boucle":    "#75C888",
}

_FALLBACK = [
    "#CCB974", "#6ACC65", "#4878D0", "#EE854A", "#D65F5F",
    "#956CB4", "#8C8C8C", "#B5BD61", "#17BECF", "#E377C2",
]
_fallback_seen = {}


def version_color(name):
    if name in COLORS:
        return COLORS[name]
    if name not in _fallback_seen:
        _fallback_seen[name] = _FALLBACK[len(_fallback_seen) % len(_FALLBACK)]
    return _fallback_seen[name]


# ---------------------------------------------------------------
# groupement par modele
# ---------------------------------------------------------------

def is_gpt(k):    return "gpt"      in k.lower()
def is_llama(k):  return "llama"    in k.lower()
def is_v2(k):     return k.lower().startswith("v2")
def is_boucle(k): return "+ boucle" in k

def strip_boucle(k): return k.replace(" + boucle", "")


def group_by_model(all_keys):
    """Retourne {"gpt": [...], "llama": [...], "other": [...]}.
    V2 phi3 (sans tag gpt/llama) est inclus dans les deux groupes.
    Les variantes + boucle suivent la meme logique que leur version de base."""
    def base(k): return strip_boucle(k)
    phi3  = [k for k in all_keys if is_v2(base(k)) and not is_gpt(base(k)) and not is_llama(base(k))]
    gpt   = [k for k in all_keys if is_gpt(base(k))]   + phi3
    llama = [k for k in all_keys if is_llama(base(k))] + phi3
    other = [k for k in all_keys if not is_gpt(base(k)) and not is_llama(base(k)) and not is_v2(base(k))]
    return {"gpt": gpt, "llama": llama, "other": other}


def version_sort_key(key):
    """V1 < V1.1 < V1.1.1 < V2 ; GPT avant Llama ; sans boucle avant avec boucle."""
    bk = strip_boucle(key)
    if is_v2(bk):       base = 40
    elif "1.1.1" in bk: base = 30
    elif "1.1"   in bk: base = 20
    else:                base = 10
    sub    = 0 if is_gpt(bk) else 1 if is_llama(bk) else 2
    bouclv = 1 if is_boucle(key) else 0
    return (base, sub, bouclv)


def latex_label(key):
    """Convertit un nom de version en label LaTeX propre."""
    lbl = key
    lbl = lbl.replace("gpt-4o-mini",  "GPT-4o-mini")
    lbl = lbl.replace("gpt-4o",       "GPT-4o")
    lbl = lbl.replace("llama3.2",     "Llama 3.2")
    lbl = lbl.replace("llama3.1",     "Llama 3.1")
    lbl = lbl.replace("phi3-scene",   r"\texttt{phi3-scene}")
    lbl = lbl.replace("+ boucle",     r"+ \textit{boucle}")
    return lbl


# ---------------------------------------------------------------
# utilitaires
# ---------------------------------------------------------------

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


def avg(values):
    return sum(values) / len(values) if values else 0.0


def print_table(title, headers, rows, footer=None):
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep  = "  " + "-" * (sum(col_w) + 3 * len(col_w))
    line = lambda cells: "  " + "   ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells))
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(line(headers))
    print(sep)
    for row in rows:
        print(line(row))
    if footer:
        print(sep)
        print(line(footer))
    print(f"{'='*70}")


# ---------------------------------------------------------------
# RECOGNITION
# ---------------------------------------------------------------

def recog_compute_stats(data):
    stats = {}
    for v, entries in data["versions"].items():
        stats[v] = {
            "prec":    avg([e["aggregated"]["precision_mean"] for e in entries]),
            "rec":     avg([e["aggregated"]["recall_mean"]    for e in entries]),
            "f1":      avg([e["aggregated"]["f1_mean"]        for e in entries]),
            "t":       avg([e["time_mean_s"]                  for e in entries]),
            "entries": entries,
        }
    return stats


def _plot_recog_group(stats, group_keys, ids, n_runs, model_label, suffix):
    if not group_keys:
        return

    ordered = sorted(group_keys, key=version_sort_key)

    # fig 1 : barres globales P / R / F1
    x_pos = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 3, figsize=(max(11, len(ordered) * 2 + 3), 5))
    fig.suptitle(
        f"Reconnaissance {model_label} — {len(ids)} prompts ({n_runs} runs)",
        fontsize=12, fontweight="bold",
    )
    for ax, metric, label in zip(axes, ["prec", "rec", "f1"], ["Precision", "Recall", "F1"]):
        vals = [stats[v][metric] * 100 for v in ordered]
        bars = ax.bar(x_pos, vals, color=[version_color(v) for v in ordered],
                      width=0.5, alpha=0.85)
        ax.set_xlim(-0.6, len(ordered) - 0.4)
        ax.set_xticks([])
        ax.set_ylim(0, 115)
        ax.set_ylabel("%")
        ax.set_title(label)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    handles = [Patch(facecolor=version_color(v), alpha=0.85, label=v) for v in ordered]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(ordered), 4),
               bbox_to_anchor=(0.5, 0.0), fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    savefig(fig, f"recog_global_{suffix}.png")

    # fig 2 : F1 par prompt
    x = np.arange(len(ids))
    w = 0.8 / len(ordered)
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    for i, v in enumerate(ordered):
        f1s = [e["aggregated"]["f1_mean"] for e in stats[v]["entries"]]
        ax2.bar(x + i * w - 0.4 + w / 2, f1s, width=w,
                label=v, color=version_color(v), alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("F1 score")
    ax2.set_title(f"F1 reconnaissance par prompt — {model_label}")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig2, f"recog_f1_per_form_{suffix}.png")

    # fig 3 : temps par prompt
    all_times = [e["time_mean_s"] for v in ordered for e in stats[v]["entries"]]
    ylim_top  = np.percentile(all_times, 95) * 1.4 if all_times else 10.0
    fig3, ax3 = plt.subplots(figsize=(14, 4))
    for v in ordered:
        times = [e["time_mean_s"] for e in stats[v]["entries"]]
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
    ax3.set_title(f"Temps de reconnaissance par prompt — {model_label}")
    ax3.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig3, f"recog_time_per_form_{suffix}.png")


def _plot_time_combined(all_versions, time_fn, ids, n_runs, title, fname):
    """Plot temps par prompt (ligne + points) pour toutes versions sur un seul graphe."""
    ordered = sorted(all_versions, key=version_sort_key)
    if not ordered:
        return

    all_times = [time_fn(e) for v in ordered for e in all_versions[v]]
    ylim_top  = np.percentile(all_times, 95) * 1.4 if all_times else 10.0

    fig, ax = plt.subplots(figsize=(14, 4))
    for v in ordered:
        times = [time_fn(e) for e in all_versions[v]]
        ax.plot(ids, times, marker="o", label=v, color=version_color(v),
                linewidth=1.5, markersize=4)
        for xi, t in enumerate(times):
            if t > ylim_top:
                ax.text(xi, ylim_top * 0.97, f"^{t:.0f}s",
                        ha="center", va="top", fontsize=7,
                        color=version_color(v), fontweight="bold")
    ax.set_ylim(0, ylim_top)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Temps (s)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig, fname)


def plot_recognition(data, latex=False):
    n_runs  = data["n_runs"]
    stats   = recog_compute_stats(data)
    ids     = [e["id"] for e in next(iter(data["versions"].values()))]
    versions = list(stats.keys())

    # tableau console global
    rows = []
    for v in sorted(versions, key=version_sort_key):
        s = stats[v]
        rows.append([v, f"{s['prec']*100:.1f}%", f"{s['rec']*100:.1f}%",
                     f"{s['f1']*100:.1f}%", f"{s['t']:.2f}s"])
    print_table(f"RECONNAISSANCE  ({n_runs} runs/prompt)",
                ["Version", "Precision", "Recall", "F1", "Temps moy"], rows)

    if data.get("synonym_results"):
        syn = data["synonym_results"]
        ok  = sum(1 for r in syn if r["matched"])
        print(f"  Synonymes : {ok}/{len(syn)} ({ok/len(syn)*100:.0f}%)")
    if data.get("threshold_results"):
        thr = data["threshold_results"]
        ok  = sum(1 for r in thr if r["correct"])
        print(f"  Seuil     : {ok}/{len(thr)} corrects ({ok/len(thr)*100:.0f}%)")

    groups = group_by_model(versions)
    for label, suffix in [("GPT", "gpt"), ("Llama", "llama"), ("autres", "other")]:
        _plot_recog_group(stats, groups[suffix.lower() if suffix != "autres" else "other"],
                          ids, n_runs, label, suffix.lower())

    _plot_time_combined(
        {v: stats[v]["entries"] for v in versions},
        lambda e: e["time_mean_s"],
        ids, n_runs,
        f"Temps de reconnaissance — toutes versions ({n_runs} runs/prompt)",
        "recog_time_all.png",
    )

    if latex:
        _print_latex_recog(stats)


def _print_latex_recog(stats):
    ordered = sorted(stats.keys(), key=version_sort_key)
    print("\n% ====== Tableau reconnaissance ======")
    print(r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lcccc}
\hline
\textbf{Version (mod\`ele)} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Temps (s)} \\
\hline""")
    for k in ordered:
        s = stats[k]
        print(f"{latex_label(k):<38} & {s['prec']*100:.1f}\\% & {s['rec']*100:.1f}\\% "
              f"& {s['f1']*100:.1f}\\% & {s['t']:.2f} \\\\")
    print(r"""\hline
\end{tabular}
\caption{F1 reconnaissance d'objets sur 30 prompts.}
\label{tab:f1-recog}
\end{table}""")


# ---------------------------------------------------------------
# PLACEMENT (relations + orientations)
# ---------------------------------------------------------------

def _plot_placement_group(rel_v, spatial_v, ids, n_runs, model_label, suffix):
    if not rel_v and not spatial_v:
        return

    ordered_rel = sorted(rel_v.keys(), key=version_sort_key)

    # fig 1 : F1 relations par prompt
    if rel_v:
        x = np.arange(len(ids))
        w = 0.8 / len(ordered_rel)
        fig, ax = plt.subplots(figsize=(16, 5))
        for i, label in enumerate(ordered_rel):
            ax.bar(x + i * w - 0.4 + w / 2,
                   [e["rel_f1_mean"] for e in rel_v[label]],
                   width=w, label=label, color=version_color(label), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("F1 relations")
        ax.set_title(f"F1 relations par prompt — {model_label}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
        ax.legend(fontsize=8)
        plt.tight_layout()
        savefig(fig, f"placement_rel_f1_per_form_{suffix}.png")

    # fig 2 : barres globales relations + orientations
    if rel_v:
        rel_stats = {
            k: {"f1": avg([e["rel_f1_mean"]   for e in v]),
                "pre": avg([e["rel_precision"] for e in v]),
                "rec": avg([e["rel_recall"]    for e in v])}
            for k, v in rel_v.items()
        }
        ori_stats = {
            k: {"f1": avg([e["ori_f1_mean"]   for e in v]),
                "pre": avg([e["ori_precision"] for e in v]),
                "rec": avg([e["ori_recall"]    for e in v])}
            for k, v in rel_v.items()
        }
        vs = ordered_rel
        x_pos = np.arange(len(vs))
        fig2, axes = plt.subplots(1, 2, figsize=(max(9, len(vs) * 2 + 3), 5))
        fig2.suptitle(f"Placement — relations et orientations — {model_label}",
                      fontsize=12, fontweight="bold")
        for ax, sts, title in zip(axes, [rel_stats, ori_stats],
                                   ["Relations F1", "Orientations F1"]):
            vals = [sts[v]["f1"] * 100 for v in vs]
            bars = ax.bar(x_pos, vals, color=[version_color(v) for v in vs],
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
        handles = [Patch(facecolor=version_color(v), alpha=0.85, label=v) for v in vs]
        fig2.legend(handles=handles, loc="lower center", ncol=min(len(vs), 4),
                    bbox_to_anchor=(0.5, 0.0), fontsize=9)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        savefig(fig2, f"placement_global_{suffix}.png")

    # fig 3 : coherence spatiale par prompt (V1)
    for label, entries in spatial_v.items():
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
        ax3.set_title(f"{label} — coherence spatiale ({model_label})")
        ax3.legend(fontsize=8)
        plt.tight_layout()
        slug = label.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        savefig(fig3, f"placement_spatial_{slug}_{suffix}.png")


def plot_placement(data, latex=False):
    n_runs = data["n_runs"]

    rel_v = {k: v for k, v in data.items()
             if k != "n_runs" and isinstance(v, list) and v and "rel_f1_mean" in v[0]}
    spa_v = {k: v for k, v in data.items()
             if k != "n_runs" and isinstance(v, list) and v and "quat_valid_rate" in v[0]}

    ids = [e["id"]
           for entries in (list(rel_v.values()) or list(spa_v.values()))[:1]
           for e in entries]

    # tableaux console
    if rel_v:
        rows_rel, rows_ori = [], []
        for label in sorted(rel_v.keys(), key=version_sort_key):
            entries = rel_v[label]
            rows_rel.append([label,
                             f"{avg([e['rel_precision'] for e in entries])*100:.1f}%",
                             f"{avg([e['rel_recall']    for e in entries])*100:.1f}%",
                             f"{avg([e['rel_f1_mean']   for e in entries])*100:.1f}%"])
            rows_ori.append([label,
                             f"{avg([e['ori_precision'] for e in entries])*100:.1f}%",
                             f"{avg([e['ori_recall']    for e in entries])*100:.1f}%",
                             f"{avg([e['ori_f1_mean']   for e in entries])*100:.1f}%"])
        print_table(f"PLACEMENT — RELATIONS  ({n_runs} runs/prompt)",
                    ["Version", "Precision", "Recall", "F1"], rows_rel)
        print_table(f"PLACEMENT — ORIENTATIONS  ({n_runs} runs/prompt)",
                    ["Version", "Precision", "Recall", "F1"], rows_ori)

    for label, entries in spa_v.items():
        print_table(
            f"PLACEMENT — {label} COHERENCE SPATIALE  ({n_runs} runs/prompt)",
            ["Quat valides", "Z >= 0", "Ordre vertical"],
            [[f"{avg([e['quat_valid_rate']    for e in entries])*100:.1f}%",
              f"{avg([e['pos_plausible_rate'] for e in entries])*100:.1f}%",
              f"{avg([e['rel_order_acc_mean'] for e in entries])*100:.1f}%"]],
        )

    # plots par groupe de modele
    all_keys = list(dict.fromkeys(list(rel_v.keys()) + list(spa_v.keys())))
    groups = group_by_model(all_keys)
    for label, suffix, gkey in [("GPT", "gpt", "gpt"), ("Llama", "llama", "llama"),
                                  ("autres", "other", "other")]:
        g_rel = {k: rel_v[k] for k in groups[gkey] if k in rel_v}
        g_spa = {k: spa_v[k] for k in groups[gkey] if k in spa_v}
        _plot_placement_group(g_rel, g_spa, ids, n_runs, label, suffix)

    all_timed = {**rel_v}
    for k, v in spa_v.items():
        if k not in all_timed:
            all_timed[k] = v
    if all_timed:
        _plot_time_combined(
            all_timed,
            lambda e: e.get("time_mean_s", 0.0),
            ids, n_runs,
            f"Temps de placement — toutes versions ({n_runs} runs/prompt)",
            "placement_time_all.png",
        )

    if latex:
        _print_latex_placement(rel_v)


def _print_latex_placement(rel_v):
    ordered = sorted(rel_v.keys(), key=version_sort_key)
    print("\n% ====== Tableau placement (rel_f1 / ori_f1) ======")
    print(r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lcc}
\hline
\textbf{Version (mod\`ele)} & \textbf{rel\_f1} & \textbf{ori\_f1} \\
\hline""")
    for k in ordered:
        entries = rel_v[k]
        rel_f1  = avg([e["rel_f1_mean"] for e in entries]) * 100
        ori_f1  = avg([e["ori_f1_mean"] for e in entries]) * 100
        print(f"{latex_label(k):<38} & {rel_f1:.1f}\\% & {ori_f1:.1f}\\% \\\\")
    print(r"""\hline
\end{tabular}
\caption{F1 sur les relations spatiales et exact match sur les orientations, sur 30 prompts.}
\label{tab:f1-placement}
\end{table}""")


# ---------------------------------------------------------------
# TEMPS TOTAL (recog + placement)
# ---------------------------------------------------------------

def plot_total_time(recog_data, placement_data):
    """Additionne recog + placement par prompt. Pour V2 phi3-scene, prend placement seul."""
    recog_stats    = recog_compute_stats(recog_data)
    placement_keys = [k for k in placement_data if k != "n_runs" and isinstance(placement_data[k], list)]

    ids = [e["id"] for e in next(iter(recog_data["versions"].values()))]

    combined = {}
    all_versions = set(recog_stats.keys()) | set(placement_keys)
    for v in all_versions:
        r_entries = recog_data["versions"].get(v, [])
        p_entries = placement_data.get(v, [])

        if is_v2(v) and not is_gpt(v) and not is_llama(v):
            # phi3 fait recog + placement en un seul appel — prendre placement uniquement
            if p_entries:
                combined[v] = [e.get("time_mean_s", 0.0) for e in p_entries]
        else:
            r_times = [e["time_mean_s"] for e in r_entries] if r_entries else [0.0] * len(ids)
            p_times = [e.get("time_mean_s", 0.0) for e in p_entries] if p_entries else [0.0] * len(ids)
            if r_entries or p_entries:
                combined[v] = [r + p for r, p in zip(r_times, p_times)]

    if not combined:
        return

    ordered = sorted(combined.keys(), key=version_sort_key)
    all_times = [t for v in ordered for t in combined[v]]
    ylim_top  = np.percentile(all_times, 95) * 1.4 if all_times else 10.0

    fig, ax = plt.subplots(figsize=(14, 4))
    for v in ordered:
        times = combined[v]
        ax.plot(ids, times, marker="o", label=v, color=version_color(v),
                linewidth=1.5, markersize=4)
        for xi, t in enumerate(times):
            if t > ylim_top:
                ax.text(xi, ylim_top * 0.97, f"^{t:.0f}s",
                        ha="center", va="top", fontsize=7,
                        color=version_color(v), fontweight="bold")
    ax.set_ylim(0, ylim_top)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Temps total (s)")
    ax.set_title("Temps total pipeline (reconnaissance + placement) — toutes versions")
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig, "total_time_all.png")


# ---------------------------------------------------------------
# TIMING
# ---------------------------------------------------------------

def timing_compute_stats(data):
    stats = {}
    for v, entries in data["versions"].items():
        gm  = avg([e["mean_s"] for e in entries])
        gsd = avg([e["std_s"]  for e in entries])
        stats[v] = {
            "mean": gm,
            "std":  gsd,
            "mn":   min(e["min_s"] for e in entries),
            "mx":   max(e["max_s"] for e in entries),
            "entries": entries,
        }
    return stats


def _plot_timing_group(data, gstats, group_keys, model_label, suffix):
    if not group_keys:
        return

    ordered = sorted(group_keys, key=version_sort_key)
    ids     = [e["id"] for e in data["versions"][ordered[0]]]

    # fig 1 : bar global + errorbars
    fig, ax = plt.subplots(figsize=(max(6, len(ordered) * 1.8 + 2), 5))
    means_ = [gstats[v]["mean"] for v in ordered]
    stds_  = [gstats[v]["std"]  for v in ordered]
    bars   = ax.bar(range(len(ordered)), means_, yerr=stds_, capsize=6,
                    color=[version_color(v) for v in ordered], width=0.5, alpha=0.85,
                    hatch=["//" if is_boucle(v) else "" for v in ordered])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Temps (s)")
    ax.set_title(f"Temps moyen — pipeline complet — {model_label}")
    for bar, m, s in zip(bars, means_, stds_):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.3,
                f"{m:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"timing_global_{suffix}.png")

    # fig 2 : courbes par prompt
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for v in ordered:
        entries = data["versions"][v]
        ax2.errorbar(ids, [e["mean_s"] for e in entries],
                     yerr=[e["std_s"] for e in entries],
                     label=v, color=version_color(v),
                     marker="o", linewidth=1.5, markersize=4, capsize=3)
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Temps (s)")
    ax2.set_title(f"Temps pipeline complet par prompt — {model_label}")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    savefig(fig2, f"timing_per_form_{suffix}.png")

    # fig 3 : boxplot
    fig3, ax3 = plt.subplots(figsize=(max(6, len(ordered) * 1.8 + 2), 5))
    all_runs = [[t for e in data["versions"][v] for t in e["runs"]] for v in ordered]
    bp = ax3.boxplot(all_runs, tick_labels=ordered, patch_artist=True)
    for patch, v in zip(bp["boxes"], ordered):
        patch.set_facecolor(version_color(v))
        patch.set_alpha(0.7)
    ax3.set_xticklabels(ordered, rotation=20, ha="right", fontsize=9)
    ax3.set_ylabel("Temps (s)")
    ax3.set_title(f"Distribution des temps — {model_label}")
    plt.tight_layout()
    savefig(fig3, f"timing_boxplot_{suffix}.png")


def plot_timing(data, latex=False):
    n_runs  = data["n_runs"]
    gstats  = timing_compute_stats(data)
    versions = list(gstats.keys())

    rows = []
    for v in sorted(versions, key=version_sort_key):
        s = gstats[v]
        rows.append([v, f"{s['mean']:.2f}s", f"{s['std']:.2f}s",
                     f"{s['mn']:.2f}s", f"{s['mx']:.2f}s"])
    print_table(f"PIPELINE COMPLET — TEMPS  ({n_runs} runs/prompt)",
                ["Version", "Moy", "Std", "Min", "Max"], rows)

    groups = group_by_model(versions)
    for label, suffix, gkey in [("GPT", "gpt", "gpt"), ("Llama", "llama", "llama"),
                                  ("autres", "other", "other")]:
        g = [k for k in groups[gkey] if k in gstats]
        _plot_timing_group(data, gstats, g, label, suffix)

    if latex:
        _print_latex_timing(gstats)


def _print_latex_timing(gstats):
    ordered = sorted(gstats.keys(), key=version_sort_key)
    print("\n% ====== Tableau latence ======")
    print(r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lccc}
\hline
\textbf{Version (mod\`ele)} & \textbf{Moyenne (s)} & \textbf{Min (s)} & \textbf{Max (s)} \\
\hline""")
    for k in ordered:
        s = gstats[k]
        print(f"{latex_label(k):<38} & {s['mean']:.2f} & {s['mn']:.2f} & {s['mx']:.2f} \\\\")
    print(r"""\hline
\end{tabular}
\caption{Latence d'ex\'ecution end-to-end du pipeline texte, sur 30 prompts $\times$ 5 ex\'ecutions par prompt.}
\label{tab:latence-texte}
\end{table}""")


# ---------------------------------------------------------------
# VISION (V3, V3.1)
# ---------------------------------------------------------------

def _vision_label(data):
    v = data.get("version", "?")
    return "V3" if v == "v3" else "V3.1" if v == "v3.1" else v


def plot_vision(v3_data, v31_data, latex=False):
    versions = []
    if v3_data:  versions.append(("V3",   v3_data))
    if v31_data: versions.append(("V3.1", v31_data))
    if not versions:
        return

    ids = [r["id"] for r in versions[0][1]["results"]]

    # tableau console global
    rows = []
    for name, d in versions:
        f1 = d.get("global_f1_mean", 0.0)
        mae = d.get("global_mae_mean", {})
        rel_f1 = avg([r["relation_f1"]["f1"] for r in d["results"]])
        t = avg([r["time_mean_s"] for r in d["results"]])
        rows.append([name,
                     f"{f1*100:.1f}%",
                     f"{rel_f1*100:.1f}%",
                     f"{mae.get('x') or 0:.3f}",
                     f"{mae.get('y') or 0:.3f}",
                     f"{mae.get('z') or 0:.3f}",
                     f"{t:.2f}s"])
    print_table("VISION (V3, V3.1)",
                ["Version", "F1 obj", "F1 rel", "MAE x", "MAE y", "MAE z", "Temps"], rows)

    # fig 1 : F1 objets + F1 relations par image
    x = np.arange(len(ids))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, key, title in zip(axes,
                              [lambda r: r["aggregated_f1"].get("f1_mean", 0),
                               lambda r: r["relation_f1"]["f1"]],
                              ["F1 objets", "F1 relations"]):
        for i, (name, d) in enumerate(versions):
            vals = [key(r) for r in d["results"]]
            ax.bar(x + (i - len(versions)/2 + 0.5) * w, vals, width=w,
                   color=version_color(name), alpha=0.85, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(title)
        ax.set_title(f"{title} par image")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
        ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig, "vision_f1_per_image.png")

    # fig 2 : MAE par axe (barres groupees) + RMSE
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    axes_lbl = ["MAE x", "MAE y", "MAE z", "RMSE"]
    bw = 0.35
    xp = np.arange(len(axes_lbl))
    for i, (name, d) in enumerate(versions):
        results = d["results"]
        vals = [
            avg([r["mae_mean"]["x"] for r in results if r["mae_mean"]["x"] is not None] or [0]),
            avg([r["mae_mean"]["y"] for r in results if r["mae_mean"]["y"] is not None] or [0]),
            avg([r["mae_mean"]["z"] for r in results if r["mae_mean"]["z"] is not None] or [0]),
            avg([r["mae_mean"]["rmse"] for r in results if r["mae_mean"]["rmse"] is not None] or [0]),
        ]
        bars = ax2.bar(xp + (i - len(versions)/2 + 0.5) * bw, vals, width=bw,
                       color=version_color(name), alpha=0.85, label=name)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(xp)
    ax2.set_xticklabels(axes_lbl)
    ax2.set_ylabel("Erreur (m)")
    ax2.set_title("Erreur moyenne sur les positions")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig2, "vision_mae_global.png")

    # fig 3 : F1 global + temps
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.5))
    names = [n for n, _ in versions]
    f1s   = [d.get("global_f1_mean", 0)*100 for _, d in versions]
    rels  = [avg([r["relation_f1"]["f1"] for r in d["results"]])*100 for _, d in versions]
    times = [avg([r["time_mean_s"] for r in d["results"]]) for _, d in versions]

    xp = np.arange(len(names))
    bw = 0.35
    axes3[0].bar(xp - bw/2, f1s, width=bw, color=[version_color(n) for n in names],
                 alpha=0.85, label="F1 objets")
    axes3[0].bar(xp + bw/2, rels, width=bw, color=[version_color(n) for n in names],
                 alpha=0.5, hatch="//", label="F1 relations")
    for i, (a, b) in enumerate(zip(f1s, rels)):
        axes3[0].text(i - bw/2, a + 2, f"{a:.1f}", ha="center", fontsize=8)
        axes3[0].text(i + bw/2, b + 2, f"{b:.1f}", ha="center", fontsize=8)
    axes3[0].set_xticks(xp)
    axes3[0].set_xticklabels(names)
    axes3[0].set_ylim(0, 115)
    axes3[0].set_ylabel("%")
    axes3[0].set_title("F1 global")
    axes3[0].legend(fontsize=9)
    axes3[0].axhline(100, color="gray", linestyle="--", linewidth=0.7)

    bars = axes3[1].bar(xp, times, color=[version_color(n) for n in names], width=0.45, alpha=0.85)
    for bar, t in zip(bars, times):
        axes3[1].text(bar.get_x() + bar.get_width()/2, t + 0.3,
                      f"{t:.2f}s", ha="center", fontsize=9, fontweight="bold")
    axes3[1].set_xticks(xp)
    axes3[1].set_xticklabels(names)
    axes3[1].set_ylabel("Temps moyen (s)")
    axes3[1].set_title("Latence")
    plt.tight_layout()
    savefig(fig3, "vision_global.png")

    # fig 4 : box plot par axe (x, y, z, rmse) avec scatter overlay des points par run
    # Chaque point = une valeur per_run d'une image. Montre la dispersion vs juste la moyenne.
    fig4, axes4 = plt.subplots(1, 4, figsize=(14, 4.5))
    axes_keys = [("mae_x", "MAE x (m)"), ("mae_y", "MAE y (m)"),
                 ("mae_z", "MAE z (m)"), ("rmse",  "RMSE (m)")]
    rng = np.random.default_rng(42)
    for ax, (key, title) in zip(axes4, axes_keys):
        data_by_version = []
        for name, d in versions:
            vals = []
            for r in d["results"]:
                for run in r.get("per_run", []):
                    v = run.get(key)
                    if v is not None:
                        vals.append(v)
            data_by_version.append(vals)
        positions = np.arange(1, len(versions) + 1)
        bp = ax.boxplot(data_by_version, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, (name, _) in zip(bp["boxes"], versions):
            patch.set_facecolor(version_color(name))
            patch.set_alpha(0.5)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        # scatter overlay (jitter horizontal)
        for pos, vals, (name, _) in zip(positions, data_by_version, versions):
            if vals:
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals,
                           color=version_color(name), edgecolor="black",
                           linewidth=0.4, s=22, alpha=0.75, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([name for name, _ in versions])
        ax.set_title(title)
        ax.set_ylabel("Erreur (m)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig4.suptitle("Distribution des erreurs de position (box plot + points par run)",
                  fontsize=11)
    plt.tight_layout()
    savefig(fig4, "vision_mae_boxplot.png")

    # fig 5 : box plot des F1 par image (un point = un run d'une image)
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    f1_data = []
    for name, d in versions:
        vals = []
        for r in d["results"]:
            for run in r.get("per_run", []):
                vals.append(run.get("f1", 0))
        f1_data.append(vals)
    positions = np.arange(1, len(versions) + 1)
    bp = ax5.boxplot(f1_data, positions=positions, widths=0.55,
                     patch_artist=True, showfliers=False)
    for patch, (name, _) in zip(bp["boxes"], versions):
        patch.set_facecolor(version_color(name))
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    for pos, vals, (name, _) in zip(positions, f1_data, versions):
        if vals:
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax5.scatter(np.full(len(vals), pos) + jitter, vals,
                        color=version_color(name), edgecolor="black",
                        linewidth=0.4, s=24, alpha=0.75, zorder=3)
    ax5.set_xticks(positions)
    ax5.set_xticklabels([name for name, _ in versions])
    ax5.set_ylabel("F1 objets")
    ax5.set_title("Distribution du F1 objets par run")
    ax5.set_ylim(-0.05, 1.1)
    ax5.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    ax5.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    savefig(fig5, "vision_f1_boxplot.png")

    if latex:
        print("\n% ====== Tableau vision ======")
        print(r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lcccccc}
\hline
\textbf{Version} & \textbf{F1 obj} & \textbf{F1 rel} & \textbf{MAE x (m)} & \textbf{MAE y (m)} & \textbf{MAE z (m)} & \textbf{Temps (s)} \\
\hline""")
        for name, d in versions:
            f1 = d.get("global_f1_mean", 0) * 100
            mae = d.get("global_mae_mean", {})
            rel_f1 = avg([r["relation_f1"]["f1"] for r in d["results"]]) * 100
            t = avg([r["time_mean_s"] for r in d["results"]])
            print(f"{name} & {f1:.1f}\\% & {rel_f1:.1f}\\% "
                  f"& {mae.get('x') or 0:.3f} & {mae.get('y') or 0:.3f} & {mae.get('z') or 0:.3f} "
                  f"& {t:.2f} \\\\")
        print(r"""\hline
\end{tabular}
\caption{Resultats des pipelines vision V3 et V3.1.}
\label{tab:vision}
\end{table}""")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["recog", "placement", "timing", "vision"], default=None)
    parser.add_argument("--latex",   action="store_true",
                        help="Affiche aussi les blocs LaTeX copiables")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    tasks = {
        "recog":     (RECOG_PATH,     plot_recognition, "python tests/test_recognition.py"),
        "placement": (PLACEMENT_PATH, plot_placement,   "python tests/test_placement.py"),
        "timing":    (TIMING_PATH,    plot_timing,      "python tests/benchmark_timing.py"),
    }

    to_run = [args.only] if args.only and args.only != "vision" else list(tasks.keys())

    loaded = {}
    if not args.only or args.only != "vision":
        for key in to_run:
            path, fn, hint = tasks[key]
            if not os.path.exists(path):
                print(f"[{key}] Fichier introuvable : {path}")
                print(f"         Lance d'abord : {hint}")
            else:
                data = load(path)
                loaded[key] = data
                fn(data, latex=args.latex)

        if "recog" in loaded and "placement" in loaded:
            plot_total_time(loaded["recog"], loaded["placement"])

    if args.only is None or args.only == "vision":
        v3_data  = load(VISION_V3_PATH)  if os.path.exists(VISION_V3_PATH)  else None
        v31_data = load(VISION_V31_PATH) if os.path.exists(VISION_V31_PATH) else None
        if v3_data or v31_data:
            plot_vision(v3_data, v31_data, latex=args.latex)
        else:
            print("[vision] Aucun fichier de resultat trouve")
            print("         Lance d'abord : python tests/test_vision.py --version all")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
