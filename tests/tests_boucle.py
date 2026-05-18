"""
python tests/tests_boucle.py --exclude table
(memes commandes que pour le truc de generation d'images formulaire)
"""
import sys
import os
import json
import subprocess
import tempfile
import traceback
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import genesis as gs
from simulation.simulationGenesis import make_morph
from pipeline.versions.v2_finetuned.pipeline_v2 import object_rec as object_rec_v2, place_scene
from pipeline_worker import build_scene_subprocess, postprocess_objects

DATA_PATH = os.path.join(ROOT, "tests", "data", "prompts_ground_truth.json")
IMAGES_BASE_DIR = os.path.join(ROOT, "tests", "images_formulaire")
RUN_SCREENSHOTS = os.path.join(SRC, "run_screenshots.py")
RUN_VALIDATION = os.path.join(SRC, "run_validation.py")
METRICS_FILE = os.path.join(ROOT, "tests", "validation_metrics.json")
SUMMARY_FILE = os.path.join(ROOT, "tests", "validation_summary.json")


def load_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["prompts"]


def save_tmp_scene(objects_list):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(objects_list, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


def count_collisions(objetsList):
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    entities = []
    for obj in objetsList:
        ent = scene.add_entity(
            make_morph(obj['path'], pos=tuple(obj['pos']),
                       quat=tuple(obj.get('quat', [0.0, 0.0, 0.0, 1.0])),
                       scale=obj.get('scale', 1.0), fixed=True),
            material=gs.materials.Rigid(rho=1000))
        entities.append(ent)
    scene.build()
    aabbs = [ent.get_AABB() for ent in entities]
    gs.destroy()
    collisions = 0
    n = len(aabbs)
    for i in range(n):
        for j in range(i+1, n):
            amin, amax = aabbs[i]
            bmin, bmax = aabbs[j]
            ox = min(amax[0], bmax[0]) - max(amin[0], bmin[0])
            oy = min(amax[1], bmax[1]) - max(amin[1], bmin[1])
            oz = min(amax[2], bmax[2]) - max(amin[2], bmin[2])
            if ox > 0 and oy > 0 and oz > 0:
                collisions += 1
    return collisions


def run_validation_subprocess(scene_json_path, prompt):
    result = subprocess.run(
        [sys.executable, RUN_VALIDATION, "prompt", scene_json_path, prompt],
        cwd=SRC, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Validation subprocess failed with code {result.returncode}")


def run_v2(prompt):
    obj_reconnus, non_reconnus = object_rec_v2(prompt)
    if not obj_reconnus:
        raise ValueError(f"Aucun objet reconnu (v2) pour: {prompt!r}")
    if non_reconnus:
        print(f"  [v2] objets non reconnus (ignores): {non_reconnus}")
    items = place_scene(prompt, obj_reconnus, build_scene_fn=build_scene_subprocess)
    items = postprocess_objects(items, obj_reconnus)
    return items


# ---------------------------------------------------------------
# EVALUATION PLACEMENT GROUND TRUTH
# ---------------------------------------------------------------

def check_relation(rel_type, subject, ref):
    sx, sy, sz = subject['pos'][0], subject['pos'][1], subject['pos'][2]
    rx, ry, rz = ref['pos'][0], ref['pos'][1], ref['pos'][2]
    if rel_type == 'on':
        return sz > rz
    elif rel_type == 'under':
        return sz < rz
    elif rel_type == 'right_of':
        return sy > ry
    elif rel_type == 'left_of':
        return sy < ry
    elif rel_type == 'in_front_of':
        return sx > rx
    elif rel_type == 'behind':
        return sx < rx
    elif rel_type in ('inside', 'against'):
        return True  # geometrie trop complexe pour etre verifiee simplement
    return False


def match_gt_to_items(ground_truth_entry, items):
    """Associe les IDs du ground truth aux items de la scene.
    Essaie d'abord un match direct par ID, puis par URDF."""
    items_by_id = {item['id']: item for item in items}
    items_by_urdf = {}
    for item in items:
        items_by_urdf.setdefault(item.get('urdf', ''), []).append(item)

    id_map = {}
    urdf_usage = {}
    for exp_obj in ground_truth_entry.get('expected_objects', []):
        gt_id = exp_obj['id']
        urdf = exp_obj['urdf']
        if gt_id in items_by_id:
            id_map[gt_id] = items_by_id[gt_id]
        else:
            candidates = items_by_urdf.get(urdf, [])
            used = urdf_usage.get(urdf, 0)
            if used < len(candidates):
                id_map[gt_id] = candidates[used]
                urdf_usage[urdf] = used + 1
    return id_map


def evaluate_placement(items, ground_truth_entry):
    """Evalue le pourcentage de relations spatiales du ground truth respectees.
    Retourne (score 0-1, nb correct, nb total evaluable)."""
    id_map = match_gt_to_items(ground_truth_entry, items)
    relations = ground_truth_entry.get('expected_relations', [])
    evaluable = [
        r for r in relations
        if r['subject'] in id_map and r['object'] in id_map
    ]
    if not evaluable:
        return None, 0, 0
    ok = sum(
        1 for r in evaluable
        if check_relation(r['type'], id_map[r['subject']], id_map[r['object']])
    )
    return ok / len(evaluable), ok, len(evaluable)


# ---------------------------------------------------------------
# BOUCLE PRINCIPALE
# ---------------------------------------------------------------

def process_version(pid, prompt, version_label, run_fn, ground_truth_entry=None):
    out_dir = os.path.join(IMAGES_BASE_DIR, pid, version_label)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  [{pid}/{version_label}] pipeline...")
    try:
        items = run_fn(prompt)
    except Exception as e:
        print(f"  [{pid}/{version_label}] ERREUR pipeline: {e}")
        traceback.print_exc()
        return False, None
    scene_path = save_tmp_scene(items)
    try:
        collisions_before = count_collisions(items)
        print(f"[{pid}/{version_label}] collisions avant: {collisions_before}")

        placement_before, placement_ok_before, placement_total = None, 0, 0
        if ground_truth_entry:
            placement_before, placement_ok_before, placement_total = evaluate_placement(items, ground_truth_entry)
            print(f"[{pid}/{version_label}] placement avant: {placement_ok_before}/{placement_total} ({placement_before:.0%})" if placement_before is not None else f"[{pid}/{version_label}] placement avant: N/A")

        print(f"[{pid}/{version_label}] boucle validation")
        run_validation_subprocess(scene_path, prompt)

        with open(scene_path, 'r', encoding='utf-8') as f:
            final_items = json.load(f)
        collisions_after = count_collisions(final_items)
        print(f"[{pid}/{version_label}] collisions apres: {collisions_after}")

        placement_after = None
        if ground_truth_entry:
            placement_after, placement_ok_after, _ = evaluate_placement(final_items, ground_truth_entry)
            print(f"[{pid}/{version_label}] placement apres: {placement_ok_after}/{placement_total} ({placement_after:.0%})" if placement_after is not None else f"[{pid}/{version_label}] placement apres: N/A")

        iterations = None
        converged = False
        runs_dir = os.path.join(ROOT, "runs")
        if os.path.exists(runs_dir):
            run_dirs = sorted(os.listdir(runs_dir), reverse=True)
            if run_dirs:
                latest_run = os.path.join(runs_dir, run_dirs[0])
                history_path = os.path.join(latest_run, "history.json")
                if os.path.exists(history_path):
                    with open(history_path, encoding='utf-8') as f:
                        history = json.load(f)
                    iterations = len(history)
                    converged = history[-1].get('valid', False)
                    print(f"[{pid}/{version_label}] iterations: {iterations}, converged: {converged}")

        metrics = {
            "id": pid,
            "prompt": prompt,
            "version": version_label,
            "collisions_before": collisions_before,
            "collisions_after": collisions_after,
            "placement_score_before": placement_before,
            "placement_score_after": placement_after,
            "placement_relations_total": placement_total,
            "iterations": iterations,
            "converged": converged,
        }
        print(f"[{pid}/{version_label}] OK")
        return True, metrics

    except Exception as e:
        print(f"  [{pid}/{version_label}] ERREUR rendu/validation: {e}")
        traceback.print_exc()
        return False, None
    finally:
        try:
            os.unlink(scene_path)
        except OSError:
            pass


def plot_metrics(all_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ids = [m["id"] for m in all_metrics]
    x = range(len(ids))

    # collisions
    fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.8), 5))
    w = 0.35
    ax.bar([i - w/2 for i in x], [m["collisions_before"] for m in all_metrics], w, label="Avant", color="#378ADD")
    ax.bar([i + w/2 for i in x], [m["collisions_after"]  for m in all_metrics], w, label="Apres",  color="#D85A30")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("Collisions")
    ax.set_title("Collisions avant / apres validation")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collisions.png"), dpi=150)
    plt.close(fig)

    # placement score
    scores_before = [m["placement_score_before"] if m["placement_score_before"] is not None else 0 for m in all_metrics]
    scores_after  = [m["placement_score_after"]  if m["placement_score_after"]  is not None else 0 for m in all_metrics]
    fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.8), 5))
    ax.bar([i - w/2 for i in x], [s * 100 for s in scores_before], w, label="Avant", color="#378ADD")
    ax.bar([i + w/2 for i in x], [s * 100 for s in scores_after],  w, label="Apres",  color="#D85A30")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("Relations correctes (%)")
    ax.set_title("Score placement (relations GT) avant / apres validation")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "placement_score.png"), dpi=150)
    plt.close(fig)

    # nb d'iterations
    iters = [m["iterations"] if m["iterations"] is not None else 0 for m in all_metrics]
    fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.8), 5))
    ax.plot(list(x), iters, marker="o", color="#1D9E75", linewidth=2)
    ax.fill_between(list(x), iters, alpha=0.15, color="#1D9E75")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("Iterations")
    ax.set_title("Nombre d'iterations par prompt")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "iterations.png"), dpi=150)
    plt.close(fig)

    # convergence cumulee
    conv = []
    for i, m in enumerate(all_metrics):
        sl = [m2 for m2 in all_metrics[:i+1] if m2["converged"] is not None]
        conv.append(sum(1 for m2 in sl if m2["converged"]) / len(sl) * 100 if sl else 0)
    fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.8), 5))
    ax.plot(list(x), conv, marker="o", color="#BA7517", linewidth=2)
    ax.fill_between(list(x), conv, alpha=0.15, color="#BA7517")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Taux de convergence cumule")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close(fig)

    print(f"plots sauvegardes dans {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Genere les images formulaire et les metriques de validation pour les prompts")
    parser.add_argument("--ids", nargs="+", help="IDs a traiter (ex: p01 p02), tous par defaut")
    parser.add_argument("--versions", nargs="+", choices=["v2"], default=["v2"], help="Versions a executer (defaut: v2)")
    parser.add_argument("--no-screenshots", action="store_true", help="Ne pas afficher le chemin des images")
    parser.add_argument("--no-metrics", action="store_true", help="Ne pas afficher le chemin des metriques")
    parser.add_argument("--exclude", nargs="+", help="Exclure les prompts contenant ces mots (ex: table chaise)")
    args = parser.parse_args()

    prompts = load_prompts()
    if args.ids:
        prompts = [p for p in prompts if p["id"] in args.ids]
    if args.exclude:
        prompts = [p for p in prompts if not any(w.lower() in p["prompt"].lower() for w in args.exclude)]

    print(f"[generate] {len(prompts)} prompt(s) x {len(args.versions)} version(s)")
    if not args.no_screenshots:
        print(f"[generate] images -> {IMAGES_BASE_DIR}")
    if not args.no_metrics:
        print(f"[generate] metriques -> {METRICS_FILE}")

    version_fns = {"v2": run_v2}

    ok_count  = 0
    err_count = 0
    all_metrics = []

    for entry in prompts:
        pid    = entry["id"]
        prompt = entry["prompt"]
        print(f"\n{'='*60}")
        print(f"  {pid}: {prompt[:70]}")
        print(f"{'='*60}")

        for version_label in args.versions:
            success, metrics = process_version(
                pid, prompt, version_label, version_fns[version_label],
                ground_truth_entry=entry,
            )
            if success:
                ok_count += 1
                if metrics:
                    all_metrics.append(metrics)
            else:
                err_count += 1

    if all_metrics:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[metrics] {len(all_metrics)} resultats sauvegardes dans {METRICS_FILE}")

        print("\n" + "="*60)
        print("RESUME DES METRIQUES")
        print("="*60)
        valid = [m for m in all_metrics if m['collisions_before'] is not None]
        if valid:
            avg_before = sum(m['collisions_before'] for m in valid) / len(valid)
            avg_after  = sum(m['collisions_after']  for m in valid) / len(valid)
            no_coll_before = sum(1 for m in valid if m['collisions_before'] == 0) / len(valid) * 100
            no_coll_after  = sum(1 for m in valid if m['collisions_after']  == 0) / len(valid) * 100
            iterations_list = [m['iterations'] for m in valid if m['iterations'] is not None]
            converged_list  = [m['converged']  for m in valid if m['converged']  is not None]
            placement_before_list = [m['placement_score_before'] for m in valid if m['placement_score_before'] is not None]
            placement_after_list  = [m['placement_score_after']  for m in valid if m['placement_score_after']  is not None]
            avg_iter = sum(iterations_list) / len(iterations_list) if iterations_list else 0
            median_iter = sorted(iterations_list)[len(iterations_list)//2] if iterations_list else 0
            converged_rate = sum(1 for c in converged_list if c) / len(converged_list) * 100 if converged_list else 0
            avg_placement_before = sum(placement_before_list) / len(placement_before_list) * 100 if placement_before_list else 0
            avg_placement_after  = sum(placement_after_list)  / len(placement_after_list)  * 100 if placement_after_list  else 0

            print(f"Collisions moyennes avant:           {avg_before:.2f}")
            print(f"Collisions moyennes apres:           {avg_after:.2f}")
            print(f"Scenes sans collision avant:         {no_coll_before:.1f}%")
            print(f"Scenes sans collision apres:         {no_coll_after:.1f}%")
            print(f"Score placement moyen avant:         {avg_placement_before:.1f}%")
            print(f"Score placement moyen apres:         {avg_placement_after:.1f}%")
            print(f"Nombre moyen d'iterations:           {avg_iter:.2f}")
            print(f"Mediane des iterations:              {median_iter}")
            print(f"Taux de convergence:                 {converged_rate:.1f}%")

            summary = {
                "total_prompts": len(all_metrics),
                "avg_collisions_before": avg_before,
                "avg_collisions_after": avg_after,
                "pct_no_collision_before": no_coll_before,
                "pct_no_collision_after": no_coll_after,
                "avg_placement_score_before": avg_placement_before,
                "avg_placement_score_after": avg_placement_after,
                "avg_iterations": avg_iter,
                "median_iterations": median_iter,
                "convergence_rate": converged_rate,
            }
            with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[metrics] resume sauvegarde dans {SUMMARY_FILE}")

            plot_metrics(all_metrics, os.path.join(ROOT, "tests", "plots"))

    total = ok_count + err_count
    print(f"\n[generate] Termine: {ok_count}/{total} succes, {err_count} erreurs")

if __name__ == "__main__":
    main()
