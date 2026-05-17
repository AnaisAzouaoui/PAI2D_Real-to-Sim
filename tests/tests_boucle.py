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
import genesis as gs
from simulation.simulationGenesis import make_morph
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.versions.v2_finetuned.pipeline_v2 import (object_rec as object_rec_v2, place_scene, phi3_cache,)
from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1_1 import object_rec as object_rec_v111
from pipeline.versions.v1_llm_only.pipeline_v1_llm import object_dim_quat
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
        ent = scene.add_entity(make_morph(obj['path'], pos=tuple(obj['pos']), quat=tuple(obj.get('quat', [0.0, 1.0, 1.0, 0.0])), scale=obj.get('scale', 1.0), fixed=True),
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
    result = subprocess.run([sys.executable, RUN_VALIDATION, "prompt", scene_json_path, prompt], cwd=SRC, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Validation subprocess failed with code {result.returncode}")

def run_v111(prompt):
    phi3_cache.pop(prompt, None)
    obj_reconnus, non_reconnus = object_rec_v111(prompt)
    if not obj_reconnus:
        raise ValueError(f"Aucun objet reconnu (v1.1.1) pour: {prompt!r}")
    if non_reconnus:
        print(f"  [v1.1.1] objets non reconnus (ignores): {non_reconnus}")
    items = place_scene(prompt, obj_reconnus, build_scene_fn=build_scene_subprocess)
    items = postprocess_objects(items, obj_reconnus)
    return items

def process_version(pid, prompt, version_label, run_fn):
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
        #avant vallidation collisions
        collisions_before = count_collisions(items)
        print(f"[{pid}/{version_label}] collisions avant: {collisions_before}")

        # validation
        print(f"[{pid}/{version_label}] boucle validation")
        run_validation_subprocess(scene_path, prompt)

        # apres la validation
        with open(scene_path, 'r', encoding='utf-8') as f:
            final_items = json.load(f)
        collisions_after = count_collisions(final_items)
        print(f"[{pid}/{version_label}] collisions apres: {collisions_after}")

        # convergence
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

        # metriques
        metrics = {"id": pid, "prompt": prompt, "version": version_label,
        "collisions_before": collisions_before,"collisions_after": collisions_after,
        "iterations": iterations,"converged": converged,}
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
    ax.bar([i + w/2 for i in x], [m["collisions_after"]  for m in all_metrics], w, label="Après",  color="#D85A30")
    ax.set_xticks(list(x)); 
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("Collisions"); 
    ax.set_title("Collisions avant / après validation")
    ax.legend(); 
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "collisions.png"), dpi=150)
    plt.close(fig)

    # nb d'iter
    iters = [m["iterations"] if m["iterations"] is not None else 0 for m in all_metrics]
    fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.8), 5))
    ax.plot(list(x), iters, marker="o", color="#1D9E75", linewidth=2)
    ax.fill_between(list(x), iters, alpha=0.15, color="#1D9E75")
    ax.set_xticks(list(x)); 
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel("Itérations"); 
    ax.set_title("Nombre d'itérations par prompt")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "iterations.png"), dpi=150)
    plt.close(fig)

    # convergence
    conv = []
    for i, m in enumerate(all_metrics):
        sl = [x for x in all_metrics[:i+1] if x["converged"] is not None]
        conv.append(sum(1 for x in sl if x["converged"]) / len(sl) * 100 if sl else 0)
    fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.8), 5))
    ax.plot(list(x), conv, marker="o", color="#BA7517", linewidth=2)
    ax.fill_between(list(x), conv, alpha=0.15, color="#BA7517")
    ax.set_xticks(list(x)); ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylim(0, 105); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Taux de convergence cumulé"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close(fig)

    print(f"plots sauvegardés dans {out_dir}/")



def main():
    parser = argparse.ArgumentParser(description="Genere les images formulaire et les metriques de validation pour les prompts")
    parser.add_argument("--ids", nargs="+", help="IDs a traiter (ex: p01 p02), tous par defaut")
    parser.add_argument("--versions", nargs="+", choices=["v1"], default=["v1"], help="Versions a executer (defaut: v1)")
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

    version_fns = {"v1": run_v111}

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
            success, metrics = process_version(pid, prompt, version_label,version_fns[version_label]
            )
            if success:
                ok_count += 1
                if metrics:
                    all_metrics.append(metrics)
            else:
                err_count += 1

    #sauvegarde des métriques
    if all_metrics:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[metrics] {len(all_metrics)} resultats sauvegardes dans {METRICS_FILE}")

        print("\n" + "="*60)
        print("RÉSUMÉ DES MÉTRIQUES")
        print("="*60)
        valid = [m for m in all_metrics if m['collisions_before'] is not None]
        if valid:
            avg_before = sum(m['collisions_before'] for m in valid) / len(valid)
            avg_after  = sum(m['collisions_after'] for m in valid) / len(valid)
            no_coll_before = sum(1 for m in valid if m['collisions_before'] == 0) / len(valid) * 100
            no_coll_after  = sum(1 for m in valid if m['collisions_after'] == 0) / len(valid) * 100
            iterations_list = [m['iterations'] for m in valid if m['iterations'] is not None]
            converged_list  = [m['converged'] for m in valid if m['converged'] is not None]
            avg_iter = sum(iterations_list)/len(iterations_list) if iterations_list else 0
            median_iter = sorted(iterations_list)[len(iterations_list)//2] if iterations_list else 0
            converged_rate = sum(1 for c in converged_list if c) / len(converged_list) * 100 if converged_list else 0
            print(f"Taux moyen de collisions avant: {avg_before:.2f}")
            print(f"Taux moyen de collisions après: {avg_after:.2f}")
            print(f"Scènes sans collision avant: {no_coll_before:.1f}%")
            print(f"Scènes sans collision après: {no_coll_after:.1f}%")
            print(f"Nombre moyen d'itérations: {avg_iter:.2f}")
            print(f"Médiane des itérations: {median_iter}")
            print(f"Taux de convergence: {converged_rate:.1f}%")

            summary = {
                "total_prompts": len(all_metrics),
                "avg_collisions_before": avg_before,
                "avg_collisions_after": avg_after,
                "pct_no_collision_before": no_coll_before,
                "pct_no_collision_after": no_coll_after,
                "avg_iterations": avg_iter,
                "median_iterations": median_iter,
                "convergence_rate": converged_rate,}
            with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"[metrics] résumé sauvegardé dans {SUMMARY_FILE}")

            plot_metrics(all_metrics, os.path.join(ROOT, "tests", "plots"))

    total = ok_count + err_count
    print(f"\n[generate] Termine: {ok_count}/{total} succes, {err_count} erreurs")

if __name__ == "__main__":
    main()