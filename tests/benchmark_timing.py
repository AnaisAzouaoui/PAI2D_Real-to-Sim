"""
Benchmark du pipeline complet (de la reconnaissance jusqu'a la generation de scene).
Mesure les temps d'execution sans boucle (5 runs) et avec boucle de validation (1 run).

V1.1: object_rec (LLM) + object_relations + orientation + buildScene (Genesis)
V2  : phi3-scene + object_rec (embedding) + place_scene (Genesis)
V1  : object_rec (embedding) + object_dim_quat (LLM, pas de Genesis)

Usage:
  python tests/benchmark_timing.py
  python tests/benchmark_timing.py --runs 3 --tag gpt-4o --model gpt-4o
  python tests/benchmark_timing.py --runs 3 --tag llama3.1 --model llama3.1
  python tests/benchmark_timing.py --no-boucle   # desactive les variantes avec boucle
"""
import sys
import os
import json
import time
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1 import object_rec as rec_v1_1
from pipeline.versions.v1_1_llm_and_primitives.placement.placement_v1_relations import object_relations
from pipeline.versions.v1_1_llm_and_primitives.placement.placement_v1_distances_orientations import orientation as extract_orientation_v1_1
from pipeline.sceneBuilding import buildScene

from pipeline.versions.v2_finetuned.pipeline_v2 import object_rec as rec_v2, place_scene, phi3_cache

from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1_1 import object_rec as rec_v1_1_1
from pipeline.versions.v1_llm_only.pipeline_v1_llm import object_dim_quat

try:
    from boucle_validation.validation_prompt import boucle_vlm_prompt
    BOUCLE_AVAILABLE = True
except ImportError as _e:
    print(f"[warning] boucle_validation non disponible ({_e}) — variantes avec boucle desactivees")
    BOUCLE_AVAILABLE = False
    def boucle_vlm_prompt(*a, **kw): raise RuntimeError("boucle_validation non disponible")

DATA_PATH    = os.path.join(ROOT, "tests", "data", "prompts_ground_truth.json")
RESULTS_DIR  = os.path.join(ROOT, "tests", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "timing_results.json")


def load_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["prompts"]


def write_tmp_scene(items):
    """Ecrit la liste d'objets dans un fichier JSON temporaire pour la boucle."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(items, f)
    f.close()
    return f.name


def run_v1_1_full(prompt):
    obj_reconnus, _ = rec_v1_1(prompt)
    if not obj_reconnus:
        return
    relations    = object_relations(prompt, obj_reconnus).get("relations", [])
    orientations = extract_orientation_v1_1(prompt, obj_reconnus)
    items = [
        {"id": k, "urdf": v["urdf"], "path": v["path"],
         "dimensions": v["dimensions"], "scale": 1.0}
        for k, v in obj_reconnus.items()
    ]
    buildScene(items, relations, orientations)


def run_v1_1_avec_boucle(prompt):
    obj_reconnus, _ = rec_v1_1(prompt)
    if not obj_reconnus:
        return
    relations    = object_relations(prompt, obj_reconnus).get("relations", [])
    orientations = extract_orientation_v1_1(prompt, obj_reconnus)
    items = [
        {"id": k, "urdf": v["urdf"], "path": v["path"],
         "dimensions": v["dimensions"], "scale": 1.0}
        for k, v in obj_reconnus.items()
    ]
    items = buildScene(items, relations, orientations)
    tmp = write_tmp_scene(items)
    try:
        boucle_vlm_prompt(prompt, tmp, max_iter=10)
    finally:
        os.unlink(tmp)


def run_v2_full(prompt):
    phi3_cache.pop(prompt, None)
    obj_reconnus, _ = rec_v2(prompt)
    if not obj_reconnus:
        return
    place_scene(prompt, obj_reconnus)


def run_v2_avec_boucle(prompt):
    phi3_cache.pop(prompt, None)
    obj_reconnus, _ = rec_v2(prompt)
    if not obj_reconnus:
        return
    items = place_scene(prompt, obj_reconnus)
    tmp = write_tmp_scene(items)
    try:
        boucle_vlm_prompt(prompt, tmp, max_iter=10)
    finally:
        os.unlink(tmp)


def run_v1_full(prompt):
    obj_reconnus, _ = rec_v1_1_1(prompt)
    object_dim_quat(prompt, obj_reconnus)


def run_v1_avec_boucle(prompt):
    obj_reconnus, _ = rec_v1_1_1(prompt)
    items = object_dim_quat(prompt, obj_reconnus)
    if not items:
        return
    tmp = write_tmp_scene(items)
    try:
        boucle_vlm_prompt(prompt, tmp, max_iter=10)
    finally:
        os.unlink(tmp)


VERSIONS_SANS_BOUCLE = [
    ("V1.1", run_v1_1_full),
    ("V2",   run_v2_full),
    ("V1",   run_v1_full),
]

VERSIONS_AVEC_BOUCLE = [
    ("V1.1", run_v1_1_avec_boucle),
    ("V2",   run_v2_avec_boucle),
    ("V1",   run_v1_avec_boucle),
]


def versioned_name(base, tag, boucle=False):
    if base == "V2":
        name = "V2 (phi3-scene)"
    else:
        name = f"{base} ({tag})" if tag else base
    return f"{name} + boucle" if boucle else name


def run_versions(versions_fn, entries, n_runs, tag, boucle=False):
    results = {}
    for base, run_fn in versions_fn:
        name = versioned_name(base, tag, boucle=boucle)
        print(f"\n  {name}")
        print(f"  {'ID':<6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        print(f"  {'-'*45}")

        per_version = []
        for entry in entries:
            prompt = entry["prompt"]
            times  = []

            for _ in range(n_runs):
                t0 = time.time()
                try:
                    run_fn(prompt)
                except Exception as e:
                    print(f"  [ERREUR] {entry['id']} : {e}")
                times.append(round(time.time() - t0, 3))

            n    = len(times)
            mean = sum(times) / n
            std  = (sum((t - mean) ** 2 for t in times) / n) ** 0.5
            per_version.append({
                "id":     entry["id"],
                "mean_s": round(mean, 3),
                "std_s":  round(std, 3),
                "min_s":  round(min(times), 3),
                "max_s":  round(max(times), 3),
                "runs":   times,
            })
            print(f"  {entry['id']:<6} {mean:>7.2f}s {std:>7.2f}s "
                  f"{min(times):>7.2f}s {max(times):>7.2f}s")

        gm = sum(r["mean_s"] for r in per_version) / len(per_version)
        print(f"  {'GLOBAL':<6} {gm:>7.2f}s")
        results[name] = per_version

    return results


def benchmark(n_runs=5, save_json=True, tag=None, avec_boucle=True):
    entries = load_prompts()

    print(f"\n{'='*65}")
    print(f"  BENCHMARK TIMING  |  {n_runs} runs/prompt/version"
          + (f"  |  tag={tag}" if tag else ""))
    print(f"{'='*65}")

    print(f"\n--- SANS BOUCLE ({n_runs} runs) ---")
    results_sans = run_versions(VERSIONS_SANS_BOUCLE, entries, n_runs, tag, boucle=False)

    results_avec = {}
    if avec_boucle and BOUCLE_AVAILABLE:
        print(f"\n--- AVEC BOUCLE (1 run, max 10 iter) ---")
        results_avec = run_versions(VERSIONS_AVEC_BOUCLE, entries, n_runs=1, tag=tag, boucle=True)
    elif avec_boucle and not BOUCLE_AVAILABLE:
        print("\n[skip] boucle_validation non disponible — variantes avec boucle ignorees")

    all_results = {**results_sans, **results_avec}

    if save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        existing = {}
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        existing["n_runs"] = n_runs
        existing.setdefault("versions", {}).update(all_results)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\n[timing] Resultats -> {RESULTS_PATH}")

    return {"n_runs": n_runs, "versions": all_results}


if __name__ == "__main__":
    import argparse
    import pipeline.utils.ollama_client as ollama_mod
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",      type=int, default=5)
    parser.add_argument("--no-save",   action="store_true")
    parser.add_argument("--no-boucle", action="store_true", help="Ne pas mesurer les variantes avec boucle")
    parser.add_argument("--tag",   type=str, default=None,
                        help="Suffixe modele ex: gpt-4o ou llama3.1")
    parser.add_argument("--model", type=str, default=None,
                        help="Modele LLM ex: gpt-4o, llama3.1, llama3.2")
    args = parser.parse_args()
    if args.model:
        ollama_mod.LLM_MODEL = args.model
        print(f"[config] LLM = {args.model}")
    benchmark(n_runs=args.runs, save_json=not args.no_save,
              tag=args.tag, avec_boucle=not args.no_boucle)
