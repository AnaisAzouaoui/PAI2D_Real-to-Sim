"""
Benchmark du pipeline complet (de la reconnaissance jusqu'a la generation de scene).
Mesure uniquement les temps d'execution, 5 runs par prompt par version.

V1.1: object_rec (LLM) + object_relations + orientation + buildScene (Genesis)
V2 : phi3-scene + object_rec (embedding) + place_scene (Genesis)
V1 : object_rec (embedding) + object_dim_quat (LLM, pas de Genesis)
"""
import sys
import os
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1 import object_rec as rec_v1_1
from pipeline.versions.v1_1_llm_and_primitives.placement.placement_v1_relations import object_relations
from pipeline.versions.v1_1_llm_and_primitives.placement.placement_v1_distances_orientations import orientation as extract_orientation_v1_1
from pipeline.sceneBuilding import buildScene
from pipeline.itemSpec import getFilePath

from pipeline.versions.v2_finetuned.pipeline_v2 import object_rec as rec_v2, place_scene, phi3_cache

from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1_1 import object_rec as rec_v1_1_1
from pipeline.versions.v1_llm_only.pipeline_v1_llm import object_dim_quat

DATA_PATH    = os.path.join(ROOT, "tests", "data", "prompts_ground_truth.json")
RESULTS_DIR  = os.path.join(ROOT, "tests", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "timing_results.json")


def load_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["prompts"]


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


def run_v2_full(prompt):
    phi3_cache.pop(prompt, None)
    obj_reconnus, _ = rec_v2(prompt)
    if not obj_reconnus:
        return
    place_scene(prompt, obj_reconnus)


def run_v1_full(prompt):
    obj_reconnus, _ = rec_v1_1_1(prompt)
    object_dim_quat(prompt, obj_reconnus)


VERSIONS = [
    ("V1.1", run_v1_1_full),
    ("V2",   run_v2_full),
    ("V1",   run_v1_full),
]


def benchmark(n_runs=5, save_json=True):
    entries = load_prompts()

    print(f"\n{'='*65}")
    print(f"  BENCHMARK TIMING  |  {n_runs} runs/prompt/version")
    print(f"{'='*65}")

    results = {name: [] for name, _ in VERSIONS}

    for name, run_fn in VERSIONS:
        print(f"\n  {name}")
        print(f"  {'ID':<6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        print(f"  {'-'*45}")

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
            results[name].append({
                "id":  entry["id"],
                "mean_s":  round(mean, 3),
                "std_s": round(std, 3),
                "min_s":round(min(times), 3),
                "max_s": round(max(times), 3),
                "runs":  times,
            })
            print(f"  {entry['id']:<6} {mean:>7.2f}s {std:>7.2f}s {min(times):>7.2f}s {max(times):>7.2f}s")

        version_means = [r["mean_s"] for r in results[name]]
        global_mean   = sum(version_means) / len(version_means)
        print(f"  {'GLOBAL':<6} {global_mean:>7.2f}s")

    output = {"n_runs": n_runs, "versions": results}

    if save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[timing] Resultats -> {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    benchmark(n_runs=args.runs, save_json=not args.no_save)
