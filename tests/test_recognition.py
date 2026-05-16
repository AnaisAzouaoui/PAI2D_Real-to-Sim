"""
Test de reconnaissance d'objets — V1.1 / V1.1.1 / V1
F1 objets sur tout, synonymes V1.1.1, calibration seuil V1.1.1
"""
import sys
import os
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1 import object_rec as rec_v1_1
from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1_1 import object_rec as rec_v1_1_1
from pipeline.versions.v1_llm_only.pipeline_v1_llm import object_dim_quat as v1_place
from metrics import compute_f1, aggregate_runs


def rec_v1(prompt):
    objs_list = v1_place(prompt, {})
    objs = {o["id"]: {"urdf": o["urdf"]} for o in objs_list if "urdf" in o}
    return objs, []

DATA_PATH = os.path.join(ROOT, "tests", "data", "prompts_ground_truth.json")
SYNONYMES_PATH = os.path.join(ROOT, "tests", "data", "synonymes.json")
RESULTS_DIR = os.path.join(ROOT, "tests", "results")
RESULTS_PATH  = os.path.join(RESULTS_DIR, "recognition_results.json")

VERSIONS = [
    ("V1.1",   rec_v1_1,   False),
    ("V1.1.1", rec_v1_1_1, False),
    ("V1",     rec_v1,     False),
]


def load_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return [p for p in data["prompts"]]


def run_version(name, rec_fn, entries, n_runs, flush_cache):
    results = []
    for entry in entries:
        prompt = entry["prompt"]
        expected = [o["urdf"] for o in entry["expected_objects"]]
        per_run = []
        for _ in range(n_runs):
            t0 = time.time()
            try:
                objs, _ = rec_fn(prompt)
                predicted = [v["urdf"] for v in objs.values()]
                m = compute_f1(predicted, expected)
                per_run.append({**m, "time_s": round(time.time() - t0, 3), "error": None})
            except Exception as e:
                per_run.append({
                    "f1": 0.0, "precision": 0.0, "recall": 0.0,
                    "VP": 0, "FP": 0, "FN": len(expected),
                    "time_s": round(time.time() - t0, 3), "error": str(e),
                })
        times = [r["time_s"] for r in per_run]
        results.append({
            "id": entry["id"],
            "aggregated":  aggregate_runs(per_run),
            "time_mean_s": round(sum(times) / len(times), 3),
        })
    return results


def run_synonym_tests():
    with open(SYNONYMES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pairs = [p for p in data.get("pairs", []) if p["user_label"] != "REMPLIR_synonyme"]
    if not pairs:
        print("[recognition] Aucun synonyme rempli — skip.")
        return []
    results = []
    print(f"\n  Synonymes V1.1.1 :")
    print(f"  {'Label':<25} {'Attendu':<30} {'OK':>4}")
    print(f"  {'-'*62}")
    for pair in pairs:
        try:
            objs, _ = rec_v1_1_1([pair["user_label"]])  # bypass LLM, test embedding seul
            predicted = [v["urdf"] for v in objs.values()]
            matched = pair["expected_urdf"] in predicted
        except Exception:
            matched = False
        results.append({"user_label": pair["user_label"],
                        "expected": pair["expected_urdf"], "matched": matched})
        print(f"  {pair['user_label']:<25} {pair['expected_urdf']:<30} {'OK' if matched else 'FAIL':>4}")
    rate = sum(r["matched"] for r in results) / len(results)
    print(f"  Synonym match rate: {rate:.2%}")
    return results


def run_threshold_tests():
    with open(SYNONYMES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    pairs = [p for p in data.get("threshold_boundary_pairs", [])]
    if not pairs:
        print("[recognition] Aucun threshold pair rempli — skip.")
        return []
    results = []
    print(f"\n  Calibration seuil V1.1.1 :")
    print(f"  {'Label':<25} {'should_match':>12} {'did_match':>10} {'OK':>5}")
    print(f"  {'-'*55}")
    for pair in pairs:
        try:
            objs, _ = rec_v1_1_1([pair["user_label"]])  # bypass LLM
            did_match = len(objs) > 0
        except Exception:
            did_match = False
        correct = did_match == pair["should_match"]
        results.append({"user_label": pair["user_label"],
                        "should_match": pair["should_match"],
                        "did_match": did_match, "correct": correct})
        print(f"  {pair['user_label']:<25} {str(pair['should_match']):>12} "
              f"{str(did_match):>10} {'OK' if correct else 'FAIL':>5}")
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"Threshold accuracy: {acc:.2%}")
    return results


def run_all(n_runs=5, save_json=True, only_versions=None, tag=None):
    entries = load_prompts()

    versions = [(n, fn, fl) for n, fn, fl in VERSIONS
                if only_versions is None or n in only_versions]
    if tag:
        versions = [(f"{n} ({tag})", fn, fl) for n, fn, fl in versions]

    print(f"\n{'='*75}")
    print(f"  TEST RECONNAISSANCE  |  {n_runs} runs/prompt")
    print(f"{'='*75}")
    header = f"{'ID':<6}" + "".join(f"  {n+' F1':>14}" for n, _, _ in versions)
    print(header)
    print("-" * 75)

    version_results = {}
    for name, rec_fn, flush in versions:
        version_results[name] = run_version(name, rec_fn, entries, n_runs, flush)

    for i, entry in enumerate(entries):
        row = f"{entry['id']:<6}"
        for name, _, _ in versions:
            agg = version_results[name][i]["aggregated"]
            row += f"  {agg['f1_mean']:>6.4f}+/-{agg['f1_std']:.4f}"
        print(row)

    print("-" * 75)
    row = f"{'GLOBAL':<6}"
    for name, _, _ in versions:
        vals = [r["aggregated"]["f1_mean"] for r in version_results[name]]
        mean = sum(vals) / len(vals)
        row += f"{mean:>6.4f} "
    print(row)

    synonym_results = run_synonym_tests()
    threshold_results = run_threshold_tests()

    if save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        existing = {}
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        existing.setdefault("versions", {}).update(version_results)
        existing["n_runs"] = n_runs
        if synonym_results:
            existing["synonym_results"] = synonym_results
        if threshold_results:
            existing["threshold_results"] = threshold_results
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\n[recognition] Resultats -> {RESULTS_PATH}")

    return {"n_runs": n_runs, "versions": version_results,
            "synonym_results": synonym_results, "threshold_results": threshold_results}


if __name__ == "__main__":
    import argparse
    import pipeline.utils.ollama_client as ollama_mod
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--versions", nargs="+", choices=[n for n, _, _ in VERSIONS],
                        help="Versions a tester (defaut: toutes)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffixe du nom de version ex: gpt-4o ou llama3.1")
    parser.add_argument("--model", type=str, default=None,
                        help="Modele LLM ex: gpt-4o, llama3.1, llama3.2")
    args = parser.parse_args()
    if args.model:
        ollama_mod.LLM_MODEL = args.model
        print(f"[config] LLM = {args.model}")
    run_all(n_runs=args.runs, save_json=not args.no_save,
            only_versions=args.versions, tag=args.tag)
