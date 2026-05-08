"""
Test de reconnaissance des relations et orientations spatiales.
V1  : LLM (object_relations + orientation)
V1.1.1 : phi3-scene
V2  : coherence spatiale uniquement (pas de relations explicites)
Aucun appel Genesis.
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

from pipeline.versions.v1_llm_prim.placement.placement_v1_relations import object_relations
from pipeline.versions.v1_llm_prim.placement.placement_v1_distances_orientations import orientation as extract_orientation_v1
from pipeline.versions.v1_llm_prim.pipeline_v1_1_1 import phi3_result, phi3_cache, object_rec as phi3_object_rec
from pipeline.versions.v2_llm_only.pipeline_v2_llm import object_dim_quat
from pipeline.versions.v1_llm_prim.object_recognition.object_rec_v1_1_embedding import object_rec as rec_v1_1
from pipeline.versions.v1_llm_prim.object_recognition.object_rec_v1_llm import object_rec as rec_v1
from metrics import (
    compute_relation_f1, compute_orientation_f1,
    compute_quat_validity, compute_position_plausibility, compute_relation_order_accuracy,
)

DATA_PATH    = os.path.join(ROOT, "tests", "data", "prompts_ground_truth.json")
RESULTS_DIR  = os.path.join(ROOT, "tests", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "placement_results.json")

VALID_RELATION_TYPES = {"on", "under", "left_of", "right_of", "in_front_of", "behind", "against", "inside"}
VALID_TURNS = {"tip_left", "tip_right", "tip_forward", "tip_backward",
               "upside_down", "turn_left", "turn_right", "turn_around"}


def load_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["prompts"]


def gt_obj_reconnus(entry):
    """Dict obj_reconnus depuis le ground truth pour isoler la qualite des relations."""
    return {
        o["id"]: {"urdf": o["urdf"], "path": "", "dimensions": None}
        for o in entry["expected_objects"]
    }


def run_version(name, entries, n_runs, extract_fn):
    """
    extract_fn(prompt, entry) -> (predicted_relations, predicted_orientations)
    """
    results = []
    print(f"\n  {name} — relations + orientations  |  {n_runs} runs/prompt")
    print(f"  {'ID':<6} {'Rel F1':>7} {'Ori F1':>7} {'Time':>6}")
    print(f"  {'-'*32}")

    for entry in entries:
        prompt  = entry["prompt"]
        gt_urdf = {o["id"]: o["urdf"] for o in entry.get("expected_objects", [])}

        expected_rel = [
            {**r, "subject": gt_urdf.get(r["subject"], r["subject"]),
                   "object":  gt_urdf.get(r["object"],  r["object"])}
            for r in entry.get("expected_relations", [])
        ]
        expected_ori = [
            {**o, "id": gt_urdf.get(o["id"], o["id"])}
            for o in entry.get("expected_orientations", [])
        ]
        per_run = []

        for _ in range(n_runs):
            t0 = time.time()
            try:
                pred_rel, pred_ori = extract_fn(prompt, entry)
                rel_metrics = compute_relation_f1(pred_rel, expected_rel)
                ori_metrics = compute_orientation_f1(pred_ori, expected_ori)
                per_run.append({
                    "rel_f1":  rel_metrics["f1"],
                    "rel_pre": rel_metrics["precision"],
                    "rel_rec": rel_metrics["recall"],
                    "ori_f1":  ori_metrics["f1"],
                    "ori_pre": ori_metrics["precision"],
                    "ori_rec": ori_metrics["recall"],
                    "time_s":  round(time.time() - t0, 3),
                    "error":   None,
                })
            except Exception as e:
                per_run.append({
                    "rel_f1": 0.0, "rel_pre": 0.0, "rel_rec": 0.0,
                    "ori_f1": 0.0, "ori_pre": 0.0, "ori_rec": 0.0,
                    "time_s": round(time.time() - t0, 3), "error": str(e),
                })

        times       = [r["time_s"] for r in per_run]
        rel_f1_mean = sum(r["rel_f1"] for r in per_run) / len(per_run)
        ori_f1_mean = sum(r["ori_f1"] for r in per_run) / len(per_run)

        results.append({
            "id":              entry["id"],
            "rel_precision":   round(sum(r["rel_pre"] for r in per_run) / len(per_run), 4),
            "rel_recall":      round(sum(r["rel_rec"] for r in per_run) / len(per_run), 4),
            "rel_f1_mean":     round(rel_f1_mean, 4),
            "ori_precision":   round(sum(r["ori_pre"] for r in per_run) / len(per_run), 4),
            "ori_recall":      round(sum(r["ori_rec"] for r in per_run) / len(per_run), 4),
            "ori_f1_mean":     round(ori_f1_mean, 4),
            "time_mean_s":     round(sum(times) / len(times), 3),
        })
        print(f"  {entry['id']:<6} {rel_f1_mean:>7.3f} {ori_f1_mean:>7.3f} {sum(times)/len(times):>5.2f}s")

    return results


def to_urdf_rel(relations, label_to_urdf):
    result = []
    for r in relations:
        if r.get("type") not in VALID_RELATION_TYPES:
            continue
        subj = label_to_urdf.get(r.get("subject", ""))
        obj  = label_to_urdf.get(r.get("object", ""))
        if subj and obj:
            result.append({**r, "subject": subj, "object": obj})
    return result


def to_urdf_ori(orientations, label_to_urdf):
    result = []
    for o in orientations:
        if o.get("turn") not in VALID_TURNS:
            continue
        urdf = label_to_urdf.get(o.get("id", ""))
        if urdf:
            result.append({**o, "id": urdf})
    return result


def extract_v1(prompt, entry):
    obj_reconnus, _ = rec_v1(prompt)
    rel_result = object_relations(prompt, obj_reconnus)
    ori_result = extract_orientation_v1(prompt, obj_reconnus)
    label_to_urdf = {label: info["urdf"] for label, info in obj_reconnus.items()}
    predicted_rel = to_urdf_rel(rel_result.get("relations", []), label_to_urdf)
    predicted_ori = to_urdf_ori(ori_result, label_to_urdf)
    return predicted_rel, predicted_ori


def extract_v1_1_1(prompt, entry):
    phi3_cache.pop(prompt, None)
    phi3 = phi3_result(prompt)
    obj_phi3, _ = phi3_object_rec(prompt)
    label_to_urdf = {label: info["urdf"] for label, info in obj_phi3.items()}
    predicted_rel = to_urdf_rel(phi3.get("relations", []), label_to_urdf)
    predicted_ori = to_urdf_ori(phi3.get("orientations", []), label_to_urdf)
    return predicted_rel, predicted_ori


def run_v2_spatial(entries, n_runs):
    """V2 produit des positions directement — on teste la coherence spatiale, pas les relations."""
    results = []
    print(f"\n  V2 — coherence spatiale (positions LLM)  |  {n_runs} runs/prompt")
    print(f"  {'ID':<6} {'Quat':>6} {'Z ok':>6} {'Order':>7} {'Time':>6}")
    print(f"  {'-'*38}")

    for entry in entries:
        prompt       = entry["prompt"]
        expected_rel = entry.get("expected_relations", [])
        per_run      = []

        for _ in range(n_runs):
            t0 = time.time()
            try:
                obj_reconnus, _ = rec_v1_1(prompt)
                scene     = object_dim_quat(prompt, obj_reconnus)
                quat_res  = compute_quat_validity(scene)
                pos_res   = compute_position_plausibility(scene)
                order_acc = compute_relation_order_accuracy(scene, expected_rel)
                per_run.append({
                    "quat_valid":    quat_res["all_valid"],
                    "pos_plausible": pos_res["all_valid"],
                    "rel_order_acc": order_acc,
                    "time_s":        round(time.time() - t0, 3),
                    "error":         None,
                })
            except Exception as e:
                per_run.append({
                    "quat_valid": False, "pos_plausible": False, "rel_order_acc": 0.0,
                    "time_s": round(time.time() - t0, 3), "error": str(e),
                })

        times      = [r["time_s"] for r in per_run]
        quat_rate  = sum(1 for r in per_run if r["quat_valid"]) / len(per_run)
        pos_rate   = sum(1 for r in per_run if r["pos_plausible"]) / len(per_run)
        order_mean = sum(r["rel_order_acc"] for r in per_run) / len(per_run)

        results.append({
            "id":                 entry["id"],
            "quat_valid_rate":    round(quat_rate, 4),
            "pos_plausible_rate": round(pos_rate, 4),
            "rel_order_acc_mean": round(order_mean, 4),
            "time_mean_s":        round(sum(times) / len(times), 3),
        })
        print(f"  {entry['id']:<6} {quat_rate:>6.0%} {pos_rate:>6.0%} {order_mean:>7.4f} {sum(times)/len(times):>5.2f}s")

    return results


def global_mean(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_all(n_runs=5, save_json=True, only_versions=None, tag=None):
    entries = load_prompts()
    all_v = only_versions or ["V1", "V1.1.1", "V2"]

    print(f"\n{'='*60}")
    print(f"  TEST PLACEMENT  |  {n_runs} runs/prompt")
    print(f"{'='*60}")

    new_results = {}

    if "V1" in all_v:
        name = f"V1 ({tag})" if tag else "V1 (gpt-4o)"
        res = run_version(name, entries, n_runs, extract_v1)
        new_results[name] = res
        print(f"\n  Globaux {name} : rel_F1={global_mean(res, 'rel_f1_mean')}  "
              f"ori_F1={global_mean(res, 'ori_f1_mean')}")

    if "V1.1.1" in all_v:
        res = run_version("V1.1.1 (phi3-scene)", entries, n_runs, extract_v1_1_1)
        new_results["V1.1.1 (phi3-scene)"] = res
        print(f"\n  Globaux V1.1.1 (phi3-scene) : rel_F1={global_mean(res, 'rel_f1_mean')}  "
              f"ori_F1={global_mean(res, 'ori_f1_mean')}")

    if "V2" in all_v:
        name = f"V2 ({tag})" if tag else "V2 (gpt-4o)"
        res = run_v2_spatial(entries, n_runs)
        new_results[name] = res
        print(f"\n  Globaux {name} : quat={global_mean(res, 'quat_valid_rate'):.0%}  "
              f"z_ok={global_mean(res, 'pos_plausible_rate'):.0%}  "
              f"order={global_mean(res, 'rel_order_acc_mean')}")

    if save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        existing = {}
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        existing["n_runs"] = n_runs
        existing.update(new_results)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\n[placement] Resultats -> {RESULTS_PATH}")

    return existing


if __name__ == "__main__":
    import argparse
    import pipeline.utils.ollama_client as ollama_mod
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--versions", nargs="+", choices=["V1", "V1.1.1", "V2"],
                        help="Versions a tester (defaut: toutes)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffixe modele ex: gpt-4o ou llama3.1")
    parser.add_argument("--model", type=str, default=None,
                        help="Modele LLM ex: gpt-4o, llama3.1, llama3.2")
    args = parser.parse_args()
    if args.model:
        ollama_mod.LLM_MODEL = args.model
        print(f"[config] LLM = {args.model}")
    run_all(n_runs=args.runs, save_json=not args.no_save,
            only_versions=args.versions, tag=args.tag)
