"""
Test V3 
F1 objets, F1 relations, MAE/RMSE positions avec offset de calibration
"""
import sys
import os
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline.versions.v3_1_vlm_refined.pipeline_v3_1_image import scene_from_image, detect_and_estimate, load_images
from metrics import (
    compute_f1, aggregate_runs, compute_relation_f1,
    compute_mae, compute_relation_order_accuracy, compute_calibration_offset,
)

DATA_PATH  = os.path.join(ROOT, "tests", "data", "images_ground_truth.json")
RESULTS_DIR = os.path.join(ROOT, "tests", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "vision_results.json")


def load_image_ground_truth():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = [img for img in data["images"]]
    if not valid:
        print("[vision] Aucune image remplie dans images_ground_truth.json")
    return valid


def compute_calibration(image_entries, n_runs):
    calib_entries = [e for e in image_entries if e.get("is_calibration")]
    if not calib_entries:
        print("[vision] Aucune image de calibration — offset = (0, 0, 0)")
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    print(f"\n[vision] Calibration sur {len(calib_entries)} image(s)...")
    all_offsets = []
    for entry in calib_entries:
        xs, ys, zs = [], [], []
        for _ in range(n_runs):
            try:
                result = scene_from_image(entry["paths"])
                pred_pos = {obj["id"]: obj["pos"] for obj in result if "pos" in obj}
                off  = compute_calibration_offset(pred_pos, entry["ground_truth_positions"])
                xs.append(off["x"]); ys.append(off["y"]); zs.append(off["z"])
            except Exception as e:
                print(f"  [calibration] erreur : {e}")
        if xs:
            all_offsets.append({
                "x": sum(xs)/len(xs),
                "y": sum(ys)/len(ys),
                "z": sum(zs)/len(zs),
            })

    if not all_offsets:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    offset = {
        "x": round(sum(o["x"] for o in all_offsets)/len(all_offsets), 4),
        "y": round(sum(o["y"] for o in all_offsets)/len(all_offsets), 4),
        "z": round(sum(o["z"] for o in all_offsets)/len(all_offsets), 4),
    }
    print(f"  Offset : x={offset['x']}, y={offset['y']}, z={offset['z']}")
    return offset


def run_single_image(entry, offset, n_runs):
    paths  = entry["paths"]
    expected_obj = entry["expected_objects"]
    expected_rel = entry["expected_relations"]

    raw_gt = entry.get("ground_truth_positions", {})
    gt_positions = {}
    for obj_id, pos in raw_gt.items():
        if isinstance(pos, list) and len(pos) == 3:
            gt_positions[obj_id] = {"x": pos[0], "y": pos[1], "z": pos[2]}
        elif isinstance(pos, dict) and "x" in pos:
            gt_positions[obj_id] = pos
    has_pos  = bool(gt_positions)

    per_run  = []
    all_positions = {}

    for i in range(n_runs):
        t0 = time.time()
        try:
            result = scene_from_image(paths)
            elapsed = time.time() - t0
            predicted_ids = [obj["id"] for obj in result if "id" in obj]
            f1_m = compute_f1(predicted_ids, expected_obj)
            pred_pos  = {obj["id"]: obj["pos"] for obj in result if "pos" in obj}

            for obj_id, pos in pred_pos.items():
                all_positions.setdefault(obj_id, {"x": [], "y": [], "z": []})
                all_positions[obj_id]["x"].append(pos[0])
                all_positions[obj_id]["y"].append(pos[1])
                all_positions[obj_id]["z"].append(pos[2])

            mae_res = (compute_mae(pred_pos, gt_positions, offset)
                       if has_pos else {"mae_x": None, "mae_y": None, "mae_z": None, "rmse": None})

            per_run.append({
                "run": i + 1,
                "f1": f1_m["f1"], "precision": f1_m["precision"], "recall": f1_m["recall"],
                "VP": f1_m["VP"], "FP": f1_m["FP"], "FN": f1_m["FN"],
                "mae_x": mae_res.get("mae_x"), "mae_y": mae_res.get("mae_y"),
                "mae_z": mae_res.get("mae_z"), "rmse":  mae_res.get("rmse"),
                "rel_order_acc": compute_relation_order_accuracy(result, expected_rel),
                "time_s": round(elapsed, 3), "error": None,
            })
        except Exception as e:
            per_run.append({
                "run": i+1, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "VP": 0, "FP": 0, "FN": len(expected_obj),
                "mae_x": None, "mae_y": None, "mae_z": None, "rmse": None,
                "rel_order_acc": 0.0, "time_s": round(time.time()-t0, 3), "error": str(e),
            })

    try:
        images = load_images(paths)
        vlm_output  = detect_and_estimate(images)
        predicted_rels = vlm_output.get("relations", [])
        rel_f1 = compute_relation_f1(predicted_rels, expected_rel)
    except Exception:
        rel_f1 = {"VP": 0, "FP": 0, "FN": len(expected_rel),
                  "precision": 0.0, "recall": 0.0, "f1": 0.0}

    valid_runs = [r for r in per_run if r["error"] is None]

    def mean_std(key):
        vals = [r[key] for r in valid_runs if r[key] is not None]
        if not vals:
            return None, None
        m = sum(vals)/len(vals)
        s = (sum((v-m)**2 for v in vals)/len(vals))**0.5
        return round(m, 4), round(s, 4)

    f1_agg  = aggregate_runs(valid_runs) if valid_runs else {}
    mae_x_mean, mae_x_std = mean_std("mae_x")
    mae_y_mean, mae_y_std = mean_std("mae_y")
    mae_z_mean, mae_z_std = mean_std("mae_z")
    rmse_mean,  rmse_std = mean_std("rmse")
    times = [r["time_s"] for r in per_run]

    position_variance = {}
    for obj_id, coords in all_positions.items():
        for axis in ("x", "y", "z"):
            vals = coords[axis]
            if len(vals) > 1:
                m   = sum(vals)/len(vals)
                std = (sum((v-m)**2 for v in vals)/len(vals))**0.5
                position_variance.setdefault(obj_id, {})[f"{axis}_std"] = round(std, 4)

    return {
        "id":entry["id"],
        "paths": paths,
        "description": entry.get("description", ""),
        "per_run":per_run,
        "aggregated_f1": f1_agg,
        "relation_f1":rel_f1,
        "mae_mean": {"x": mae_x_mean, "y": mae_y_mean, "z": mae_z_mean, "rmse": rmse_mean},
        "mae_std": {"x": mae_x_std,  "y": mae_y_std,  "z": mae_z_std,  "rmse": rmse_std},
        "position_variance": position_variance,
        "time_mean_s": round(sum(times)/len(times), 3),
        "time_std_s":  round((sum((t-sum(times)/len(times))**2 for t in times)/len(times))**0.5, 3),
    }


def run_all(n_runs=5, save_json=True):
    entries = load_image_ground_truth()
    if not entries:
        return {}

    offset = compute_calibration(entries, n_runs=n_runs)
    test_entries = [e for e in entries if not e.get("is_calibration")]
    results  = []

    print(f"\n{'='*70}")
    print(f"  TEST VISION  |  {n_runs} runs/image  |  offset={offset}")
    print(f"{'='*70}")
    print(f"{'ID':<8} {'F1 obj':>8} {'F1 rel':>8} {'MAE x':>7} {'MAE y':>7} {'MAE z':>7} {'RMSE':>7} {'Time':>7}")
    print(f"{'-'*70}")

    for entry in test_entries:
        r = run_single_image(entry, offset, n_runs)
        agg = r["aggregated_f1"]
        mae_m = r["mae_mean"]
        f1_rel = r.get("relation_f1", {}).get("f1", 0.0)
        results.append(r)

        def fmt(v): return f"{v:.3f}" if v is not None else "  N/A"
        print(f"{r['id']:<8} {agg.get('f1_mean', 0):>8.4f} {f1_rel:>8.4f} "
              f"{fmt(mae_m['x']):>7} {fmt(mae_m['y']):>7} {fmt(mae_m['z']):>7} "
              f"{fmt(mae_m['rmse']):>7} {r['time_mean_s']:>7.2f}")

    f1_vals = [r["aggregated_f1"].get("f1_mean", 0) for r in results]
    f1_mean = sum(f1_vals)/len(f1_vals) if f1_vals else 0.0
    mae_x_vals = [r["mae_mean"]["x"] for r in results if r["mae_mean"]["x"] is not None]
    mae_y_vals = [r["mae_mean"]["y"] for r in results if r["mae_mean"]["y"] is not None]
    mae_z_vals = [r["mae_mean"]["z"] for r in results if r["mae_mean"]["z"] is not None]

    print(f"\n  Global F1 objects : {f1_mean:.4f}")
    if mae_x_vals:
        print(f"  Global MAE x : {sum(mae_x_vals)/len(mae_x_vals):.4f} m")
        print(f"  Global MAE y : {sum(mae_y_vals)/len(mae_y_vals):.4f} m")
        print(f"  Global MAE z : {sum(mae_z_vals)/len(mae_z_vals):.4f} m")

    output = {
        "n_runs": n_runs,
        "calibration_offset": offset,
        "results": results,
        "global_f1_mean":round(f1_mean, 4),
        "global_mae_mean": {
            "x": round(sum(mae_x_vals)/len(mae_x_vals), 4) if mae_x_vals else None,
            "y": round(sum(mae_y_vals)/len(mae_y_vals), 4) if mae_y_vals else None,
            "z": round(sum(mae_z_vals)/len(mae_z_vals), 4) if mae_z_vals else None,
        },
    }

    if save_json:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[vision] Resultats -> {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run_all(n_runs=args.runs, save_json=not args.no_save)
