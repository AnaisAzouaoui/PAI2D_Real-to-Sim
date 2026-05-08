import math
from collections import Counter


def base(label):
    parts = label.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else label


def compute_orientation_f1(predicted, expected):
    """
    F1 sur les orientations.
    predicted / expected : [{"id": "...", "turn": "..."}]
    Matching sur (base(id), turn).
    """
    def key(o):
        return (base(o.get("id", "")), o.get("turn", ""))

    pred_count = Counter(key(o) for o in predicted)
    exp_count  = Counter(key(o) for o in expected)

    vp = sum((pred_count & exp_count).values())
    fp = sum(pred_count.values()) - vp
    fn = sum(exp_count.values()) - vp

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall    = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "VP": vp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def compute_f1(predicted, expected):
    """
   VP, FP, FN, precision, recall, F1 avec matching multiset (Counter)
    """
    pred_count = Counter(base(x) for x in predicted) # le counter c'est pour compter les doublons de deux urdf deux fois pour que ce soit accurate
    exp_count  = Counter(base(x) for x in expected)
    vp = sum((pred_count & exp_count).values())
    fp = sum(pred_count.values())- vp
    fn = sum(exp_count.values())- vp
    precision = vp/(vp+fp) if (vp+fp) > 0 else 0
    recall = vp/(vp + fn) if (vp + fn) > 0 else 0
    f1 = (2 *precision*recall / (precision + recall) if (precision + recall) > 0 else 0)

    data = {
        "VP": vp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1":round(f1, 4),
    }
    
    return data 


def aggregate_runs(per_run_metrics):
    """
    Calcule mean +/- std sur precision, recall, f1 a partir d'une liste de dicts.
    """
    def stats(key):
        vals = [r[key] for r in per_run_metrics]
        mean = sum(vals)/ len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        return round(mean,4), round(math.sqrt(variance),4)
    p_mean,p_std = stats("precision")
    r_mean,r_std = stats("recall")
    f_mean,f_std = stats("f1")
    data ={
        "precision_mean": p_mean, "precision_std": p_std,
        "recall_mean": r_mean, "recall_std": r_std,
        "f1_mean": f_mean, "f1_std": f_std,
    }
    return data


def compute_relation_accuracy(predicted, expected):
    """
    Fraction des relations attendues trouvees dans predicted.
    Matching sur (type, subject_base, object_base) en ignorant les suffixes _2, _3...
    """
    if not expected:
        return 1.0

    def base(label: str) -> str:
        parts = label.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return label

    def rel_key(r: dict):
        return (r.get("type", ""), base(r.get("subject", "")), base(r.get("object", "")))

    pred_keys = {rel_key(r) for r in predicted}
    matches   = sum(1 for r in expected if rel_key(r) in pred_keys)
    return round(matches / len(expected), 4)


def compute_relation_f1(predicted: list, expected: list) -> dict:
    """
    VP, FP, FN, precision, recall, F1 sur des listes de relations.
    Matching sur (type, base(subject), base(object)) — multiset via Counter.
    """
    def base(label: str) -> str:
        parts = label.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else label

    def rel_key(r: dict):
        return (r.get("type", ""), base(r.get("subject", "")), base(r.get("object", "")))

    pred_keys = Counter(rel_key(r) for r in predicted)
    exp_keys  = Counter(rel_key(r) for r in expected)

    vp = sum((pred_keys & exp_keys).values())
    fp = sum(pred_keys.values()) - vp
    fn = sum(exp_keys.values())  - vp

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall    = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "VP": vp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def compute_mae(predicted: dict, expected: dict, offset: dict = None) -> dict:
    """
    MAE par axe et RMSE global apres soustraction de l'offset de calibration.
    predicted : {obj_id: [x, y, z]}
    expected  : {obj_id: {"x": float, "y": float, "z": float}}
    offset    : {"x": float, "y": float, "z": float} ou None
    Seuls les objets presents dans les deux dicts sont pris en compte.
    """
    if offset is None:
        offset = {"x": 0.0, "y": 0.0, "z": 0.0}

    errors_x, errors_y, errors_z = [], [], []

    for obj_id, gt in expected.items():
        if obj_id not in predicted:
            continue
        pred = predicted[obj_id]
        px = pred[0] - offset["x"]
        py = pred[1] - offset["y"]
        pz = pred[2] - offset["z"]
        errors_x.append(abs(px - gt["x"]))
        errors_y.append(abs(py - gt["y"]))
        errors_z.append(abs(pz - gt["z"]))

    if not errors_x:
        return {"mae_x": None, "mae_y": None, "mae_z": None, "rmse": None, "n_matched": 0}

    mae_x = sum(errors_x) / len(errors_x)
    mae_y = sum(errors_y) / len(errors_y)
    mae_z = sum(errors_z) / len(errors_z)

    sq_errors = [ex**2 + ey**2 + ez**2
                 for ex, ey, ez in zip(errors_x, errors_y, errors_z)]
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

    return {
        "mae_x":     round(mae_x, 4),
        "mae_y":     round(mae_y, 4),
        "mae_z":     round(mae_z, 4),
        "rmse":      round(rmse, 4),
        "n_matched": len(errors_x),
    }


def compute_quat_validity(objects: list) -> dict:
    """
    Verifie que la norme de chaque quaternion vaut 1.0 +/- 0.01.
    Accepte le format [w, x, y, z] (V2) et [x, y, z, w] (V3) — la norme est la meme.
    """
    invalid_ids = []
    norms = {}
    for obj in objects:
        quat = obj.get("quat", [])
        if len(quat) != 4:
            invalid_ids.append(obj.get("id", "?"))
            norms[obj.get("id", "?")] = None
            continue
        norm = math.sqrt(sum(q ** 2 for q in quat))
        norms[obj.get("id", "?")] = round(norm, 4)
        if abs(norm - 1.0) > 0.01:
            invalid_ids.append(obj.get("id", "?"))

    return {
        "all_valid":   len(invalid_ids) == 0,
        "invalid_ids": invalid_ids,
        "norms":       norms,
    }


def compute_position_plausibility(objects: list) -> dict:
    """
    Verifie que z_base >= 0 pour chaque objet.
    z_base = pos[2] - dimensions[2] * scale / 2
    """
    invalid_ids = []
    for obj in objects:
        pos = obj.get("pos", [0, 0, 0])
        dims = obj.get("dimensions", [0, 0, 0])
        scale = obj.get("scale", 1.0)
        if len(pos) < 3 or len(dims) < 3:
            continue
        z_base = pos[2] - dims[2] * scale / 2.0
        if z_base < -0.001:
            invalid_ids.append(obj.get("id", "?"))

    return {
        "all_valid":   len(invalid_ids) == 0,
        "invalid_ids": invalid_ids,
    }


def compute_relation_order_accuracy(objects: list, expected_relations: list) -> float:
    """
    Pour les relations de type 'on': verifie que z_subject > z_object.
    Retourne la fraction des relations verticales correctement ordonnees.
    """
    pos_by_id = {obj["id"]: obj.get("pos", [0, 0, 0]) for obj in objects}
    vertical_relations = [r for r in expected_relations if r.get("type") in {"on", "under"}]

    if not vertical_relations:
        return 1.0

    correct = 0
    for rel in vertical_relations:
        subject_id = rel.get("subject", "")
        object_id  = rel.get("object", "")
        if subject_id not in pos_by_id or object_id not in pos_by_id:
            continue
        z_subject = pos_by_id[subject_id][2]
        z_object  = pos_by_id[object_id][2]
        if rel["type"] == "on" and z_subject > z_object:
            correct += 1
        elif rel["type"] == "under" and z_subject < z_object:
            correct += 1

    return round(correct / len(vertical_relations), 4)


def compute_calibration_offset(predicted: dict, ground_truth: dict) -> dict:
    """
    Calcule l'offset systematique entre les positions predites par V3 et les positions reelles.
    offset = mean(predicted - ground_truth) par axe, sur tous les objets presents dans les deux.
    predicted    : {obj_id: [x, y, z]}
    ground_truth : {obj_id: {"x": float, "y": float, "z": float}}
    """
    dx, dy, dz = [], [], []

    for obj_id, gt in ground_truth.items():
        if obj_id not in predicted:
            continue
        pred = predicted[obj_id]
        dx.append(pred[0] - gt["x"])
        dy.append(pred[1] - gt["y"])
        dz.append(pred[2] - gt["z"])

    if not dx:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    return {
        "x": round(sum(dx) / len(dx), 4),
        "y": round(sum(dy) / len(dy), 4),
        "z": round(sum(dz) / len(dz), 4),
    }
