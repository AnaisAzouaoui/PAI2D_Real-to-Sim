"""
Evaluation du modele fine-tune via Ollama (GGUF local).

Usage :
  python training/evaluate_ollama.py
  python training/evaluate_ollama.py --n 200
  python training/evaluate_ollama.py --test training/data/dataset_merged_test.jsonl --n 50
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inference"))
from extractor import extract_labels

HELD_OUT = { # objet qu'on a volontairement exclu du dataset d'entrainement 
    "baignoire", "piano", "moto", "aquarium", "carafe",
    "banana", "remote", "wardrobe", "dishwasher",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def f1_objects(pred, true):
    p, t = set(pred or []), set(true or [])
    if not t:
        return 1.0 if not p else 0.0
    tp = len(p & t)
    pr = tp / len(p) if p else 0.0
    re = tp / len(t)
    return 2 * pr * re / (pr + re) if pr + re else 0.0


def rel_key(r):
    return (r["type"], r["subject"], r["object"])


def f1_relations(pred, true):
    p, t = set(rel_key(r) for r in (pred or [])), set(rel_key(r) for r in (true or []))
    if not t:
        return 1.0 if not p else 0.0
    tp = len(p & t)
    pr = tp / len(p) if p else 0.0
    re = tp / len(t)
    return 2 * pr * re / (pr + re) if pr + re else 0.0


def is_held_out(ex):
    for id_ in ex["output"]["objets"]:
        base = id_.rsplit("_", 1)[0] if id_[-1].isdigit() else id_
        if base in HELD_OUT:
            return True
    return False


def evaluate(examples):
    parse_ok = []
    f1_obj = []
    root_acc = []
    rel_em = []
    rel_f1 = []
    ori_em = []
    held_f1 = []
    errors = []
    by_config = {}

    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] {ex['input'][:60]}...", flush=True)
        try:
            pred = extract_labels(ex["input"])
            true = ex["output"]

            parse_ok.append(1.0)

            f1_obj.append(f1_objects(pred.get("objets", []), true["objets"]))

            root_acc.append(1.0 if pred.get("root") == true.get("root") else 0.0)

            pred_rels = pred.get("relations", [])
            true_rels = true.get("relations", [])
            rel_em.append(1.0 if set(rel_key(r) for r in pred_rels) == set(rel_key(r) for r in true_rels) else 0.0)
            rel_f1.append(f1_relations(pred_rels, true_rels))

            pred_ori = set((o["id"], o["turn"]) for o in pred.get("orientations", []))
            true_ori = set((o["id"], o["turn"]) for o in true.get("orientations", []))
            ori_em.append(1.0 if pred_ori == true_ori else 0.0)

            if is_held_out(ex):
                held_f1.append(f1_objects(pred.get("objets", []), true["objets"]))

            cfg = ex.get("config", "unknown")
            if cfg not in by_config:
                by_config[cfg] = []
            by_config[cfg].append(rel_em[-1])

        except Exception as e:
            parse_ok.append(0.0)
            f1_obj.append(0.0)
            root_acc.append(0.0)
            rel_em.append(0.0)
            rel_f1.append(0.0)
            ori_em.append(0.0)
            errors.append((i, str(e)))

    return {
        "parse_ok":   parse_ok,
        "f1_obj":     f1_obj,
        "root_acc":   root_acc,
        "rel_em":     rel_em,
        "rel_f1":     rel_f1,
        "ori_em":     ori_em,
        "held_f1":    held_f1,
        "errors":     errors,
        "by_config":  by_config,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="training/data/dataset_merged_test.jsonl")
    parser.add_argument("--n",    type=int, default=100, help="Nombre d'exemples (defaut 100, 0 = tous)")
    args = parser.parse_args()

    examples = load_jsonl(args.test)
    if args.n and args.n < len(examples):
        examples = examples[:args.n]

    print(f"Evaluation sur {len(examples)} exemples ({args.test})...\n")

    res = evaluate(examples)

    avg = lambda l: sum(l) / len(l) if l else None

    print(f"\n{'='*45}")
    print(f"  Resultats sur {len(examples)} exemples")
    print(f"{'='*45}")

    metrics = [
        ("JSON parse OK",  avg(res["parse_ok"])),
        ("F1 objets", avg(res["f1_obj"])),
        ("Root accuracy", avg(res["root_acc"])),
        ("Relation F1", avg(res["rel_f1"])),
        ("Relation exact match",  avg(res["rel_em"])),
        ("Orientation exact match", avg(res["ori_em"])),
    ]

    for label, val in metrics:
        bar = ""
        if val is not None:
            filled = int(val * 20)
            bar = f" [{'#'*filled}{'.'*(20-filled)}]"
            print(f" {label:<28} {val:.1%}{bar}")
        else:
            print(f" {label:<28} N/A")

    if res["held_f1"]:
        val = avg(res["held_f1"])
        filled = int(val * 20)
        bar = f"  [{'#'*filled}{'.'*(20-filled)}]"
        print(f"  {'F1 held-out (generalisation)':<28} {val:.1%}{bar}  ({len(res['held_f1'])} ex)")

    print(f"{'='*45}")

    if res["by_config"]:
        print(f"\n  Relation exact match par config :")
        for cfg, vals in sorted(res["by_config"].items(), key=lambda x: -avg(x[1])):
            v = avg(vals)
            filled = int(v * 15)
            bar = f"[{'#'*filled}{'.'*(15-filled)}]"
            print(f"    {cfg:<20} {v:.1%}  {bar}  ({len(vals)} ex)")

    if res["errors"]:
        print(f"\n  Erreurs : {len(res['errors'])} exemples n'ont pas pu etre evalues")


if __name__ == "__main__":
    main()
