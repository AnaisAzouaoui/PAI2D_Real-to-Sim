"""
Evaluation du modele fine-tune sur le test set.

Usage :
  python evaluate.py --model phi3_finetuned --test data/dataset_test.jsonl
"""
import argparse
import json
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "Tu es un assistant qui extrait les objets, relations spatiales, distances et orientations "
    "d'une description de scene. Reponds uniquement en JSON valide.\n"
    "Format : {\"objets\": [...], \"root\": \"...\", "
    "\"relations\": [{\"type\": \"...\", \"subject\": \"...\", \"object\": \"...\", \"distance\": <float optionnel>}], "
    "\"orientations\": [{\"id\": \"...\", \"turn\": \"...\"}]}\n"
    "Types de relations : on, under, left_of, right_of, in_front_of, behind, against, inside\n"
    "Champ 'distance' (en metres) autorise UNIQUEMENT sur : left_of, right_of, in_front_of, behind\n"
    "Turns valides : tip_left, tip_right, tip_forward, tip_backward, upside_down, turn_left, turn_right, turn_around\n"
    "Convention : premiere instance = 'banane', deuxieme = 'banane_2'.\n"
    "Si pas de distance mentionnee : ne pas inclure le champ. Si tout est debout : orientations = []."
)

HELD_OUT_OBJECTS = {
    "baignoire", "piano", "moto", "aquarium", "carafe",
    "banana", "remote", "wardrobe", "dishwasher",
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_output(text):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def predict(model, tokenizer, prompt, device):
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_output(generated)


def f1_objects(pred_objets, true_objets):
    pred_set = set(pred_objets or [])
    true_set = set(true_objets)
    if not true_set:
        return 1.0 if not pred_set else 0.0
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(true_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def relation_exact_match(pred_rels, true_rels):
    def key(r):
        return (r["type"], r["subject"], r["object"])
    pred_set = set(key(r) for r in (pred_rels or []))
    true_set = set(key(r) for r in true_rels)
    return pred_set == true_set


def orientations_exact_match(pred_orients, true_orients):
    def key(o):
        return (o["id"], o["turn"])
    pred_set = set(key(o) for o in (pred_orients or []))
    true_set = set(key(o) for o in (true_orients or []))
    return pred_set == true_set


def distance_metrics(pred_rels, true_rels):
    """Retourne (mae_or_None, presence_f1_or_None) pour les distances.
    - mae : moyenne des |pred_dist - true_dist| sur les relations matchees ou les deux ont une distance
    - presence_f1 : F1 sur la presence/absence du champ distance par relation matchee
    """
    def key(r):
        return (r["type"], r["subject"], r["object"])
    pred_map = {key(r): r for r in (pred_rels or [])}
    true_map = {key(r): r for r in (true_rels or [])}

    common_keys = set(pred_map) & set(true_map)
    if not common_keys:
        return None, None

    abs_errors  = []
    tp = fp = fn = tn = 0
    for k in common_keys:
        pd = pred_map[k].get("distance")
        td = true_map[k].get("distance")
        if td is not None and pd is not None:
            tp += 1
            try:
                abs_errors.append(abs(float(pd) - float(td)))
            except (TypeError, ValueError):
                pass
        elif td is None and pd is not None:
            fp += 1
        elif td is not None and pd is None:
            fn += 1
        else:
            tn += 1

    mae = (sum(abs_errors) / len(abs_errors)) if abs_errors else None
    if tp + fp + fn == 0:
        presence_f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        presence_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return mae, presence_f1


def is_held_out(example):
    for id_ in example["output"]["objets"]:
        base = id_.rsplit("_", 1)[0] if id_[-1].isdigit() else id_
        if base in HELD_OUT_OBJECTS:
            return True
    return False


def evaluate(model, tokenizer, examples, device):
    metrics = {
        "f1_objets":              [],
        "root_acc":               [],
        "relation_exact_match":   [],
        "orientations_exact_match": [],
        "distance_mae":           [],
        "distance_presence_f1":   [],
        "coref_no_double":        [],
        "multi_instance_ok":      [],
        "held_out_f1":            [],
    }

    for ex in examples:
        pred = predict(model, tokenizer, ex["input"], device)
        true = ex["output"]

        if pred is None:
            metrics["f1_objets"].append(0.0)
            metrics["root_acc"].append(0.0)
            metrics["relation_exact_match"].append(0.0)
            metrics["orientations_exact_match"].append(0.0)
            continue

        metrics["f1_objets"].append(f1_objects(pred.get("objets"), true["objets"]))
        metrics["root_acc"].append(1.0 if pred.get("root") == true.get("root") else 0.0)
        metrics["relation_exact_match"].append(1.0 if relation_exact_match(pred.get("relations"), true["relations"]) else 0.0)
        metrics["orientations_exact_match"].append(1.0 if orientations_exact_match(pred.get("orientations"), true.get("orientations", [])) else 0.0)

        # distances : MAE et F1 sur la presence/absence
        mae, presence_f1 = distance_metrics(pred.get("relations"), true["relations"])
        if mae is not None:
            metrics["distance_mae"].append(mae)
        if presence_f1 is not None:
            metrics["distance_presence_f1"].append(presence_f1)

        # coreference : verifier qu'un objet reference deux fois n'est pas duplique
        true_counts = {}
        for id_ in true["objets"]:
            base = id_.rsplit("_", 1)[0] if id_[-1].isdigit() else id_
            true_counts[base] = true_counts.get(base, 0) + 1
        coref_objects = [id_ for id_, cnt in true_counts.items() if cnt == 1]
        if coref_objects:
            pred_bases = [id_.rsplit("_", 1)[0] if id_[-1].isdigit() else id_ for id_ in (pred.get("objets") or [])]
            no_double = all(pred_bases.count(b) == 1 for b in coref_objects)
            metrics["coref_no_double"].append(1.0 if no_double else 0.0)

        # multi-instance : verifier que les _2 sont bien presents
        multi_objects = [id_ for id_ in true["objets"] if id_[-1].isdigit()]
        if multi_objects:
            pred_objets = set(pred.get("objets") or [])
            all_present = all(id_ in pred_objets for id_ in multi_objects)
            metrics["multi_instance_ok"].append(1.0 if all_present else 0.0)

        # held-out : generalisation sur objets non vus
        if is_held_out(ex):
            metrics["held_out_f1"].append(f1_objects(pred.get("objets"), true["objets"]))

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    return {k: avg(v) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Dossier du modele fine-tune")
    parser.add_argument("--base",  default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--test",  required=True, help="Fichier JSONL de test")
    parser.add_argument("--n",     type=int, default=None, help="Nombre d'exemples (None = tous)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.model)
    model.eval()

    examples = load_jsonl(args.test)
    if args.n:
        examples = examples[:args.n]
    print(f"Evaluation sur {len(examples)} exemples...")

    results = evaluate(model, tokenizer, examples, device)

    print("\n--- Resultats ---")
    labels = {
        "f1_objets":                "F1 objets",
        "root_acc":                 "Root accuracy",
        "relation_exact_match":     "Relation exact match",
        "orientations_exact_match": "Orientations exact match",
        "distance_mae":             "Distance MAE (m)",
        "distance_presence_f1":     "Distance presence F1",
        "coref_no_double":          "Coreference (pas de doublon)",
        "multi_instance_ok":        "Multi-instance (_2 present)",
        "held_out_f1":              "Generalisation (objets non vus)",
    }
    for key, label in labels.items():
        val = results[key]
        if val is not None:
            print(f"  {label:<35} {val:.3f}")
        else:
            print(f"  {label:<35} N/A")


if __name__ == "__main__":
    main()
