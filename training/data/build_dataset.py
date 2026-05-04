"""
python build_dataset.py --n 600 --out dataset.jsonl --workers 3
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from spec_generator import generate_spec
from phrase_generator import generate_example

HELD_OUT_FR = {"baignoire", "piano", "moto", "aquarium", "carafe"}
HELD_OUT_EN = {"banana", "remote", "wardrobe", "dishwasher"}


def is_held_out(example):
    held_out = HELD_OUT_FR | HELD_OUT_EN
    for id_ in example["output"]["objets"]:
        base = id_.rsplit("_", 1)[0] if (id_[-1].isdigit() and "_" in id_) else id_
        if base in held_out:
            return True
    return False


def load_existing(raw_path):
    if not os.path.exists(raw_path):
        return []
    with open(raw_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_batch(n_target, raw_path, max_workers=3):
    examples = load_existing(raw_path)
    already  = len(examples)
    if already >= n_target:
        print(f"Dataset deja complet ({already} exemples).")
        return examples

    remaining = n_target - already
    # genere plus de specs que necessaire pour compenser rejets + paraphrase (facteur 1.4)
    n_brut = int(remaining * 1.4)

    specs = [generate_spec() for _ in range(n_brut)]

    # inject quelques objets held-out pour le test set
    fr_keys = list(HELD_OUT_FR)
    en_keys = list(HELD_OUT_EN)
    for i in random.sample(range(n_brut), min(int(n_brut * 0.25), n_brut)):
        spec = specs[i]
        pool = en_keys if spec.get("lang") == "en" else fr_keys
        base = random.choice(pool)
        if base not in spec["objets"]:
            spec["objets"] = [base] + spec["objets"]
            if len(spec["objets"]) > 1:
                partner = random.choice(spec["objets"][1:])
                spec["relations"].append({
                    "type": random.choice(["left_of", "right_of", "in_front_of", "behind"]),
                    "subject": base,
                    "object": partner,
                })

    rejected = 0
    f_raw = open(raw_path, "a", encoding="utf-8")

    bar = tqdm(
        total=remaining,
        desc="Exemples valides",
        unit="ex",
        bar_format="{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_example, spec): spec for spec in specs}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    # generate_example retourne une liste d'entrees (paraphrase incluse)
                    for entry in result:
                        f_raw.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        f_raw.flush()
                        examples.append(entry)
                        bar.update(1)
                        if len(examples) - already >= remaining:
                            break
                else:
                    rejected += 1
                    bar.set_postfix(rejetes=rejected)
                if len(examples) - already >= remaining:
                    break
    finally:
        bar.close()
        f_raw.close()

    total_processed = len(examples) - already + rejected
    pct_rejet = 100 * rejected / total_processed if total_processed > 0 else 0
    print(f"Taux de rejet : {rejected}/{total_processed} ({pct_rejet:.1f}%)")

    return examples


def split_and_save(examples, base):
    held   = [e for e in examples if is_held_out(e)]
    normal = [e for e in examples if not is_held_out(e)]
    random.shuffle(held)
    random.shuffle(normal)
    n = len(normal)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    train = normal[:n_train]
    val   = normal[n_train:n_train + n_val]
    test  = normal[n_train + n_val:] + held
    random.shuffle(test)

    for split, data in [("train", train), ("val", val), ("test", test)]:
        path = f"{base}_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {split:<6} : {len(data):>5} exemples -> {path}")

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=str, default="dataset.jsonl")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="Reprend depuis le fichier raw existant")
    args = parser.parse_args()

    base = args.out.replace(".jsonl", "")
    raw_path = f"{base}_raw.jsonl"
    if not args.resume and os.path.exists(raw_path):
        os.remove(raw_path)

    t0 = time.time()
    examples = generate_batch(args.n, raw_path, max_workers=args.workers)
    elapsed = (time.time() - t0) / 60
    print(f"\nGeneration terminee en {elapsed:.1f} min | {len(examples)} exemples valides")

    train, val, test = split_and_save(examples, base)
    print(f"Split : train={len(train)} | val={len(val)} | test={len(test)}")

    stats = {}
    for ex in examples:
        c = ex.get("config", "unknown")
        stats[c] = stats.get(c, 0) + 1
    print("\nDistribution des configs :")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v:>5} ({100*v/len(examples):.1f}%)")


if __name__ == "__main__":
    main()
