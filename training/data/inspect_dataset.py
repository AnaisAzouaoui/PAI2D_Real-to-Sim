"""
Inspecte et nettoie un fichier dataset raw.
Les entrees avec erreurs fatales sont supprimees et loguees.
Les entrees avec warnings sont gardees mais signalees.

Usage:
    python inspect_dataset.py [--input FILE] [--output FILE]

Par defaut:
    --input  dataset_v4_raw.jsonl
    --output dataset_v4_raw_clean.jsonl
"""
import argparse
import json
import os
import random
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from object_pool import (
    RELATION_TYPES,
    RELATION_TYPES_WITH_DISTANCE,
    ORIENTATIONS,
    DISTANCE_RANGE,
    OBJECT_NAMES_FR,
)

VALID_CONFIGS = {
    "basic", "with_distance", "with_orientation", "dense", "coreference",
    "multi_instance", "longue", "anglais", "stacking", "surface_commune", "aligned",
}

# du plus specifique au plus general pour eviter les faux positifs par sous-chaine
KEYWORD_CHECKS = [
    ("en dessous de", {"under"}),
    ("en dessous", {"under"}),
    ("in front of", {"in_front_of"}),
    ("to the left of", {"left_of"}),
    ("to the right of", {"right_of"}),
    ("cote a cote", {"left_of", "right_of"}),
    ("next to", {"left_of", "right_of"}),
    ("beside", {"left_of", "right_of"}),
    ("pres de", {"left_of", "right_of"}),
    ("face a", {"in_front_of"}),
    ("a cote", {"left_of", "right_of"}),
    ("a gauche", {"left_of"}),
    ("a droite", {"right_of"}),
    ("derriere", {"behind"}),
    ("behind", {"behind"}),
    ("devant", {"in_front_of"}),
    ("dessous", {"under"}),
    ("contre", {"against"}),
    ("against", {"against"}),
    ("inside", {"inside"}),
    ("left of", {"left_of"}),
    ("right of", {"right_of"}),
]


def normalize(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return (
        text.lower()
        .replace("'", " ").replace("'", " ").replace("ʼ", " ")
        .replace("`", " ").replace("'", " ")
        .replace("-", " ").replace("_", " ")
    )


def base_id(obj_id):
    # enleve le suffixe numerique type table_2 -> table
    if "_" in obj_id and obj_id.rsplit("_", 1)[1].isdigit():
        return obj_id.rsplit("_", 1)[0]
    return obj_id


def object_in_phrase(obj_id, phrase, config=""):
    b = base_id(obj_id)
    name = b if config == "anglais" else OBJECT_NAMES_FR.get(b, b)
    name_norm = normalize(name)
    phrase_norm = normalize(phrase)
    if name_norm in phrase_norm:
        return True
    # pour les noms composes (ex: table de nuit), tous les mots doivent etre la
    words = name_norm.split()
    return len(words) > 1 and all(w in phrase_norm for w in words)


def check_entry(entry):
    errors = []
    warnings = []

    # champs obligatoires au premier niveau
    for field in ("input", "config", "output"):
        if field not in entry:
            errors.append(f"champ manquant: '{field}'")
    if errors:
        return errors, warnings

    inp = entry["input"]
    cfg = entry["config"]
    out = entry["output"]

    for field in ("objets", "relations"):
        if field not in out:
            errors.append(f"output.{field} manquant")
    if errors:
        return errors, warnings

    # orientations absent mais pas cite dans la phrase -> on rajoute juste le champ vide
    if "orientations" not in out:
        out["orientations"] = []
        warnings.append("output.orientations absent -> champ vide ajoute")

    objets = out["objets"]
    rels = out["relations"]
    orients = out["orientations"]

    # --- checks structurels ---

    if cfg not in VALID_CONFIGS:
        errors.append(f"config inconnue: '{cfg}'")

    if not objets:
        errors.append("output.objets est vide")
        return errors, warnings

    seen_ids = {}
    for obj in objets:
        seen_ids[obj] = seen_ids.get(obj, 0) + 1
    for obj, cnt in seen_ids.items():
        if cnt > 1:
            errors.append(f"id duplique dans objets: '{obj}' ({cnt}x)")

    objets_set = set(objets)
    seen_rels = set()
    rel_types_out = set()

    for r in rels:
        rtype = r.get("type")
        subj = r.get("subject")
        obj_ = r.get("object")
        rel_types_out.add(rtype)

        if subj == obj_:
            errors.append(f"self-relation: sujet=objet='{subj}'")

        if rtype not in RELATION_TYPES:
            errors.append(f"type de relation inconnu: '{rtype}'")

        key = (rtype, subj, obj_)
        if key in seen_rels:
            errors.append(f"relation dupliquee: {rtype}({subj}, {obj_})")
        seen_rels.add(key)

        if "distance" in r:
            lo, hi = DISTANCE_RANGE
            d = r["distance"]
            if rtype not in RELATION_TYPES_WITH_DISTANCE:
                errors.append(f"distance sur relation non-distancable: '{rtype}'")
            elif not (lo <= d <= hi):
                errors.append(f"distance hors plage [{lo}, {hi}]: {d}")

        # warning seulement : JSON mal forme mais semantique potentiellement ok
        for role in ("subject", "object"):
            val = r.get(role)
            if val and val not in objets_set:
                warnings.append(f"relation cite '{val}' absent de output.objets")

    # relations contradictoires symetriques (A rel B et B rel A)
    for rtype, subj, obj_ in list(seen_rels):
        if (rtype, obj_, subj) in seen_rels and subj != obj_:
            errors.append(
                f"relations contradictoires: {subj} {rtype} {obj_} <-> {obj_} {rtype} {subj}"
            )

    for o in orients:
        if o.get("turn") not in ORIENTATIONS:
            errors.append(f"orientation inconnue: '{o.get('turn')}'")
        if o.get("id") not in objets_set:
            warnings.append(f"orientation cite '{o.get('id')}' absent de output.objets")

    # --- checks semantiques ---

    norm_inp = normalize(inp)
    padded_inp = f" {norm_inp} "

    # 1. mot-cle directionnel present dans l'input mais relation incompatible dans l'output
    # les espaces evitent les faux positifs ("la droite" ne doit pas matcher "a droite")
    for keyword, expected_types in KEYWORD_CHECKS:
        if f" {keyword} " in padded_inp:
            if not (expected_types & rel_types_out):
                errors.append(
                    f"semantique: '{keyword}' dans input mais aucune relation"
                    f" {sorted(expected_types)} dans output"
                )
            break  # un seul mot-cle suffit pour valider/flaguer cet axe

    # 2. chaque objet de l'output doit apparaitre dans la phrase d'input
    for obj_id in objets:
        if not object_in_phrase(obj_id, inp, config=cfg):
            errors.append(f"semantique: objet '{obj_id}' introuvable dans l'input")

    return errors, warnings


def print_section(title):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset_v4_raw.jsonl")
    parser.add_argument("--output", default="dataset_v4_raw_clean.jsonl")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input)
    output_path = os.path.join(base_dir, args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    total = len(entries)
    kept = []
    removed = []
    warned = []
    error_counter = Counter()

    for idx, entry in enumerate(entries):
        errors, warnings = check_entry(entry)
        if errors:
            removed.append((idx + 1, entry, errors))
            for e in errors:
                error_counter[e.split(":")[0].strip()] += 1
        else:
            kept.append(entry)
            if warnings:
                warned.append((idx + 1, entry, warnings))

    print_section(f"ENTREES SUPPRIMEES ({len(removed)})")
    for lineno, entry, errors in removed:
        inp = entry.get("input", "?")[:90]
        print(f"\n  [L{lineno}] {inp}")
        for e in errors:
            print(f"    ! {e}")

    if warned:
        print_section(f"WARNINGS - entrees conservees ({len(warned)})")
        for lineno, entry, warnings in warned:
            inp = entry.get("input", "?")[:90]
            print(f"\n  [L{lineno}] {inp}")
            for w in warnings:
                print(f"    ~ {w}")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print_section("STATS GENERALES")
    pct_rm = 100 * len(removed) / total if total else 0
    print(f"Total entrees : {total}")
    print(f"Supprimees: {len(removed):>5}  ({pct_rm:.1f}%)")
    print(f"Conservees : {len(kept):>5}  ({100 - pct_rm:.1f}%)")
    print(f" Avec warnings: {len(warned):>5}")

    print_section("DISTRIBUTION CONFIGS (dataset propre)")
    config_cnt = Counter(e.get("config", "?") for e in kept)
    for cfg, cnt in sorted(config_cnt.items(), key=lambda x: -x[1]):
        print(f"  {cfg:<22} {cnt:>5}  ({100*cnt/len(kept):.1f}%)")

    print_section("DISTRIBUTION TYPES DE RELATIONS")
    rel_cnt = Counter()
    for e in kept:
        for r in e["output"]["relations"]:
            rel_cnt[r.get("type", "?")] += 1
    total_rels = sum(rel_cnt.values())
    for rtype, cnt in sorted(rel_cnt.items(), key=lambda x: -x[1]):
        print(f"  {rtype:<15} {cnt:>6}  ({100*cnt/total_rels:.1f}%)")

    print_section("DISTRIBUTION ORIENTATIONS")
    orient_cnt = Counter()
    for e in kept:
        for o in e["output"]["orientations"]:
            orient_cnt[o.get("turn", "?")] += 1
    total_or = sum(orient_cnt.values()) or 1
    for turn, cnt in sorted(orient_cnt.items(), key=lambda x: -x[1]):
        print(f"  {turn:<15} {cnt:>6}  ({100*cnt/total_or:.1f}%)")

    print_section("MOYENNES PAR ENTREE (dataset propre)")
    n = len(kept)
    avg_obj = sum(len(e["output"]["objets"]) for e in kept) / n
    avg_rel = sum(len(e["output"]["relations"]) for e in kept) / n
    avg_or = sum(len(e["output"]["orientations"]) for e in kept) / n
    avg_dist = sum(
        sum(1 for r in e["output"]["relations"] if "distance" in r)
        for e in kept
    ) / n
    print(f"objets: {avg_obj:.2f}")
    print(f"relations: {avg_rel:.2f}")
    print(f"orientations : {avg_or:.2f}")
    print(f"distances : {avg_dist:.2f}")

    print_section("TOP ERREURS")
    for err, cnt in error_counter.most_common(15):
        print(f"{err:<45} {cnt:>5}")

    print(f"\nDataset propre -> {output_path}\n")

    split_and_save(kept, output_path)


HELD_OUT = {"baignoire", "piano", "moto", "aquarium", "carafe", "banana", "remote", "wardrobe", "dishwasher"}


def is_held_out(entry):
    for obj_id in entry["output"]["objets"]:
        b = base_id(obj_id)
        if b in HELD_OUT:
            return True
    return False


TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
# test = le reste (~5%)


def split_and_save(entries, clean_path):
    all_entries = list(entries)
    random.shuffle(all_entries)

    n = len(all_entries)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = all_entries[:n_train]
    val = all_entries[n_train:n_train + n_val]
    test = all_entries[n_train + n_val:]

    base = clean_path.replace("_clean.jsonl", "").replace("_raw.jsonl", "").replace(".jsonl", "")
    print_section("SPLIT TRAIN / VAL / TEST")
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        path = f"{base}_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {split_name:<6} : {len(data):>5} entrees -> {path}")


if __name__ == "__main__":
    main()
