import random
from object_pool import (
    OBJECT_POOL,
    TAILLE_SCORE,
    RELATION_TYPES,
    RELATION_TYPES_WITH_DISTANCE,
    ORIENTATIONS,
    DISTANCE_RANGES_BY_SIZE,
    DISTANCE_ROUND,
    DEFAULT_DISTANCE_PROB,
    DEFAULT_ORIENTATION_PROB,
)

CONFIGS = [
    "basic", "with_distance", "with_orientation", "dense",
    "coreference", "multi_instance", "longue",
    "no_relation", "negation", "anglais",
]

# total = 1.00
CONFIG_WEIGHTS = [
    0.25,  # basic
    0.12,  # with_distance
    0.10,  # with_orientation
    0.10,  # dense
    0.10,  # coreference
    0.10,  # multi_instance
    0.08,  # longue
    0.06,  # no_relation
    0.05,  # negation
    0.04,  # anglais
]

# relations semantiquement valides selon la taille des objets
VALID_RELATIONS_BY_SIZE = {
    "on":           [("petit", "grand"), ("petit", "moyen"), ("moyen", "grand"), ("petit", "petit"), ("moyen", "moyen")],
    "under":        [("grand", "petit"), ("moyen", "petit"), ("grand", "moyen")],
    "left_of":      "any",
    "right_of":     "any",
    "in_front_of":  "any",
    "behind":       "any",
    "against":      "any",
    "inside":       [("petit", "grand"), ("petit", "moyen"), ("moyen", "grand")],
}

EN_OBJECTS = {
    "washing_machine", "refrigerator", "dishwasher", "sofa", "wardrobe",
    "bookshelf", "desk", "chair", "lamp", "plant", "laptop", "bottle",
    "cup", "plate", "apple", "banana", "book", "pen", "key", "remote"
}


def get_base(id_):
    parts = id_.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return id_


def taille(id_):
    return OBJECT_POOL.get(get_base(id_), "moyen")


def assign_ids(objects_drawn):
    counts = {}
    ids = []
    for obj in objects_drawn:
        counts[obj] = counts.get(obj, 0) + 1
        ids.append(obj if counts[obj] == 1 else f"{obj}_{counts[obj]}")
    return ids


def is_valid_relation(rel_type, subj_id, obj_id):
    subj_taille = taille(subj_id)
    obj_taille  = taille(obj_id)
    rule = VALID_RELATIONS_BY_SIZE[rel_type]
    if rule == "any":
        return True
    return (subj_taille, obj_taille) in rule


def sample_distance(subj_id, obj_id):
    """Tire une distance realiste en metres selon la taille du plus grand des deux objets.
    Retourne un float arrondi a DISTANCE_ROUND."""
    sizes = (taille(subj_id), taille(obj_id))
    if "grand" in sizes:
        bucket = "grand"
    elif "moyen" in sizes:
        bucket = "moyen"
    else:
        bucket = "petit"
    lo, hi = DISTANCE_RANGES_BY_SIZE[bucket]
    raw = random.uniform(lo, hi)
    # arrondi au pas DISTANCE_ROUND, garde 2 decimales pour eviter les flottants longs
    return round(round(raw / DISTANCE_ROUND) * DISTANCE_ROUND, 2)


def sample_relations(ids, n_relations):
    relations = []
    attempts = 0
    while len(relations) < n_relations and attempts < 200:
        attempts += 1
        rel_type = random.choice(RELATION_TYPES)
        subj, obj = random.sample(ids, 2)
        if not is_valid_relation(rel_type, subj, obj):
            continue
        # une seule relation par paire d'objets (dans n'importe quel sens)
        pair = {subj, obj}
        if any({r["subject"], r["object"]} == pair for r in relations):
            continue
        relations.append({"type": rel_type, "subject": subj, "object": obj})

    # garantit que chaque objet apparait dans au moins une relation
    covered = set()
    for r in relations:
        covered.add(r["subject"])
        covered.add(r["object"])
    for id_ in ids:
        if id_ not in covered:
            others = [i for i in ids if i != id_]
            if not others:
                continue
            partner = random.choice(others)
            rel_type = random.choice(RELATION_TYPES)
            if not is_valid_relation(rel_type, id_, partner):
                rel_type = "left_of"
            relations.append({"type": rel_type, "subject": id_, "object": partner})

    return relations


def inject_distances(relations, prob=DEFAULT_DISTANCE_PROB, min_count=0):
    """Ajoute aleatoirement un champ 'distance' aux relations eligible
    Si min_count > 0, force au moins ce nombre de distances (si possible)."""
    eligible = [r for r in relations if r["type"] in RELATION_TYPES_WITH_DISTANCE]
    for r in eligible:
        if random.random() < prob:
            r["distance"] = sample_distance(r["subject"], r["object"])

    # garantit min_count distances
    current = sum(1 for r in eligible if "distance" in r)
    missing = max(0, min_count - current)
    if missing > 0:
        without = [r for r in eligible if "distance" not in r]
        random.shuffle(without)
        for r in without[:missing]:
            r["distance"] = sample_distance(r["subject"], r["object"])
    return relations


def sample_orientations(ids, prob=DEFAULT_ORIENTATION_PROB, min_count=0):
    """Pour chaque id, proba prob d'avoir une orientation tiree dans ORIENTATIONS
    Si min_count > 0, force au moins ce nombre d'orientations"""
    orientations = []
    for id_ in ids:
        if random.random() < prob:
            orientations.append({"id": id_, "turn": random.choice(ORIENTATIONS)})

    missing = max(0, min_count- len(orientations))
    if missing > 0:
        oriented_ids = {o["id"] for o in orientations}
        candidates = [i for i in ids if i not in oriented_ids]
        random.shuffle(candidates)
        for id_ in candidates[:missing]:
            orientations.append({"id": id_, "turn": random.choice(ORIENTATIONS)})
    return orientations


def compute_root(ids, relations):
    scores = {}
    for id_ in ids:
        taille = taille(id_)
        nb_rel = sum(1 for r in relations if r["subject"] == id_ or r["object"] == id_)
        scores[id_] = TAILLE_SCORE[taille] * (1 + nb_rel)
    return max(scores, key=lambda k: scores[k])


def generate_spec(config=None):
    if config is None:
        config = random.choices(CONFIGS, weights=CONFIG_WEIGHTS, k=1)[0]

    if config == "anglais":
        pool_keys = list(EN_OBJECTS)
    else:
        pool_keys = [k for k in OBJECT_POOL if k not in EN_OBJECTS]

    # sampling distance/orientation par defaut sur toutes les configs
    dist_min   = 0
    orient_min = 0

    if config == "basic":
        drawn = random.sample(pool_keys, random.randint(2, 3))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(1, 2))

    elif config == "with_distance":
        # force au moins 2 distances (sur 2-4 objets, donc 1-3 relations)
        drawn = random.sample(pool_keys, random.randint(2, 4))
        ids = assign_ids(drawn)
        # tire jusqu'a obtenir assez de relations eligibles a distance
        for _ in range(5):
            relations = sample_relations(ids, random.randint(2, min(4, len(ids))))
            n_eligible = sum(1 for r in relations if r["type"] in RELATION_TYPES_WITH_DISTANCE)
            if n_eligible >= 2:
                break
        dist_min = 2

    elif config == "with_orientation":
        # force au moins 2 orientations
        drawn = random.sample(pool_keys, random.randint(2, 4))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(1, min(3, len(ids))))
        orient_min = 2

    elif config == "dense":
        # scene complexe : 3-5 objets, >=2 distances + >=2 orientations
        drawn = random.sample(pool_keys, random.randint(3, 5))
        ids = assign_ids(drawn)
        for _ in range(5):
            relations = sample_relations(ids, random.randint(3, min(5, len(ids))))
            n_eligible = sum(1 for r in relations if r["type"] in RELATION_TYPES_WITH_DISTANCE)
            if n_eligible >= 2:
                break
        dist_min   = 2
        orient_min = 2

    elif config == "coreference":
        # un meme objet (ancre) est reference dans toutes les relations (meme ID, pas de doublon)
        # minimum 3 objets pour avoir au moins 2 relations sur l'ancre
        drawn = random.sample(pool_keys, random.randint(3, 4))
        ids = assign_ids(drawn)
        anchor = random.choice(ids)
        others = [i for i in ids if i != anchor]
        relations = []
        for other in others:  # tous les autres ont une relation avec l'ancre
            rel = random.choice(RELATION_TYPES)
            if not is_valid_relation(rel, other, anchor):
                rel = "left_of"
            relations.append({"type": rel, "subject": other, "object": anchor})

    elif config == "multi_instance":
        # meme type d'objet tire 2 fois -> id + id_2
        base_obj = random.choice(pool_keys)
        others   = random.sample([k for k in pool_keys if k != base_obj], random.randint(1, 3))
        drawn    = others + [base_obj, base_obj]
        random.shuffle(drawn)
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(2, min(4, len(ids))))

    elif config == "no_relation":
        drawn = random.sample(pool_keys, random.randint(2, 5))
        ids = assign_ids(drawn)
        relations = []

    elif config == "longue":
        drawn = random.choices(pool_keys, k=random.randint(4, 6))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(3, min(5, len(ids))))

    elif config == "negation":
        # un objet est mentionne comme absent -> ne doit PAS apparaitre dans objets
        drawn  = random.sample(pool_keys, random.randint(3, 5))
        ids_all = assign_ids(drawn)
        absent  = random.choice(ids_all)
        ids     = [i for i in ids_all if i != absent]
        relations = sample_relations(ids, random.randint(0, min(2, len(ids))))

    elif config == "anglais":
        drawn = random.sample(pool_keys, min(random.randint(2, 4), len(pool_keys)))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(1, min(3, len(ids))))

    else:
        raise ValueError(f"config inconnue : {config}")

    # injection distance + orientation (sampling par defaut + min force par la config)
    relations    = inject_distances(relations, prob=DEFAULT_DISTANCE_PROB, min_count=dist_min)
    orientations = sample_orientations(ids, prob=DEFAULT_ORIENTATION_PROB, min_count=orient_min)

    root = compute_root(ids, relations) if ids else None

    return {
        "objets":       ids,
        "root":         root,
        "relations":    relations,
        "orientations": orientations,
        "config":       config,
        "lang":         "en" if config == "anglais" else "fr",
    }


if __name__ == "__main__":
    for config in CONFIGS:
        spec = generate_spec(config=config)
        print(f"\n[{config}]")
        print(f"  objets       : {spec['objets']}")
        print(f"  root         : {spec['root']}")
        print(f"  relations    : {spec['relations']}")
        print(f"  orientations : {spec['orientations']}")
