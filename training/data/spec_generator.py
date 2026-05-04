import random
from object_pool import (
    OBJECT_POOL,
    EN_OBJECTS,
    RELATION_TYPES,
    RELATION_TYPES_WITH_DISTANCE,
    ORIENTATIONS,
    DISTANCE_RANGE,
    DISTANCE_ROUND,
    DISTANCE_PROB,
    ORIENTATION_PROB,
)

CONFIGS = [
    "basic",
    "with_distance",
    "with_orientation",
    "dense",
    "coreference",
    "multi_instance",
    "longue",
    "anglais",
    "stacking",
    "surface_commune",
    "aligned",
]

CONFIG_WEIGHTS = [
    0.19,  # basic
    0.09,  # with_distance
    0.03,  # with_orientation
    0.08,  # dense
    0.08,  # coreference
    0.08,  # multi_instance
    0.08,  # longue
    0.04,  # anglais
    0.11,  # stacking
    0.11,  # surface_commune
    0.11,  # aligned
]


def assign_ids(objects_drawn):
    counts = {}
    ids = []
    for obj in objects_drawn:
        counts[obj] = counts.get(obj, 0) + 1
        ids.append(obj if counts[obj] == 1 else f"{obj}_{counts[obj]}")
    return ids


def sample_distance():
    lo, hi = DISTANCE_RANGE
    raw = random.uniform(lo, hi)
    return round(round(raw / DISTANCE_ROUND) * DISTANCE_ROUND, 2)


def sample_relations(ids, n_relations):
    relations = []
    attempts = 0
    while len(relations) < n_relations and attempts < 200:
        attempts += 1
        rel_type = random.choice(RELATION_TYPES)
        subj, obj = random.sample(ids, 2)
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
            relations.append({"type": rel_type, "subject": id_, "object": partner})

    return relations


def inject_distances(relations, prob=DISTANCE_PROB, min_count=0):
    eligible = [r for r in relations if r["type"] in RELATION_TYPES_WITH_DISTANCE]
    for r in eligible:
        if "distance" not in r and random.random() < prob:
            r["distance"] = sample_distance()

    current = sum(1 for r in eligible if "distance" in r)
    missing = max(0, min_count - current)
    if missing > 0:
        without = [r for r in eligible if "distance" not in r]
        random.shuffle(without)
        for r in without[:missing]:
            r["distance"] = sample_distance()
    return relations


def sample_orientations(ids, prob=ORIENTATION_PROB, min_count=0):
    orientations = []
    for id_ in ids:
        if random.random() < prob:
            orientations.append({"id": id_, "turn": random.choice(ORIENTATIONS)})

    missing = max(0, min_count - len(orientations))
    if missing > 0:
        oriented_ids = {o["id"] for o in orientations}
        candidates = [i for i in ids if i not in oriented_ids]
        random.shuffle(candidates)
        for id_ in candidates[:missing]:
            orientations.append({"id": id_, "turn": random.choice(ORIENTATIONS)})
    return orientations


def generate_spec(config=None):
    if config is None:
        config = random.choices(CONFIGS, weights=CONFIG_WEIGHTS, k=1)[0]

    if config == "anglais":
        pool_keys = list(EN_OBJECTS)
    else:
        pool_keys = [k for k in OBJECT_POOL if k not in EN_OBJECTS]

    dist_min   = 0
    orient_min = 0

    if config == "basic":
        drawn = random.sample(pool_keys, random.randint(2, 3))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(1, 2))

    elif config == "with_distance":
        drawn = random.sample(pool_keys, random.randint(2, 4))
        ids = assign_ids(drawn)
        for _ in range(5):
            relations = sample_relations(ids, random.randint(2, min(4, len(ids))))
            n_eligible = sum(1 for r in relations if r["type"] in RELATION_TYPES_WITH_DISTANCE)
            if n_eligible >= 2:
                break
        dist_min = 2

    elif config == "with_orientation":
        drawn = random.sample(pool_keys, random.randint(2, 4))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(1, min(3, len(ids))))
        orient_min = 2

    elif config == "dense":
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
        drawn = random.sample(pool_keys, random.randint(3, 4))
        ids = assign_ids(drawn)
        anchor = random.choice(ids)
        others = [i for i in ids if i != anchor]
        relations = []
        for other in others:
            rel = random.choice(RELATION_TYPES)
            relations.append({"type": rel, "subject": other, "object": anchor})

    elif config == "multi_instance":
        base_obj = random.choice(pool_keys)
        others   = random.sample([k for k in pool_keys if k != base_obj], random.randint(1, 3))
        drawn    = others + [base_obj, base_obj]
        random.shuffle(drawn)
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(2, min(4, len(ids))))

    elif config == "longue":
        drawn = random.choices(pool_keys, k=random.randint(4, 6))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(3, min(5, len(ids))))

    elif config == "anglais":
        drawn = random.sample(pool_keys, min(random.randint(2, 4), len(pool_keys)))
        ids = assign_ids(drawn)
        relations = sample_relations(ids, random.randint(1, min(3, len(ids))))

    elif config == "stacking":
        # chaine de relations "on" : ids[0] on ids[1] on ids[2]...
        n = random.randint(3, 4)
        drawn = random.sample(pool_keys, n)
        ids = assign_ids(drawn)
        relations = []
        for i in range(len(ids) - 1):
            relations.append({"type": "on", "subject": ids[i], "object": ids[i + 1]})
        # relation laterale optionnelle entre deux elements de la pile
        if random.random() < 0.3 and len(ids) >= 3:
            a, b = random.sample(ids, 2)
            pair = {a, b}
            if not any({r["subject"], r["object"]} == pair for r in relations):
                lat = random.choice(["left_of", "right_of", "in_front_of", "behind"])
                relations.append({"type": lat, "subject": a, "object": b})

    elif config == "surface_commune":
        # 1 surface + 2-4 items poses dessus
        n_items = random.randint(2, 4)
        surface = random.choice(pool_keys)
        item_pool = [k for k in pool_keys if k != surface]
        items_drawn = random.sample(item_pool, n_items)
        drawn = [surface] + items_drawn
        ids = assign_ids(drawn)
        surface_id = ids[0]
        item_ids = ids[1:]
        relations = [{"type": "on", "subject": it, "object": surface_id} for it in item_ids]
        # relation laterale optionnelle entre items
        if len(item_ids) >= 2 and random.random() < 0.5:
            a, b = random.sample(item_ids, 2)
            lat = random.choice(["left_of", "right_of"])
            relations.append({"type": lat, "subject": a, "object": b})

    elif config == "aligned":
        # 3-4 objets alignes dans une direction, avec distances optionnelles
        n = random.randint(3, 4)
        drawn = random.sample(pool_keys, n)
        ids = assign_ids(drawn)
        direction = random.choice(["left_of", "right_of", "in_front_of", "behind"])
        relations = []
        for i in range(len(ids) - 1):
            rel = {"type": direction, "subject": ids[i], "object": ids[i + 1]}
            if random.random() < 0.4:
                rel["distance"] = sample_distance()
            relations.append(rel)

    else:
        raise ValueError(f"config inconnue : {config}")

    relations    = inject_distances(relations, prob=DISTANCE_PROB, min_count=dist_min)
    orientations = sample_orientations(ids, prob=ORIENTATION_PROB, min_count=orient_min)

    return {
        "objets":       ids,
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
        print(f"  relations    : {spec['relations']}")
        print(f"  orientations : {spec['orientations']}")
