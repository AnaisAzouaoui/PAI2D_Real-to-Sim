"""
Pipeline V1.1.1
Un seul appel Ollama vers phi3-scene (finetuned) pour extraire labels, root, relations, orientations.
Matching URDF via embeddings (meme logique que V1.1).
Scales via phi3:mini (meme logique que V1.1).
"""
import re
import sys
import os
import numpy as np

from pipeline.utils.catalogue import objets_list
from pipeline.versions.v1_llm_prim.placement.placement_v1_relations import (
    fix_ids_in_result, scale, final_json,
)
from pipeline.utils.helpers import fuzzy_match
from pipeline.sceneBuilding import buildScene

# project root (au-dessus de src/) pour importer training.inference.extractor
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.inference.extractor import extract_labels as _phi3_extract

# ---------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------
SEUIL = 0.55
embed_model    = None
catalogue_vecs = None
catalogue_keys = None


def _get_embed_model():
    global embed_model
    if embed_model is None:
        print("[v1_1_1] chargement modele embedding...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    return embed_model


def get_catalogue():
    global catalogue_vecs, catalogue_keys
    if catalogue_vecs is None:
        data = objets_list()
        catalogue_keys = list(data.keys())
        names = [data[k]["name"].strip() for k in catalogue_keys]
        catalogue_vecs = _get_embed_model().encode(names)
        print(f"[v1_1_1] catalogue encode : {len(catalogue_keys)} objets")
    return catalogue_vecs, catalogue_keys


def cosinus(query_vec, cat_vecs):
    q = query_vec / np.linalg.norm(query_vec)
    c = cat_vecs / np.linalg.norm(cat_vecs, axis=1, keepdims=True)
    return c @ q


# ---------------------------------------------------------------
# CACHE PHI3-SCENE
# ---------------------------------------------------------------
phi3_cache = {}


def phi3_result(prompt):
    """Appel phi3-scene unique par prompt (resultat mis en cache)"""
    if prompt not in phi3_cache:
        print("[v1_1_1] appel phi3-scene...")
        phi3_cache[prompt] = _phi3_extract(prompt)
        print(f"[v1_1_1] phi3-scene retourne : {phi3_cache[prompt]}")
    return phi3_cache[prompt]


# ---------------------------------------------------------------
# OBJECT RECOGNITION
# ---------------------------------------------------------------
SUFFIX_RE = re.compile(r'_\d+$')


def object_rec(prompt):
    """Labels depuis phi3-scene, matching URDF via embeddings.

    phi3-scene utilise deja la convention banane / banane_2 / banane_3
    donc on utilise ces labels directement comme IDs dans obj_reconnus
    Pour l'embedding on retire le suffixe _N pour  matcher le concept de base
    """
    phi3 = phi3_result(prompt)
    raw_labels = phi3.get("objets", [])

    if not raw_labels:
        return {}, []

    cat_vecs, cat_keys = get_catalogue()
    objects_data = objets_list()

    obj_reconnus = {}
    non_reconnus = []

    for label in raw_labels:
        label_id = str(label).lower().strip()
        base_query = SUFFIX_RE.sub('', label_id)  # "banane_2" -> "banane"

        query_vec  = _get_embed_model().encode(base_query)
        scores     = cosinus(query_vec, cat_vecs)
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < SEUIL:
            print(f"[v1_1_1] '{label_id}' -> score {best_score:.2f} sous le seuil -> non reconnu")
            non_reconnus.append(label_id)
            continue

        urdf = cat_keys[best_idx]
        print(f"[v1_1_1] '{label_id}' -> '{urdf}' (score {best_score:.2f})")

        obj_reconnus[label_id] = {
            "urdf":       urdf,
            "path":       objects_data[urdf]["path"],
            "dimensions": objects_data[urdf]["dimensions"],
        }

    print("[v1_1_1] Reconnus:", list(obj_reconnus.keys()))
    print("[v1_1_1] Non reconnus:", non_reconnus)
    return obj_reconnus, non_reconnus


# ---------------------------------------------------------------
# CORRECTION DES ORIENTATIONS
# ---------------------------------------------------------------
VALID_TURNS = {
    "tip_left", "tip_right", "tip_forward", "tip_backward",
    "upside_down", "turn_left", "turn_right", "turn_around",
}


def _fix_orientations(orientations, valid_ids):
    fixed = []
    for o in orientations:
        id_val = o.get("id", "")
        turn   = o.get("turn", "")
        matched = fuzzy_match(id_val, valid_ids)
        if not matched:
            print(f"[v1_1_1] orientation ignoree (ID inconnu) : {o}")
            continue
        if turn not in VALID_TURNS:
            print(f"[v1_1_1] orientation ignoree (turn invalide '{turn}') : {o}")
            continue
        fixed.append({"id": matched, "turn": turn})
    return fixed


# ---------------------------------------------------------------
# PLACEMENT
# ---------------------------------------------------------------
def place_scene(prompt, obj_reconnus):
    """Root, relations, orientations depuis phi3-scene. Scales via phi3:mini"""
    phi3 = phi3_result(prompt)  # depuis le cache
    valid_ids = set(obj_reconnus.keys())

    # relations depuis phi3-scene
    relations_data = {
        "relations": phi3.get("relations", []),
    }
    relations_data = fix_ids_in_result(relations_data, valid_ids)
    relations = relations_data["relations"]

    # orientations depuis phi3-scene
    orientations = _fix_orientations(phi3.get("orientations", []), valid_ids)

    # scales via phi3:mini (desactive temporairement — scale=1.0 pour tous)
    # scales_result = scale(prompt, obj_reconnus)
    # obj_reconnus  = final_json(obj_reconnus, scales_result)

    # construction des items
    items = []
    for label, info in obj_reconnus.items():
        items.append({
            "id": label,
            "urdf": info["urdf"],
            "path":info.get("path", ""),
            "dimensions": info.get("dimensions"),
            "scale": 1.0,
        })

    print("[v1_1_1] items:", items, "| relations:", relations, "| orientations:", orientations)
    items = buildScene(items, relations, orientations)
    for item in items:
        pos  = item.get("pos", (0, 0, 0))
        item["pos"] = list(pos)
        x, y, z, w  = item.get("quat", (0, 0, 0, 1))
        item["quat"] = [w, x, y, z]

    return items


def modify_scene(prompt, current_scene_json, obj_reconnus):
    """Meme interface que v1.1. Vide le cache pour forcer un nouvel appel phi3-scene"""
    phi3_cache.pop(prompt, None)
    return place_scene(prompt, obj_reconnus)
