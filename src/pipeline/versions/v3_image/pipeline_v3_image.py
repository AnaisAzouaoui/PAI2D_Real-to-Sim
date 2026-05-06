import base64
import io
import os
import re
import requests
from collections import defaultdict
from pathlib import Path
from PIL import Image
import json
from dotenv import load_dotenv
from openai import OpenAI
from pipeline.utils.ollama_client import URL
from pipeline.utils.groundingdino_detector import detect_contours
from pipeline.versions.v1_llm_prim.object_recognition.object_rec_v1_1_embedding import object_rec
from pipeline.sceneBuilding import initPosAndQuat, processRelations, processOrientations, get_genesis_dimensions
from pipeline.itemSpec import loadScale

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

RE_JSON = re.compile(r"\{.*\}", re.DOTALL)
RE_CODE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
RE_SUFFIX = re.compile(r'_\d+$')


# OpenAI  -> "gpt-4o", "gpt-4o-mini"
# Ollama  -> "llama3.2-vision"
VISION_MODEL = "gpt-4o"

VISION_TIMEOUT = 600
MAX_IMAGE_SIZE = 1024   

SCENE_WIDTH = 5.0
SCENE_DEPTH = 5.0
DEPTH_NUDGE = 0.3

VERTICAL_RELATIONS = {"on", "under", "inside"}
DEPTH_RELATIONS    = {"in_front_of", "behind"}

VALID_ORIENTATIONS = {
    "default", "turn_left", "turn_right", "turn_around",
    "tip_forward", "tip_backward", "tip_left", "tip_right", "upside_down",
}

openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=key)
    return openai_client


def is_openai_model(model):
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


# -----------------------------------------------------------------------------
# IMAGE LOADING
# -----------------------------------------------------------------------------

def open_image(source):
    img = Image.open(str(source))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_images(image_sources):
    # garde la signature historique: retourne directement les base64 pour le VLM
    images = []
    for source in image_sources:
        img = open_image(source)
        images.append(pil_to_base64(img))
        print(f"[load_images] {source} -> {img.size} RGB")
    return images


# -----------------------------------------------------------------------------
# VISION LLM CALLS 
# -----------------------------------------------------------------------------

def call_openai_vision(images, system_prompt, user_prompt):
    client = get_openai_client()

    # build user content: text + one block per image
    content = [{"type": "text", "text": user_prompt}]
    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high",
            },
        })

    print(f"[call_vision_llm] sending {len(images)} image(s) to {VISION_MODEL}...")
    response = get_openai_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user","content": content},
        ],
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    print("[call_vision_llm] response received.")
    return json.loads(response.choices[0].message.content)


def call_ollama_vision(images, system_prompt, user_prompt):
    payload = {
        "model": VISION_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "images": images,
        "stream": True,
        "options": {"temperature": 0, "num_predict": 768},
    }
    print(f"[call_vision_llm] sending request to Ollama ({VISION_MODEL}, streaming)...")
    response = requests.post(URL, json=payload, stream=True, timeout=VISION_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama vision error (HTTP {response.status_code}): {response.text}")

    full_response = ""
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        full_response += chunk.get("response", "")
        if chunk.get("done"):
            break

    print("[call_vision_llm] response received.")
    m = RE_CODE_BLOCK.search(full_response)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = RE_JSON.search(full_response)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"VLM did not return valid JSON. Raw response:\n{full_response[:400]}")


def call_vision_llm(images, system_prompt, user_prompt):
    if is_openai_model(VISION_MODEL):
        return call_openai_vision(images, system_prompt, user_prompt)
    return call_ollama_vision(images, system_prompt, user_prompt)


# -----------------------------------------------------------------------------
# STEP 1 — VISUAL DETECTION
# -----------------------------------------------------------------------------

def detect_and_estimate(images):
    """
    VLM analyses image(s) and returns:
      - view_type : "frontal" or "top_down"
      - objects   : list of {label, pos_norm, orientation}
      - relations : list of {type, subject, object}
    """
    print(f"[detect_and_estimate] appel du VLM sur {len(images)} image(s)...")
    system_prompt = """You are a strict JSON API. Your goal is to reconstruct a 3D scene \
that matches the image(s) as closely as possible.

TASK:
1. Determine the view_type:
   - "frontal"  : camera is roughly horizontal, looking at the scene from the side (most room photos)
   - "top_down" : camera is above, looking straight down at the scene

2. Detect every distinct visible object in the image(s).

3. For each object estimate:
   - pos_norm: [x, y] — normalised centre of the object IN THE IMAGE.
       x = 0.0 -> left edge,  x = 1.0 -> right edge.
       y = 0.0 -> top edge,   y = 1.0 -> bottom edge.
       Be as precise as possible: use the actual centre of the object's bounding box.
   - label_en: short English label for the same object (e.g. "washing machine", "mug", "banana").
       Used by an English-only object detector. Keep it simple and generic.
   - orientation: choose ONE value that best describes the visible rotation:
       "default"      — normal upright, no visible rotation
       "turn_left"    — rotated 90 left around the vertical axis (seen from above)
       "turn_right"   — rotated 90 right around the vertical axis (seen from above)
       "turn_around"  — rotated 180 around vertical axis (facing opposite direction)
       "tip_forward"  — tilted/fallen forward toward the camera
       "tip_backward" — tilted/fallen backward away from the camera
       "tip_left"     — tilted/fallen to the left
       "tip_right"    — tilted/fallen to the right
       "upside_down"  — completely inverted
       -> Use "default" unless a rotation is CLEARLY visible.

4. Extract ONLY the spatial relations that are unambiguously visible:
   - "on"         : A rests on top of B
   - "under"      : A is below B
   - "inside"     : A is inside B
   - "left_of"    : A is to the left of B
   - "right_of"   : A is to the right of B
   - "in_front_of": A is closer to camera than B
   - "behind"     : A is further from camera than B
   - "against"    : A is pressed flat against B

OUTPUT FORMAT — return ONLY this JSON, nothing else:
{
  "view_type": "<frontal|top_down>",
  "objects": [
    {
      "label": "<nom en francais>",
      "label_en": "<short english name>",
      "pos_norm": [<x 0.0-1.0>, <y 0.0-1.0>],
      "orientation": "<value from the list above>"
    }
  ],
  "relations": [
    {"type": "<relation>", "subject": "<label_A>", "object": "<label_B>"}
  ]
}

RULES:
- One entry per distinct instance (two bananas = two entries with different pos_norm).
- Simple French labels: "frigo", "lave-linge", "poubelle", "banane", "mug", etc.
- subject/object in relations must be EXACT labels from the objects list above.
- "relations": [] if nothing is clearly visible.
- Multiple images = same scene from different angles — combine them.
- Return valid JSON only."""

    result = call_vision_llm(
        images,
        system_prompt,
        "Analyse the image(s) as precisely as possible and return the JSON.",
    )
    if not result.get("objects"):
        print("[detect_and_estimate] aucun objet trouve dans la reponse du VLM.")
        return {"view_type": "frontal", "objects": [], "relations": []}
    if result.get("view_type") not in ("frontal", "top_down"):
        result["view_type"] = "frontal"
    print(f"[detect_and_estimate] termine -> vue={result['view_type']}, {len(result['objects'])} objet(s), {len(result.get('relations', []))} relation(s)")
    return result


# -----------------------------------------------------------------------------
# COORDINATE CONVERSION
# -----------------------------------------------------------------------------

def pos_norm_to_world(pos_norm, view_type):
    """
    Convert normalised image coordinates [0..1, 0..1] to world (x, y).

    World convention (from sceneBuilding.apply_relation):
      x = depth axis  (in_front_of = more x, behind = less x)
      y = lateral axis (right_of = more y, left_of = less y)
      z = vertical     (on = more z)

    Frontal view (camera horizontal, looking along -x):
      image x (left-right)         -> world y  (centered)
      image y (top=far, bot=near)  -> world x  (top -> larger x = farther)

    Top-down view (camera above, looking along -z):
      image x -> world x  (centered)
      image y -> world y  (inverted: top of image = positive y)
    """
    px, py = pos_norm
    if view_type == "top_down":
        world_x = round((px - 0.5) * SCENE_WIDTH,  3)
        world_y = round((0.5 - py) * SCENE_DEPTH, 3)
    else:  # frontal (default)
        world_x = round((0.5 - py) * SCENE_DEPTH, 3)
        world_y = round((px - 0.5) * SCENE_WIDTH,  3)
    return world_x, world_y


# -----------------------------------------------------------------------------
# ARRANGER LA DEPTH POUR FRONT VIEW
# -----------------------------------------------------------------------------

def nudge_depth_from_relations(items, depth_relations, view_type):
    if view_type != "frontal" or not depth_relations:
        return

    items_dict = {item["id"]: item for item in items}
    for rel in depth_relations:
        subj = items_dict.get(rel["subject"])
        obj  = items_dict.get(rel["object"])
        if not subj or not obj:
            continue
        subj_x = subj["pos"][0]
        obj_x  = obj["pos"][0]
        if rel["type"] == "in_front_of" and subj_x <= obj_x:
            subj["pos"][0] = obj_x + DEPTH_NUDGE
        elif rel["type"] == "behind" and subj_x >= obj_x:
            subj["pos"][0] = obj_x - DEPTH_NUDGE


# -----------------------------------------------------------------------------
# LABEL MAPPING HELPERS
# -----------------------------------------------------------------------------

def build_cat_by_base(objet_reconnus):
    cat_by_base = defaultdict(list)
    for cat_label in objet_reconnus:
        base = RE_SUFFIX.sub("", cat_label)
        cat_by_base[base].append(cat_label)
    for base in cat_by_base:
        cat_by_base[base].sort()
    return cat_by_base


def build_item_estimates(detected_objects, objet_reconnus, cat_by_base):
    base_counters  = defaultdict(int)
    item_estimates = {}
    for obj in detected_objects:
        vl       = obj["label"]
        cat_list = cat_by_base.get(vl, [])
        idx      = base_counters[vl]
        if idx < len(cat_list):
            item_estimates[cat_list[idx]] = obj
        base_counters[vl] += 1
    for cat_label in objet_reconnus:
        item_estimates.setdefault(cat_label, {})
    return item_estimates


def assign_contours(detected_objects, detections, image_size):
    # Pour chaque objet detecte par le VLM, on cherche le contour GroundingDINO le plus proche
    # de son pos_norm parmi ceux dont le label correspond. Le contour trouve est stocke
    # dans obj["contour"] et son centre ecrase pos_norm (plus precis).
    img_w, img_h = image_size
    restantes = list(detections)
    for obj in detected_objects:
        label_en = (obj.get("label_en") or obj.get("label") or "").strip().lower()
        if not label_en:
            continue
        candidates = [d for d in restantes if d["label"] in label_en or label_en in d["label"]]
        if not candidates:
            continue
        px, py = obj.get("pos_norm", [0.5, 0.5])
        cible = (px * img_w, py * img_h)
        def dist_sq(d):
            cx = (d["contour"][0] + d["contour"][2]) / 2
            cy = (d["contour"][1] + d["contour"][3]) / 2
            return (cx - cible[0])**2 + (cy - cible[1])**2
        meilleure = min(candidates, key=dist_sq)
        obj["contour"] = meilleure["contour"]
        cx = (meilleure["contour"][0] + meilleure["contour"][2]) / 2
        cy = (meilleure["contour"][1] + meilleure["contour"][3]) / 2
        obj["pos_norm"] = [cx / img_w, cy / img_h]
        restantes.remove(meilleure)


def load_depth_map(path):
    # accepte .npy (numpy) ou n'importe quelle image (PNG/JPG/EXR via PIL)
    # convention: valeurs en metres, plus grand = plus loin de la camera
    import numpy as np
    if path.lower().endswith(".npy"):
        return np.load(path)
    img = Image.open(path)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def sample_depth_at_contour(depth_map, contour):
    h, w = depth_map.shape[:2]
    cx = int(round((contour[0] + contour[2]) / 2))
    cy = int(round((contour[1] + contour[3]) / 2))
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))
    return float(depth_map[cy, cx])


def remap_relations(relations, cat_by_base, valid_labels):
    remapped = []
    for rel in relations:
        subj_list = cat_by_base.get(rel["subject"], [])
        obj_list  = cat_by_base.get(rel["object"],  [])
        subj = subj_list[0] if subj_list else rel["subject"]
        obj  = obj_list[0]  if obj_list  else rel["object"]
        if subj in valid_labels and obj in valid_labels:
            remapped.append({"type": rel["type"], "subject": subj, "object": obj})
    return remapped


# def find_root(catalogue_labels, vertical_relations):
#     """Root = object not resting on anything. Fallback: first label."""
#     supported = {r["subject"] for r in vertical_relations if r["type"] in VERTICAL_RELATIONS}
#     for label in catalogue_labels:
#         if label not in supported:
#             return label
#     return next(iter(catalogue_labels))


# -----------------------------------------------------------------------------
# PATH RESOLUTION
# -----------------------------------------------------------------------------

def resolve_path(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ["mobility.urdf", "kinbody.xml", "textured.obj", "nontextured.stl", "nontextured.ply"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        google_16k = os.path.join(path, "google_16k")
        if os.path.isdir(google_16k):
            for name in ["textured.obj", "nontextured.stl", "nontextured.ply"]:
                candidate = os.path.join(google_16k, name)
                if os.path.exists(candidate):
                    return candidate
        for ext in (".obj", ".stl", ".ply"):
            for f in sorted(os.listdir(path)):
                if f.endswith(ext) and "collision" not in f.lower():
                    return os.path.join(path, f)
    return path


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def scene_from_image(image_paths, depth_map_paths=None, dims_fn=None):
    # Pipeline complet:
    #   1. Charge les images (PIL et base64)
    #   2. VLM gpt-4o -> labels FR/EN, relations, orientations, view_type
    #   3. GroundingDINO -> contours 2D pixel-accurate par label
    #   4. Embedding -> mapping vers le catalogue
    #   5. Genesis -> dimensions reelles 3D
    #   6. Conversion image -> monde (centre du contour pour plus de precision)
    #   7. Si depth_map fournie, on raffine world_x des objets racines
    #   8. processRelations pour le z, puis on repositionne les enfants on/inside
    #      avec leur position relative DANS le contour du parent
    #
    # depth_map_paths: liste optionnelle de chemins (un par image), meme resolution
    # que les images apres open_image. Format: .npy ou image PIL, valeurs en metres.
    print(f"\n[scene_from_image] === debut du pipeline sur {len(image_paths)} image(s) ===")
    images_pil = [open_image(p) for p in image_paths]
    images_b64 = [pil_to_base64(img) for img in images_pil]

    print("[scene_from_image] etape 1/6 - detection visuelle (gpt-4o)...")
    detected_raw = detect_and_estimate(images_b64)
    detected_objects = detected_raw.get("objects", [])
    detected_relations = detected_raw.get("relations", [])
    view_type = detected_raw.get("view_type", "frontal")
    print(f"[scene_from_image] view_type={view_type}, {len(detected_objects)} objets detectes")

    if not detected_objects:
        return []

    labels = [o["label"] for o in detected_objects]
    print(f"[scene_from_image] labels FR: {labels}")

    print("[scene_from_image] etape 2/6 - contours 2D (GroundingDINO)...")
    labels_en = [o.get("label_en") or o.get("label") for o in detected_objects]
    detections = detect_contours(images_pil[0], labels_en)
    print(f"[scene_from_image] {len(detections)} contour(s) detecte(s)")
    assign_contours(detected_objects, detections, images_pil[0].size)
    nb_avec_contour = sum(1 for o in detected_objects if "contour" in o)
    print(f"[scene_from_image] {nb_avec_contour}/{len(detected_objects)} objets associes a un contour")

    print("[scene_from_image] etape 3/6 - reconnaissance catalogue (embedding)...")
    objet_reconnus, non_reconnus = object_rec(labels)
    print(f"[scene_from_image] {len(objet_reconnus)} objet(s) trouve(s) dans le catalogue")
    if non_reconnus:
        print(f"[scene_from_image] pas dans le catalogue: {non_reconnus}")
    if not objet_reconnus:
        return []

    valid_labels = set(objet_reconnus.keys())
    cat_by_base = build_cat_by_base(objet_reconnus)
    item_estimates = build_item_estimates(detected_objects, objet_reconnus, cat_by_base)

    all_relations = remap_relations(detected_relations, cat_by_base, valid_labels)
    vertical_relations = [r for r in all_relations if r["type"] in VERTICAL_RELATIONS]
    depth_relations = [r for r in all_relations if r["type"] in DEPTH_RELATIONS]
    print(f"[scene_from_image] {len(all_relations)} relation(s) ({len(vertical_relations)} verticales, {len(depth_relations)} profondeur)")

    items = [
        {
            "id": cat_label,
            "urdf": info["urdf"],
            "path": resolve_path(info["path"]),
            "dimensions": info.get("dimensions") or [1.0, 1.0, 1.0],
            **{k: v for k, v in loadScale({"urdf": info["urdf"]}).items() if k in ("scale", "default_quat")},
        }
        for cat_label, info in objet_reconnus.items()
    ]

    orientations = [
        {"id": cat_label, "turn": item_estimates[cat_label].get("orientation", "default")}
        for cat_label in objet_reconnus
        if item_estimates.get(cat_label, {}).get("orientation", "default") != "default"
           and item_estimates.get(cat_label, {}).get("orientation") in VALID_ORIENTATIONS
    ]
    print(f"[scene_from_image] etape 4/6 - orientations ({len(orientations)} non-default) + dimensions Genesis...")
    items = (dims_fn or get_genesis_dimensions)(items, orientations)

    # Depth map optionnelle
    depth_map = None
    if depth_map_paths:
        print(f"[scene_from_image] depth map fournie: {depth_map_paths[0]}")
        depth_map = load_depth_map(depth_map_paths[0])

    print(f"[scene_from_image] etape 5/6 - conversion coords image -> monde (vue={view_type})...")
    for item in items:
        est = item_estimates.get(item["id"], {})
        pos_norm = est.get("pos_norm", [0.5, 0.5])
        world_x, world_y = pos_norm_to_world(pos_norm, view_type)
        # Si on a une depth map et un contour, on remplace world_x par la mesure de profondeur
        if depth_map is not None and "contour" in est:
            world_x = round(sample_depth_at_contour(depth_map, est["contour"]), 3)
        _, _, z = item["pos"]
        item["pos"] = [world_x, world_y, z]
        print(f"  {item['id']}: monde=({world_x:.3f}, {world_y:.3f}, {z:.3f})")

    print(f"[scene_from_image] etape 6/6 - placement vertical + repositionnement enfants...")
    on_parent_map = {
        r["subject"]: r["object"]
        for r in vertical_relations
        if r["type"] in ("on", "inside")
    }
    image_xy = {item["id"]: item["pos"][:2] for item in items}
    items = processRelations(items, vertical_relations)
    items_dict = {item["id"]: item for item in items}

    # Pour les enfants on/inside: on utilise leur position relative DANS le contour du parent
    # pour les placer precisement sur la surface du parent dans le monde 3D.
    for item in items:
        parent_id = on_parent_map.get(item["id"])
        parent = items_dict.get(parent_id) if parent_id else None
        if not parent:
            # objet racine: on garde la position calculee depuis pos_norm_to_world
            item["pos"][0], item["pos"][1] = image_xy[item["id"]]
            continue

        child_est = item_estimates.get(item["id"], {})
        parent_est = item_estimates.get(parent_id, {})
        child_contour = child_est.get("contour")
        parent_contour = parent_est.get("contour")

        parent_dx = parent["dimensions"][0]
        parent_dy = parent["dimensions"][1]
        # marge de securite de 4 cm : l'AABB Genesis n'est pas parfaitement alignee
        # avec le rendu visuel, donc sans ca l'objet enfant depasse legerement le bord
        SURFACE_MARGIN = 0.04
        margin_x = max(0.0, (parent_dx - item["dimensions"][0]) / 2 - SURFACE_MARGIN)
        margin_y = max(0.0, (parent_dy - item["dimensions"][1]) / 2 - SURFACE_MARGIN)

        # Si on n'a pas les deux contours, fallback: meme profondeur que le parent, latera depuis pos_norm
        if not (child_contour and parent_contour):
            ix, iy = image_xy[item["id"]]
            px, py = parent["pos"][0], parent["pos"][1]
            item["pos"][1] = max(py - margin_y, min(iy, py + margin_y))
            item["pos"][0] = px if view_type == "frontal" else max(px - margin_x, min(ix, px + margin_x))
            continue

        # Position relative du centre de l'enfant dans le contour du parent (clamp [0,1])
        cx = (child_contour[0] + child_contour[2]) / 2
        cy = (child_contour[1] + child_contour[3]) / 2
        pw_img = max(1.0, parent_contour[2] - parent_contour[0])
        ph_img = max(1.0, parent_contour[3] - parent_contour[1])
        rel_x = max(0.0, min(1.0, (cx - parent_contour[0]) / pw_img))
        rel_y = max(0.0, min(1.0, (cy - parent_contour[1]) / ph_img))

        px, py = parent["pos"][0], parent["pos"][1]
        if view_type == "frontal":
            # image_x -> world_y (lateral). image_y encode la hauteur, pas la profondeur.
            target_y = py - parent_dy / 2 + rel_x * parent_dy
            item["pos"][1] = max(py - margin_y, min(target_y, py + margin_y))
            item["pos"][0] = px
        else:
            # top_down: image_x -> world_x, image_y -> world_y (inverse)
            target_x = px - parent_dx / 2 + rel_x * parent_dx
            item["pos"][0] = max(px - margin_x, min(target_x, px + margin_x))
            target_y = py + parent_dy / 2 - rel_y * parent_dy
            item["pos"][1] = max(py - margin_y, min(target_y, py + margin_y))

    if depth_map is None:
        nudge_depth_from_relations(items, depth_relations, view_type)

    print(f"[scene_from_image] === pipeline termine: {len(items)} objet(s) place(s) ===\n")
    return items
