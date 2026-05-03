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
from pipeline.versions.v1_llm_prim.object_recognition.object_rec_v1_1_embedding import object_rec
from pipeline.sceneBuilding import initPosAndQuat, processRelations, processOrientations

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

def load_images(image_sources):
    images = []
    for source in image_sources:
        img = Image.open(str(source))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        print(f"[load_images] {source} -> {img.size} RGB")
    return images


# -----------------------------------------------------------------------------
# VISION LLM CALLS 
# -----------------------------------------------------------------------------

def call_openai_vision(images, system_prompt, user_prompt):
    client = _get_openai_client()

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
    response = _get_openai_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
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
    return path


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def scene_from_image(image_paths):
    """
    Generate a 3D scene from one or more images.

    Pipeline:
      1. Load + resize images
      2. Vision LLM (OpenAI or Ollama) -> view_type, objects, relations
      3. Embedding object_rec -> catalogue mapping
      4. build_item_estimates -> ordered, collision-free assignment
      5. processOrientations -> quat + dimension swap
      6. initPosAndQuat + processRelations (vertical only) -> z
      7. pos_norm_to_world (view-aware) -> x, y
      8. nudge_depth_from_relations -> depth ordering (frontal only)

    Returns: list of dicts [{id, urdf, path, scale, quat, pos}, ...]
    """
    print(f"\n[scene_from_image] === debut du pipeline sur {len(image_paths)} image(s) ===")
    images = load_images(image_paths)

    print("[scene_from_image] etape 1/5 - detection visuelle...")
    detected_raw       = detect_and_estimate(images)
    detected_objects   = detected_raw.get("objects", [])
    detected_relations = detected_raw.get("relations", [])
    view_type          = detected_raw.get("view_type", "frontal")
    print(f"[scene_from_image] view_type={view_type}, {len(detected_objects)} objects detected")

    if not detected_objects:
        print("[scene_from_image] no objects detected.")
        return []

    labels = [o["label"] for o in detected_objects]
    print(f"[scene_from_image] labels detectes: {labels}")

    print("[scene_from_image] etape 2/5 - reconnaissance des objets (embedding)...")
    objet_reconnus, non_reconnus = object_rec(labels)
    print(f"[scene_from_image] {len(objet_reconnus)} objet(s) trouve(s) dans le catalogue")
    if non_reconnus:
        print(f"[scene_from_image] pas dans le catalogue: {non_reconnus}")
    if not objet_reconnus:
        print("[scene_from_image] aucun objet reconnu, arret.")
        return []

    valid_labels   = set(objet_reconnus.keys())
    cat_by_base    = build_cat_by_base(objet_reconnus)
    item_estimates = build_item_estimates(detected_objects, objet_reconnus, cat_by_base)

    all_relations      = remap_relations(detected_relations, cat_by_base, valid_labels)
    vertical_relations = [r for r in all_relations if r["type"] in VERTICAL_RELATIONS]
    depth_relations    = [r for r in all_relations if r["type"] in DEPTH_RELATIONS]
    print(f"[scene_from_image] {len(all_relations)} relation(s) remappee(s) ({len(vertical_relations)} verticales, {len(depth_relations)} profondeur)")

    items = [
        {
            "id":cat_label,
            "urdf":info["urdf"],
            "path":resolve_path(info["path"]),
            "dimensions": info.get("dimensions") or [1.0, 1.0, 1.0],
            "scale":1.0,
        }
        for cat_label, info in objet_reconnus.items()
    ]

    orientations = [
        {"id": cat_label, "turn": item_estimates[cat_label].get("orientation", "default")}
        for cat_label in objet_reconnus
        if item_estimates.get(cat_label, {}).get("orientation", "default") != "default"
           and item_estimates.get(cat_label, {}).get("orientation") in VALID_ORIENTATIONS
    ]
    print(f"[scene_from_image] etape 3/5 - orientations ({len(orientations)} non-default)...")
    items = processOrientations(items, orientations)

    print("[scene_from_image] etape 4/5 - initialisation des positions et relations verticales...")
    items = initPosAndQuat(items)
    items = processRelations(items, vertical_relations)

    print(f"[scene_from_image] etape 5/5 - conversion coords image -> monde (vue={view_type})...")
    for item in items:
        pos_norm         = item_estimates.get(item["id"], {}).get("pos_norm", [0.5, 0.5])
        world_x, world_y = pos_norm_to_world(pos_norm, view_type)
        _, _, z          = item["pos"]
        item["pos"]      = [world_x, world_y, z]
        print(f"  {item['id']}: pos_norm={pos_norm} -> monde=({world_x:.3f}, {world_y:.3f}, {z:.3f})")

    nudge_depth_from_relations(items, depth_relations, view_type)

    print(f"[scene_from_image] === pipeline termine: {len(items)} objet(s) place(s) ===\n")
    return items
