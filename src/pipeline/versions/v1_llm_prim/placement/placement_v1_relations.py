from pipeline.utils.ollama_client import validate_json_response
from pipeline.utils.helpers import normalize, fuzzy_match
from pipeline.sceneBuilding import buildScene
from .placement_v1_distances_orientations import orientation

#---------------------------------------------------------------------
# RECONNAITRE LES RELATIONS ENTRE OBJETS
#---------------------------------------------------------------------

VALID_RELATION_TYPES = {"on", "under", "left_of", "right_of", "in_front_of", "behind", "against", "inside"}
RELATION_TYPES_WITH_DISTANCE = {"left_of", "right_of", "in_front_of", "behind"}


def fix_ids_in_result(result, valid_ids):
  """Corrige les IDs des relations pour qu'ils correspondent aux ID"""
  # corriger les relations
  fixed_relations = []
  for rel in result.get("relations", []):
    rel_type = rel.get("type", "")
    if rel_type not in VALID_RELATION_TYPES:
      print(f"[object_relations] Relation ignoree (type invalide '{rel_type}') : {rel}")
      continue

    subj = rel.get("subject", "")
    obj = rel.get("object", "")
    matched_subj = fuzzy_match(subj, valid_ids)
    matched_obj = fuzzy_match(obj, valid_ids)

    if not matched_subj or not matched_obj:
      print(f"[object_relations] Relation ignoree (ID non resolu) : {rel}")
      continue

    if matched_subj != subj:
      print(f"[object_relations] subject '{subj}' -> corrige en '{matched_subj}'")
    if matched_obj != obj:
      print(f"[object_relations] object '{obj}' -> corrige en '{matched_obj}'")

    rel["subject"] = matched_subj
    rel["object"] = matched_obj

    # validation de la distance (metres). Conservee uniquement sur les relations qui la supportent
    # et si c'est un nombre > 0.
    if "distance" in rel:
      if rel_type not in RELATION_TYPES_WITH_DISTANCE:
        print(f"[object_relations] distance ignoree pour relation '{rel_type}' : {rel}")
        rel.pop("distance", None)
      else:
        dist = rel.get("distance")
        try:
          dist_val = float(dist)
        except (TypeError, ValueError):
          print(f"[object_relations] distance non numerique '{dist}', retiree : {rel}")
          rel.pop("distance", None)
        else:
          if dist_val <= 0:
            print(f"[object_relations] distance <= 0 ({dist_val}), retiree : {rel}")
            rel.pop("distance", None)
          else:
            rel["distance"] = dist_val

    fixed_relations.append(rel)

  result["relations"] = fixed_relations
  return result


def scale(prompt,objets_rec):
  """genere les scales des objets relativement"""
  objects_info = "\n".join(
    f'- id: "{id}"| dimensions: {info["dimensions"]} m'
    for id, info in objets_rec.items()
  )

  print(["Calcule des scales"])

  system_prompt = f"""You are a strict JSON API. You compute realistic scale factors for 3D objects in a simulation scene.

  OBJECTS TO SCALE (current URDF dimensions in meters — these are the raw model dimensions, NOT real-world sizes):
  {objects_info}

  WHAT SCALE MEANS:
  - scale = 1.0  -> keep the model at its current URDF dimensions
  - scale = 0.5  -> make the object half the size of its URDF dimensions
  - scale = 2.0  -> make the object twice the size of its URDF dimensions
  Your goal: choose scale so that the resulting size (dimensions * scale) matches the REAL-WORLD size of the object.

  REAL-WORLD SIZE REFERENCES (approximate, use these to reason):
  - washing machine / lave-linge : ~0.60 x 0.60 x 0.85 m
  - fridge / réfrigérateur        : ~0.70 x 0.70 x 1.80 m
  - dishwasher / lave-vaisselle   : ~0.60 x 0.60 x 0.85 m
  - oven / four                   : ~0.60 x 0.60 x 0.60 m
  - toaster                       : ~0.30 x 0.20 x 0.20 m
  - trash bin / poubelle          : ~0.35 x 0.35 x 0.55 m
  - laptop / ordinateur portable  : ~0.35 x 0.25 x 0.02 m
  - USB key / clé usb             : ~0.07 x 0.02 x 0.01 m
  - drawer / tiroir               : ~0.50 x 0.40 x 0.15 m
  - door / porte                  : ~0.90 x 0.05 x 2.10 m
  - tap / robinet                 : ~0.10 x 0.10 x 0.20 m
  - chest of drawers / meuble     : ~0.80 x 0.45 x 1.00 m
  - mug / tasse / cup             : ~0.09 x 0.09 x 0.10 m
  - banana / banane               : ~0.04 x 0.18 x 0.04 m
  - apple / pomme                 : ~0.08 x 0.08 x 0.08 m
  - bottle / bouteille            : ~0.08 x 0.08 x 0.25 m
  - bowl / bol                    : ~0.16 x 0.16 x 0.08 m
  - scissors / ciseaux            : ~0.02 x 0.10 x 0.20 m
  - knife / couteau               : ~0.02 x 0.03 x 0.22 m
  - plate / assiette              : ~0.26 x 0.26 x 0.03 m
  - chair / chaise                : ~0.50 x 0.50 x 0.90 m
  - table                         : ~1.20 x 0.80 x 0.75 m

  HOW TO COMPUTE SCALE:
  For each object, find its dominant dimension (longest axis) in the URDF, then:
    scale = real_world_dominant_size / urdf_dominant_size
  Round to 2 decimal places. Keep scale between 0.001 and 5.0.

  ADJUSTMENTS FROM USER PROMPT:
  Read the user's scene description. If the user mentions a size qualifier for an object, adjust its scale:
  - "petit/petite/small/tiny"   -> multiply the computed scale by 0.6
  - "grand/grande/big/large"    -> multiply the computed scale by 1.5
  - "immense/enormous/giant"    -> multiply the computed scale by 2.0
  - "minuscule/tiny/miniature"  -> multiply the computed scale by 0.3
  Apply the qualifier only to the object it is explicitly attached to. Do NOT apply it to other objects.

  CONSISTENCY CHECK — before outputting, verify:
  - Scales are mutually coherent: a USB key must be much smaller than a washing machine
  - No two objects of different real-world sizes have the same scale unless their URDF dimensions compensate

  OUTPUT FORMAT — return ONLY this JSON:
  {{
    "scales": {{
      "<id>": <float>,
      "<id>": <float>
    }}
  }}

  EXAMPLES:
  URDF dims: lave-linge=[1.10, 1.53, 1.00], poubelle=[1.15, 1.73, 1.15]
  User: "un lave linge avec une poubelle a cote"
  → lave-linge real ~0.85m tall, urdf 1.00m -> scale ≈ 0.85. poubelle real ~0.55m, urdf 1.15m dominant → scale ≈ 0.48
  Output: {{"scales": {{"lave-linge": 0.85, "poubelle": 0.48}}}}

  URDF dims: lave-linge=[1.10, 1.53, 1.00], poubelle=[1.15, 1.73, 1.15]
  User: "un lave linge avec une petite poubelle a cote"
  → same as above, but "petite" on poubelle -> 0.48 * 0.6 = 0.29
  Output: {{"scales": {{"lave-linge": 0.85, "poubelle": 0.29}}}}

  Output raw JSON only, no markdown, no comments, no explanation."""

  user_prompt = prompt
  payload = {
      "model": "phi3:mini",
      "system": system_prompt,
      "prompt": user_prompt,
      "stream": False,
      "format": "json",
      "options": {"temperature": 0}
    }

  scales = validate_json_response(payload)

  print(scales)
  return scales


def object_relations(prompt=None, objets_rec=None):
  """
  param:
    prompt : prompt utilisateur
    objets_rec : dict des objets reconnus {id: {urdf, path, dimensions}}

  retourne : dict {"root": "<id>", "relations": [{"type": ..., "subject": ..., "object": ...}]}
  """

  ids = list(objets_rec.keys())
  objects_list = "\n".join(f'- "{id}"' for id in ids)
  print(objects_list)

  if not prompt or not objets_rec:
    return {"root": ids[0] if ids else "", "relations": []}

  system_prompt = f"""You are a strict JSON API that extracts spatial relations between objects from a scene description.

        OBJECTS IN THE SCENE — use ONLY these exact ids (copy them exactly, character by character):
        {objects_list}

        RELATION TYPES (the only allowed values for "type"):
        - "on"          : A rests on top of B              (e.g. "a book on the table")
        - "under"       : A is below B                     (e.g. "a rug under the sofa")
        - "left_of"     : A is to the left of B            (e.g. "the lamp left of the desk")
        - "right_of"    : A is to the right of B           (e.g. "the chair right of the bed")
        - "in_front_of" : A is in front of B               (e.g. "a chair in front of a desk")
        - "behind"      : A is behind B                    (e.g. "a plant behind the sofa")
        - "inside"      : A is inside B                    (e.g. "a book inside the drawer")
        - "against"     : A is against B                   (e.g. "a lamp against the wall")

        OPTIONAL DISTANCE FIELD:
        For the relation types "left_of", "right_of", "in_front_of", "behind" ONLY,
        if the prompt explicitly mentions a distance between the two objects
        (e.g. "a 3m a gauche", "at 50 cm in front", "2 metres behind"),
        add a field "distance" with the value in METERS as a float.
        - Convert cm -> m (50 cm -> 0.5), mm -> m (200 mm -> 0.2).
        - Do NOT invent a distance: if the prompt says nothing about distance, omit the field.
        - Never add "distance" on "on", "under", "inside", "against".

        OUTPUT FORMAT — return ONLY this JSON:
        {{
          "relations": [
            {{"type": "<relation>", "subject": "<id_A>", "object": "<id_B>"}},
            {{"type": "<relation>", "subject": "<id_A>", "object": "<id_B>", "distance": <float_meters>}}
          ]
        }}

        EXAMPLES:
        Scene: "a washing machine with a bin to its right"
        Objects: ["lave-linge", "poubelle"]
        Output: {{"relations": [{{"type": "right_of", "subject": "poubelle", "object": "lave-linge"}}]}}

        Scene: "je veux une banane a cote d'un mug"
        Objects: ["banane", "mug"]
        Output: {{"relations": [{{"type": "right_of", "subject": "mug", "object": "banane"}}]}}

        Scene: "a desk with a computer on it, a chair in front"
        Objects: ["bureau", "ordinateur", "chaise"]
        Output: {{"relations": [{{"type": "on", "subject": "ordinateur", "object": "bureau"}}, {{"type": "in_front_of", "subject": "chaise", "object": "bureau"}}]}}

        Scene: "je veux une banane a 3m a gauche d un mug"
        Objects: ["banane", "mug"]
        Output: {{"relations": [{{"type": "left_of", "subject": "banane", "object": "mug", "distance": 3.0}}]}}

        Scene: "un mug a 50 cm devant une banane"
        Objects: ["mug", "banane"]
        Output: {{"relations": [{{"type": "in_front_of", "subject": "mug", "object": "banane", "distance": 0.5}}]}}

        STRICT RULES:
        - "subject" and "object" must be exact ids from the list above — no other values allowed
        - "type" must be exactly one of: on, under, left_of, right_of, in_front_of, behind, inside, against
        - only extract relations explicitly stated or clearly implied in the scene description
        - do NOT invent relations that are not mentioned
        - if only one object is in the scene, return {{"relations": []}}
        - output raw JSON only, no markdown, no comments, no explanation"""

  valid_ids = set(ids)
  user_prompt = prompt

  for attempt in range(3):
    payload = {
      "model": "llama3.1",
      "system": system_prompt,
      "prompt": user_prompt,
      "stream": False,
      "format": "json",
      "options": {"temperature": 0}
    }

    result = validate_json_response(payload)
    result = fix_ids_in_result(result, valid_ids)
    # verifier que toutes les relations sont valides apres correction
    all_ids_ok = all(
      rel["subject"] in valid_ids and rel["object"] in valid_ids
      for rel in result.get("relations", [])
    )
    if all_ids_ok:
      break

    print(f"[object_relations] retry {attempt + 1} — IDs encore invalides")
    user_prompt = prompt + f"\nIMPORTANT: Use ONLY these exact ids: {ids}. Do NOT modify, translate, or change them in any way."

  print(f"[object_relations] Resultat final : {result}")
  return result


def final_json(objets, scales):
  dict_obj = objets
  dict_scales = scales['scales']
  valid_ids = list(dict_obj.keys())

  for key, value in dict_scales.items():
    matched = fuzzy_match(key, valid_ids)
    if not matched:
      print(f"[final_json] scale ignore, ID inconnu: '{key}'")
      continue
    if not isinstance(value, (int, float)):
      print(f"[final_json] scale invalide ({value}) pour '{matched}', fallback 1.0")
      value = 1.0
    dims = dict_obj[matched].get("dimensions") or []
    if any(abs(value - d) < 1e-4 for d in dims):
      print(f"[final_json] scale suspect ({value}) pour '{matched}' (= une dimension), fallback 1.0")
      value = 1.0
    dict_obj[matched]["scale"] = value

  return dict_obj


def modify_scene(prompt, current_scene_json, objet_reconnus):
  """V1/V1.1 : meme interface que pipeline_v2_llm.modify_scene.
  current_scene_json est ignore — V1 re-run le placement complet depuis le prompt."""
  relations_data = object_relations(prompt, objet_reconnus)
  relations = relations_data.get("relations", [])

  scales_result = scale(prompt, objet_reconnus)
  updated_objets = final_json(objet_reconnus, scales_result)

  items = []
  for label, info in updated_objets.items():
    items.append({
      "id":         label,
      "urdf":       info["urdf"],
      "path":       info.get("path", ""),
      "dimensions": info.get("dimensions"),
      "scale":      info.get("scale", 1.0),
    })

  orientations_data = orientation(prompt, updated_objets)
  items = buildScene(items, relations, orientations_data)

  for item in items:
    pos = item.get("pos", (0, 0, 0))
    item["pos"] = list(pos)
    quat = item.get("quat", (0, 0, 0, 1))
    x, y, z, w = quat
    item["quat"] = [w, x, y, z]

  return items


# objets_test  = {'poubelle': ..., 'refrigerateur': ..., 'machine a laver': ..., 'toaster': ...}
# user_prompt = "je veux un poubelle sur un lave linge et derrier un refrigeratuer et une petite machine a laver.  "
# scales = {'scales': {'poubelle': 0.35, 'refrigerateur': 1.27, 'machine a laver': 0.68}}
# print(final_json(objets_test, scales))

# user_prompt = "je veux un poubelle sur un lave linge "
# objets_rec, objets_non_rec = object_rec(user_prompt)
# resultat = object_relations(user_prompt, objets_rec)
# print(resultat)
