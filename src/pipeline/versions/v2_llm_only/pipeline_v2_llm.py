import json
from pipeline.utils.catalogue import objects_desc
from pipeline.utils.ollama_client import validate_json_response

#--------------------------
# prompt textuel
#--------------------------

# DIMENSION ET QUATERNIONS
def object_dim_quat(prompt, objet_reconnus):
  """retourne la liste des objets avec les dimensions et quaternions
  forme :
      {
      "id": "poubelle",
      "urdf": "10357_poubelle",
      "path": "..../src/../objets/10357_poubelle/mobility.urdf",
      "scale": 0.26,
      "quat": [0.7071,0.0,0.0,0.7071],
      "pos": [0.0,0.0,1.07]
      }
  """

  catalogue_desc = objects_desc()

  system_prompt = f"""You are a strict JSON API. You place 3D objects in a simulation scene.
  All dimensions are in meters. The floor is at z = 0.

  OBJECTS TO PLACE:
  {catalogue_desc}

  OUTPUT FORMAT — return ONLY this JSON object:
  {{
    "objects": [
      {{
        "id": "<object id>",
        "urdf": "<urdf name>",
        "scale": <float>,
        "quat": [<w>, <x>, <y>, <z>],
        "pos": [<x>, <y>, <z>]
      }}
    ]
  }}

  STRICT RULES:
  - "id" and "urdf" must be copied verbatim from the objects list above
  - scale: adjust so the object looks realistic in the scene (1.0 = use real dimensions as-is)
  - quat: orientation as [w, x, y, z] — default orientation = [1.0,0.0,0.0,0]
  - pos z for objects resting on the FLOOR: dimensions[2] * scale / 2
  - pos z for objects placed ON another object: parent_pos_z + parent_dimensions[2] * parent_scale / 2 + this_dimensions[2] * this_scale / 2
  - IMPORTANT: always compute parent object position first, then compute child position on top of it
  - objects must NOT overlap — space them according to the scene description
  - output raw JSON only, no markdown, no comments, no explanation"""

  payload = {
    "model": "llama3.1",
    "system": system_prompt,
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {"temperature": 0}
  }

  objetsList = validate_json_response(payload).get("objects", [])

  for obj in objetsList:
    if obj["id"] in objet_reconnus:
      obj["path"] = objet_reconnus[obj["id"]]["path"]

  #print(json.dumps(objetsList, indent=2))
  return objetsList

#-------modifier la scene
def modify_scene(prompt, current_objects_json, objet_reconnus):
  """Modifie une scene existante selon le prompt utilisateur."""
  catalog_desc = objects_desc()

  # convert scene quats from [x,y,z,w] (scipy) to [w,x,y,z] so the LLM sees a consistent format
  try:
    scene_data = json.loads(current_objects_json)
    for obj in scene_data:
      if "quat" in obj and len(obj["quat"]) == 4:
        x, y, z, w = obj["quat"]
        obj["quat"] = [w, x, y, z]
    current_objects_json = json.dumps(scene_data)
  except (json.JSONDecodeError, ValueError):
    pass

  objects_info = "\n".join(
    f'- id: "{id_}" | urdf: "{info["urdf"]}" | dimensions: {info["dimensions"]} m'
    for id_, info in objet_reconnus.items()
  )

  system_prompt = f"""You are a strict JSON API that modifies an existing 3D scene.
        All dimensions are in meters. The floor is at z = 0.

        CURRENT SCENE (JSON):
        {current_objects_json}

        KNOWN OBJECTS:
        {objects_info}

        FULL CATALOGUE (for adding new objects):
        {catalog_desc}

        INSTRUCTIONS:
        - Apply ONLY the change the user asks for
        - Keep all other objects EXACTLY as they are (same id, urdf, scale, quat, pos)
        - For "move left": decrease x. For "move right": increase x.
        - For "move forward": increase y. For "move back": decrease y.
        - For "move up": increase z. For "move down": decrease z.
        - Small movements = ~0.3m, medium = ~0.6m, large = ~1.0m
        - For "remove": delete the object from the list
        - For "replace X with Y": change the urdf and id, keep similar pos/scale
        - For "add": add a new object with appropriate pos (no overlap)
        - pos z for objects on the floor: dimensions[2] * scale / 2
        - pos z for objects ON another: parent_pos_z + parent_dimensions[2] * parent_scale / 2 + this_dimensions[2] * this_scale / 2

        OUTPUT FORMAT — return ONLY:
        {{
          "objects": [
            {{
              "id": "<id>",
              "urdf": "<urdf>",
              "scale": <float>,
              "quat": [<w>, <x>, <y>, <z>],
              "pos": [<x>, <y>, <z>]
            }}
          ]
        }}
        Output raw JSON only, no markdown, no comments."""

  payload = {
    "model": "llama3.1",
    "system": system_prompt,
    "prompt": prompt,
    "stream": False,
    "format": "json",
    "options": {"temperature": 0}
  }

  objetsList = validate_json_response(payload).get("objects", [])

  for obj in objetsList:
    info = objet_reconnus.get(obj.get("id"), {})
    if not obj.get("path") and info.get("path"):
      obj["path"] = info["path"]
    if not obj.get("dimensions") and info.get("dimensions"):
      obj["dimensions"] = info["dimensions"]
    obj.pop("root", None)

  return objetsList


# user_prompt = "je veux un poubelle sur un lave linge "
# objets_rec, objets_non_rec = object_rec(user_prompt)
# resultat = object_dim_quat(user_prompt, objets_rec)
# print(resultat)
