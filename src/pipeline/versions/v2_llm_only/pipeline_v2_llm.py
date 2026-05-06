import json
from pipeline.utils.catalogue import objects_desc
from pipeline.utils.ollama_client import call_llm
from pipeline.itemSpec import loadScale


#--------------------------
# prompt textuel
#--------------------------

# DIMENSION ET QUATERNIONS
def object_dim_quat(prompt, objet_reconnus):
    """Retourne la liste des objets avec les dimensions et quaternions."""
    for info in objet_reconnus.values():
        loadScale(info)

    catalogue_desc = objects_desc()
    fixed_scales = "\n".join(
        f"  - {lbl}: {info.get('scale', 1.0)}"
        for lbl, info in objet_reconnus.items()
    )

    system_prompt = f"""You are a strict JSON API. You place 3D objects in a simulation scene.
  All dimensions are in meters. The floor is at z = 0.

  OBJECTS TO PLACE:
  {catalogue_desc}

  FIXED SCALES (use these exact values, do not change):
{fixed_scales}

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
  - scale: use the FIXED SCALE value provided above (do not modify)
  - quat: orientation as [w, x, y, z]
    * upright (default): [1.0, 0.0, 0.0, 0.0]
    * lying along X axis (on its side): [0.707, 0.0, 0.707, 0.0]
    * lying along Y axis (on its side): [0.707, 0.707, 0.0, 0.0]
    * For elongated objects (dimensions[2] >> dimensions[0]), if the scene description or context implies it is lying flat on a surface (e.g. a screwdriver, knife, pen, ruler), use a lying quat. Choose the axis that best matches the described orientation.
  - For an upright object: effective_height = dimensions[2] * scale
  - For a lying object: effective_height = dimensions[0] * scale  (the smallest cross-section becomes the height)
  - pos z for objects resting on the FLOOR: effective_height / 2
  - pos z for objects placed ON another object: parent_pos_z + parent_dimensions[2] * parent_scale / 2 + effective_height / 2
  - IMPORTANT: always compute parent object position first, then compute child position on top of it
  - objects must NOT overlap — space them according to the scene description
  - output raw JSON only, no markdown, no comments, no explanation"""

    objetsList = call_llm(system_prompt, prompt).get("objects", [])

    for obj in objetsList:
        if obj["id"] in objet_reconnus:
            obj["path"] = objet_reconnus[obj["id"]]["path"]

    return objetsList


#------- modifier la scene
def modify_scene(prompt, current_objects_json, objet_reconnus):
    """Modifie une scene existante selon le prompt utilisateur."""
    for info in objet_reconnus.values():
        loadScale(info)

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
        f'- id: "{id_}" | urdf: "{info["urdf"]}" | dimensions: {info["dimensions"]} m | scale: {info.get("scale", 1.0)}'
        for id_, info in objet_reconnus.items()
    )

    system_prompt = f"""You are a strict JSON API that modifies an existing 3D scene.
        All dimensions are in meters. The floor is at z = 0.

        CURRENT SCENE (JSON):
        {current_objects_json}

        KNOWN OBJECTS (with fixed scales):
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
        - For "add": add a new object with appropriate pos (no overlap) and use its fixed scale
        - scale: use the fixed scale from KNOWN OBJECTS (do not change it)
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

    objetsList = call_llm(system_prompt, prompt).get("objects", [])

    for obj in objetsList:
        info = objet_reconnus.get(obj.get("id"), {})
        if not obj.get("path") and info.get("path"):
            obj["path"] = info["path"]
        if not obj.get("dimensions") and info.get("dimensions"):
            obj["dimensions"] = info["dimensions"]
        obj.pop("root", None)

    return objetsList
