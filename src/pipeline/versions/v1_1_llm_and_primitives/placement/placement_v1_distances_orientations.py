from pipeline.utils.ollama_client import call_llm
from pipeline.utils.helpers import fuzzy_match


#---------------------------------------------
#           ORIENTATIONS
#---------------------------------------------

VALIDE_ORIENTATIONS = {
    'tip_left',
    'tip_right',
    'upside_down',
    'tip_forward',
    'tip_backward',
    'turn_right',
    'turn_left',
    'turn_around',
}


def orientation(prompt, objets_rec):
    """Defini les orientations des objets a partir du prompt utilisateur.

    :param prompt: prompt utilisateur
    :param objets_rec: dict des objets reconnus {id: {...}}
    :return: liste d'orientations au format attendu par processOrientations :
             [{"id": "<id>", "turn": "<turn>"}]
             Si un objet n'est pas mentionne avec une orientation explicite,
             il n'apparait pas dans la liste et restera debout (quat par defaut).
    """
    if not prompt or not objets_rec:
        return []

    ids = list(objets_rec.keys())
    objects_list = "\n".join(f'- "{id}"' for id in ids)

    system_prompt = f"""You are a strict JSON API that extracts EXPLICIT orientation changes for 3D objects from a scene description.

OBJECTS IN THE SCENE -- use ONLY these exact ids (copy them exactly):
{objects_list}

VALID TURN VALUES (the only allowed values for "turn"):
- "tip_left"      : object tipped onto its left side
- "tip_right"     : object tipped onto its right side
- "tip_forward"   : object tipped forward (lying on its front / face down)
- "tip_backward"  : object tipped backward (lying on its back)
- "upside_down"   : object flipped upside down
- "turn_left"     : object rotated 90 degrees to the left (yaw)
- "turn_right"    : object rotated 90 degrees to the right (yaw)
- "turn_around"   : object rotated 180 degrees (facing opposite)

DEFAULT BEHAVIOR:
By default all objects are standing upright. Do NOT output an orientation for objects that are not explicitly reoriented in the prompt.

TRIGGER WORDS (non exhaustive):
- "couche", "couchee", "allonge", "sur le cote"        -> tip_left or tip_right or tip_forward
- "sur le dos", "renverse"                              -> tip_backward
- "face contre terre", "sur le ventre"                  -> tip_forward
- "a l envers", "retourne", "tete en bas", "upside down" -> upside_down
- "tourne a gauche", "pivote a gauche"                  -> turn_left
- "tourne a droite", "pivote a droite"                  -> turn_right
- "retourne vers l arriere", "fait demi-tour", "dos a"  -> turn_around

OUTPUT FORMAT -- return ONLY this JSON:
{{
  "orientations": [
    {{"id": "<id>", "turn": "<turn>"}}
  ]
}}

STRICT RULES:
- "id" must be exactly one id from the list above (copy-paste exactly)
- "turn" must be exactly one of: tip_left, tip_right, tip_forward, tip_backward, upside_down, turn_left, turn_right, turn_around
- Include an entry ONLY if the prompt explicitly states a non-default orientation for that object
- If no object has an explicit orientation change, return {{"orientations": []}}
- Output raw JSON only, no markdown, no comments, no explanation

EXAMPLES:
Scene: "un mug sur une table"
Objects: ["mug", "table"]
Output: {{"orientations": []}}

Scene: "une banane couchee a cote d un mug"
Objects: ["banane", "mug"]
Output: {{"orientations": [{{"id": "banane", "turn": "tip_forward"}}]}}

Scene: "un mug a l envers sur une table"
Objects: ["mug", "table"]
Output: {{"orientations": [{{"id": "mug", "turn": "upside_down"}}]}}
"""

    try:
        response = call_llm(system_prompt, prompt)
    except (ValueError, RuntimeError, ConnectionError) as e:
        print(f"[orientation] echec LLM ({e}), aucune orientation extraite")
        return []

    raw_orientations = response.get("orientations", [])
    if not isinstance(raw_orientations, list):
        print(f"[orientation] format inattendu : {raw_orientations}")
        return []

    valid_ids = set(ids)
    fixed = []
    for ori in raw_orientations:
        if not isinstance(ori, dict):
            continue
        raw_id = ori.get("id", "")
        turn = ori.get("turn", "")

        matched_id = fuzzy_match(raw_id, valid_ids)
        if not matched_id:
            print(f"[orientation] id non resolu, ignore : {ori}")
            continue
        if turn not in VALIDE_ORIENTATIONS:
            print(f"[orientation] turn invalide '{turn}', ignore : {ori}")
            continue

        if matched_id != raw_id:
            print(f"[orientation] id '{raw_id}' -> corrige en '{matched_id}'")
        fixed.append({"id": matched_id, "turn": turn})

    print(f"[orientation] resultat final : {fixed}")
    return fixed
