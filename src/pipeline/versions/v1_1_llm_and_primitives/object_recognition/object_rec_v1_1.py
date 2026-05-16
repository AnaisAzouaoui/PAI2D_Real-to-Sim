from pipeline.utils.catalogue import objets_list, objects_desc
from pipeline.utils.ollama_client import call_llm


#---------------------------------------------------------------------
# RECONNAISSANCE D'OBJETS AVEC PROMPT TEXTUEL
#---------------------------------------------------------------------

def validate_matches(result, objects_data):
  """Verifie par le LLM que chaque mapping label urdf est coherent
  Retourne ([(label, urdf), ...] valides, [labels] rejetes)"""
  reconnus = result.get("obj_reconnus", {})
  matches = [(label, urdf) for label, urdf in reconnus.items() if urdf in objects_data]
  if not matches:
    return [], list(reconnus.keys())

  pairs_desc = "\n".join(
    f'{i+1}. "{label}" => "{objects_data[urdf]["name"].strip()}"'
    for i, (label, urdf) in enumerate(matches)
  )

  system_prompt = """You verify semantic matches between a scene label and a catalogue object name.
        For each numbered pair, answer: does the label refer to the SAME TYPE of real-world object?

        Examples of TRUE matches (synonyms or same object):
        "frigo" => "refrigerateur" = true
        "machine à laver" => "lave linge" = true
        "ordi" => "ordinateur portable" = true
        "poubelle" => "poubelle" = true

        Examples of FALSE matches (different objects):
        "baignoire" => "ordinateur portable" = false
        "chaise" => "poubelle" = false
        "table" => "four" = false
        "lit" => "toaster" = false
        "table" => "meuble tiroirs" = false

        Return ONLY: {"verdicts": [true, false, ...]} — one boolean per pair, in order."""

  raw_verdicts = call_llm(system_prompt, pairs_desc).get("verdicts", [])

  # normaliser : le LLM retourne deds fois un bool au lieu d'une liste
  if isinstance(raw_verdicts, bool):
    verdicts = [raw_verdicts]
  elif isinstance(raw_verdicts, list):
    verdicts = raw_verdicts
  else:
    verdicts = []

  accepted = []
  rejected = []
  for i, (label, urdf) in enumerate(matches):
    if i < len(verdicts) and verdicts[i]:
      accepted.append((label, urdf))
    else:
      print(f"[validation] rejete: '{label}' != '{objects_data[urdf]['name'].strip()}'")
      rejected.append(label)

  return accepted, rejected


def object_rec(prompt=None):
  """
  param:
    prompt : prompt utilisateur

  retourne : dict des objets reconnus {id: {urdf, path, dimensions}}
  """

  objects_data = objets_list()
  valid_urdfs = set(objects_data.keys())
  catalogue_desc = objects_desc()

  print("[OBJETS DU CATALOGUE]",catalogue_desc)

  if not prompt:
    return {}, []

  system_prompt = f"""You are a strict JSON API that matches objects from a scene description to a fixed catalogue of 3D models.

        CATALOGUE — the ONLY valid "urdf" values:
        {catalogue_desc}

        EXAMPLES:
        Input: "un frigo et une machine à laver côte à côte"
        Output: {{"objets_non_reconnus": [], "obj_reconnus": {{"frigo": "refrigerateur", "machine à laver": "lave_linge"}}}}

        Input: "une chaise et un four"
        Output: {{"objets_non_reconnus": ["chaise"], "obj_reconnus": {{"four": "four"}}}}

        Input: "deux toasters derriere une machine a laver"
        Output: {{"objets_non_reconnus": [], "obj_reconnus": {{"toaster_1": "toaster","toaster_2": "toaster","machine à laver": "lave_linge"}}}}

        Input: "une baignoire à côté d'un lave-linge"
        Output: {{"objets_non_reconnus": ["baignoire"], "obj_reconnus": {{"lave-linge": "lave_linge"}}}}
        IMPORTANT: if "baignoire" does NOT exist in the catalogue — it must go in objets_non_reconnus, NOT be mapped to an unrelated object.

        OUTPUT FORMAT — return ONLY this JSON:
        {{
          "objets_non_reconnus": ["<unmatched word>"],
          "obj_reconnus": {{
            "<label from scene>": "<exact urdf from catalogue>",
            "<label from scene>": "<exact urdf from catalogue>"
          }}
        }}

        STRICT RULES:
        - each value in "obj_reconnus" must be EXACTLY one of the catalogue entries, copied verbatim
        - include a label in "obj_reconnus" ONLY if it has a DIRECT, OBVIOUS semantic match (same type of object)
        - NEVER substitute an unavailable object with a different one (e.g. do NOT map "baignoire" to "ordinateur_portable")
        - every scene word with no match goes in "objets_non_reconnus"
        - if no object is recognized: {{"objets_non_reconnus": [...], "obj_reconnus": {{}}}}
        - output raw JSON only, no markdown, no comments, no explanation"""

  user_prompt = prompt
  result = {}

  for attempt in range(3):  # 3 essais pour que le LLM tente de reconnaitre les objets
    result = call_llm(system_prompt, user_prompt)

    invalid = [urdf for urdf in result.get("obj_reconnus", {}).values()
               if urdf not in valid_urdfs]
    if not invalid:
      break
    print(f"[retry {attempt + 1}] urdfs invalides : {invalid}")
    user_prompt = prompt + f"\nPREVIOUS ERROR: these urdfs do not exist: {invalid}. Remove or correct them."

  # Controle semantique : verifier que chaque label correspond vraiment a l'objet mappe
  valid_matches, rejected_labels = validate_matches(result, objects_data)
  non_reconnus = result.get("objets_non_reconnus", []) + rejected_labels

  objet_reconnus = {
    label: {
      "urdf": urdf,
      "path": objects_data[urdf]["path"],
      "dimensions": objects_data[urdf]["dimensions"]
    }
    for label, urdf in valid_matches
  }

  print("Reconnus:", objet_reconnus)
  print("Non reconnus:", non_reconnus)
  return objet_reconnus, non_reconnus


# premiere phase du prompt : reconnaitre les objets
# objectis :
# - soit le LLM reconnait l'exact objet dans la liste des objets
# - sinon il liste les objets non reconnu et propose les alternatives les plus proches
# - il faut renvoyer les objets et leur paths

# deuxieme phase du prompt :
# - une fois les objets reconnu, on recupere leur dimensions
# - on repasse le prompt corrige avec les dimentions definis et les objets bien adequats

# troisieme phase:
# il faut que le user soit capable de relancer le model, ca doit etre comme une discussion
# une fois que objectList est cree et la scene est generee le user peut demander a changer d'objet
# ou a bouger les objets et le LLM doit etre capable de modifier objetList et refaire

# => objectif final, un Json de la forme :
"""
{
  "id": "lave_linge",
  "urdf": "lave_linge",
  "path": "/Users/anaisazouaoui/Desktop/PAI2D/projet Real-to-Sim/PAI2D_Real-to-Sim/src/../objets/lave_linge/mobility.urdf",
  "scale": 0.55,
  "quat": [0.7071,0.0,0.0,0.7071],
  "pos": [0.0,0.0,0.43]
}
{
  "id": "poubelle",
  "urdf": "poubelle",
  "path": "/Users/anaisazouaoui/Desktop/PAI2D/projet Real-to-Sim/PAI2D_Real-to-Sim/src/../objets/poubelle/mobility.urdf",
  "scale": 0.26,
  "quat": [0.7071,0.0,0.0,0.7071],
  "pos": [0.0,0.0,1.07]
}
"""
