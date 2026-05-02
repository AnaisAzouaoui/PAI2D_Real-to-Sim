import json
import re
import threading
import time
import unicodedata
import os
#import anthropic
from object_pool import STYLE_EXAMPLES, RELATION_TYPES_WITH_DISTANCE, OBJECT_NAMES_FR
from dotenv import load_dotenv
load_dotenv()

# rate limiter : max 45 appels/min (limite Anthropic = 50)
rate_lock  = threading.Lock()
call_times = []
MAX_RPM    = 45

def _rate_limit():
    with rate_lock:
        now = time.time()
        call_times[:] = [t for t in call_times if now - t < 60]
        if len(call_times) >= MAX_RPM:
            wait = 60 - (now - call_times[0]) + 0.5
            time.sleep(max(0, wait))
        call_times.append(time.time())

# MODEL = "claude-haiku-4-5-20251001"
# client = anthropic.Anthropic()

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-mini"

COREF_MARKERS    = ["cette", "ce ", "cet ", "la meme", "le meme", "celui", "celle", "this ", "that ", "the same"]
NEGATION_MARKERS = ["sans ", "pas de", "aucun", "aucune", "without ", "no ", "n'y a pas", "ni "]
MULTI_MARKERS    = ["autre", "deuxieme", "deuxième", "second", "2eme", "2ème", "another", "2nd", "supplementaire", "supplémentaire"]

# detecte une distance dans la phrase 
DISTANCE_REGEX = re.compile(
    r"\b(\d+(?:[\.,]\d+)?)\s*(meters?|m(?:[èe]tres?)?|cm|mm)\b",
    re.IGNORECASE,
)

# format francais usuel "1m5" = 1.5 metres un petit peu chiant tout ca 
DISTANCE_REGEX_FR = re.compile(r"\b(\d+)m(\d{1,2})\b", re.IGNORECASE)

# variantes
ORIENTATION_MARKERS = {
    "tip_left":     ["sur le cote gauche", "couche a gauche", "couchee a gauche", "incline a gauche", "inclinee a gauche", "incline vers la gauche", "inclinee vers la gauche", "penche a gauche", "penchee a gauche", "penche vers la gauche", "penchee vers la gauche", "tipped left", "tilted left", "leaning left"],
    "tip_right":    ["sur le cote droit", "couche a droite", "couchee a droite", "incline a droite", "inclinee a droite", "incline vers la droite", "inclinee vers la droite", "penche a droite", "penchee a droite", "penche vers la droite", "penchee vers la droite", "tipped right", "tilted right", "leaning right"],
    "tip_forward":  ["couche", "couchee", "allonge", "allongee", "sur le ventre", "face contre terre", "penche en avant", "penchee en avant", "incline en avant", "inclinee en avant", "incline vers l avant", "inclinee vers l avant", "penche vers l avant", "penchee vers l avant", "bascule vers l avant", "basculee vers l avant", "tipped forward", "tilted forward", "leaning forward", "lying", "fallen forward"],
    "tip_backward": ["sur le dos", "renverse", "renversee", "penche en arriere", "penchee en arriere", "incline en arriere", "inclinee en arriere", "incline vers l arriere", "inclinee vers l arriere", "penche vers l arriere", "penchee vers l arriere", "bascule vers l arriere", "basculee vers l arriere", "tipped backward", "tilted backward", "leaning backward", "fallen backward"],
    "upside_down":  ["a l envers", "tete en bas", "retourne", "retournee", "upside down", "inverted", "flipped"],
    "turn_left":    ["tourne a gauche", "tournee a gauche", "pivote a gauche", "pivotee a gauche", "turned left", "rotated left", "facing left"],
    "turn_right":   ["tourne a droite", "tournee a droite", "pivote a droite", "pivotee a droite", "turned right", "rotated right", "facing right"],
    "turn_around":  ["fait demi-tour", "demi tour", "dos a", "tourne dans l autre sens", "tourne a 180", "180 degres", "de dos", "turned around", "facing away", "turned 180", "facing the other way", "facing opposite"],
}

# relations ou le sujet doit etre plus petit ou egal a l'objet (sens physique)
ON_VALID_PAIRS = {("petit", "grand"), ("petit", "moyen"), ("moyen", "grand"),
                  ("petit", "petit"), ("moyen", "moyen"), ("grand", "grand")}

CONFIG_INSTRUCTIONS = {
    "basic":            "Ecris une phrase simple et naturelle.",
    "with_distance":    "Mentionne explicitement chaque distance avec son unite (ex: 'a 50 cm', 'a 2 metres', 'a 1m5'). La valeur dans la phrase doit correspondre exactement a la distance dans la spec (en metres).",
    "with_orientation": "Decris explicitement la rotation/orientation de chaque objet concerne (ex: 'a l envers', 'couchee', 'tournee vers la gauche', 'sur le dos'). Utilise un terme naturel coherent avec le 'turn' donne.",
    "dense":            "Scene riche : exprime explicitement TOUTES les distances avec leur unite ET TOUTES les orientations donnees. Phrase fluide en plusieurs propositions.",
    "coreference":      "Un objet apparait dans plusieurs relations. Utilise obligatoirement 'cette X' ou 'ce X' pour le referencer la deuxieme fois (jamais un nouveau X).",
    "multi_instance":   "Il y a deux instances du meme type (ex: banane et banane_2). Introduis le premier normalement, le second avec 'une autre X' ou 'un deuxieme X'.",
    "no_relation":      "Liste les objets sans exprimer de relation spatiale.",
    "longue":           "Ecris une description en plusieurs propositions enchainees.",
    "negation":         "Mentionne l'objet absent obligatoirement avec 'sans X', 'pas de X' ou 'aucun X'. Ne l'inclus PAS dans les objets du JSON.",
    "anglais":          "Write the sentence in English. Use 'another X' or 'a second X' for duplicate objects. Express distances in meters/cm naturally.",
}

SYSTEM = """Tu es un generateur de donnees d'entrainement pour un modele NLP de scene 3D.
Tu recois une spec (objets, root, relations, orientations) et tu dois ecrire UNE SEULE phrase naturelle qui decrit exactement cette scene.

Reponds UNIQUEMENT avec la phrase, sans guillemets, sans JSON, sans explication.

STYLE : la phrase doit sonner naturelle, comme ce qu'un humain taperait pour decrire une scene.
- Prefere les formes nominales et conversationnelles : "une chaise sur le canape", "je veux un frigo a cote d'un lave-linge", "place le livre sur la table"
- Evite les formes robotiques du type "La chaise est sur le canape." ou "Le frigo est positionne a gauche de..."
- Varie les formulations : parfois sans sujet ("un bureau avec une lampe dessus"), parfois imperatif ("mets une plante devant la fenetre"), parfois "je veux/voudrais..."

OBLIGATOIRE : tous les objets de la spec doivent apparaitre dans la phrase. N'en oublie aucun.

ORIENTATIONS — signification exacte de chaque valeur (ne pas confondre) :
- tip_left / tip_right : l objet est incline sur le cote (ex: "penche vers la gauche", "tilted right")
- tip_forward / tip_backward : l objet est incline vers l avant ou l arriere (ex: "penche en avant", "tilted backward")
- upside_down : l objet est retourne completement a l envers, tete en bas (ex: "a l envers", "upside down") — PAS turn_around
- turn_left / turn_right : l objet pivote sur lui-meme (ex: "tourne a gauche", "turned right")
- turn_around : l objet fait face dans la DIRECTION OPPOSEE, il a le dos vers nous (ex: "de dos", "tourne dans l autre sens", "facing away") — PAS a l envers"""


def normalize(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().replace("'", " ").replace("'", " ").replace("-", " ").replace("_", " ")


def display_name(obj_id):
    """Retourne le nom d'affichage francais d'un objet (sans suffixe _2, _3...)."""
    base = obj_id.rsplit("_", 1)[0] if obj_id[-1].isdigit() else obj_id
    return OBJECT_NAMES_FR.get(base, base)


def object_in_phrase(obj_id, phrase, config=""):
    name = display_name(obj_id)
    # pour config anglais on garde l'ID tel quel
    if config == "anglais":
        name = obj_id.rsplit("_", 1)[0] if obj_id[-1].isdigit() else obj_id
    name_norm   = normalize(name)
    phrase_norm = normalize(phrase)
    if name_norm in phrase_norm:
        return True
    # pour les noms composes (ex: lave vaisselle), verifier chaque mot
    words = name_norm.split()
    return len(words) > 1 and all(w in phrase_norm for w in words)


def parse_response(text):
    phrase = text.strip().strip('"').strip("'")
    return phrase if phrase else None


def phrase_distances_meters(phrase):
    """Extrait toutes les distances de la phrase, converties en metres."""
    found = []
    # consommer d'abord les "1m5" pour eviter qu'ils soient capturees comme "1m" par DISTANCE_REGEX
    consumed_spans = []
    for match in DISTANCE_REGEX_FR.finditer(phrase):
        meters   = int(match.group(1))
        dec_str  = match.group(2)
        decimal  = int(dec_str) / (10 ** len(dec_str))  # "5"->0.5, "05"->0.05
        found.append(round(meters + decimal, 3))
        consumed_spans.append(match.span())

    for match in DISTANCE_REGEX.finditer(phrase):
        if any(start <= match.start() < end for start, end in consumed_spans):
            continue
        raw_val = match.group(1).replace(",", ".")
        try:
            val = float(raw_val)
        except ValueError:
            continue
        unit = match.group(2).lower()
        if unit.startswith("cm"):
            val /= 100.0
        elif unit == "mm":
            val /= 1000.0
        # sinon : metres (m, metre, metres)
        found.append(round(val, 3))
    return found


def orientation_marker_in_phrase(turn, phrase):
    phrase_norm = normalize(phrase)
    for marker in ORIENTATION_MARKERS.get(turn, []):
        if normalize(marker) in phrase_norm:
            return True
    return False


# def _specs_match(spec, extracted, phrase):
#     objets       = extracted.get("objets", [])
#     root         = extracted.get("root")
#     relations    = extracted.get("relations", [])
#     orientations = extracted.get("orientations", [])
#     if set(spec["objets"]) != set(objets):
#         return False, "objets_mismatch"
#     if spec["root"] != root or root not in objets:
#         return False, "root_invalid"
#     def rel_key(r):
#         return (r["type"], r["subject"], r["object"])
#     if set(rel_key(r) for r in spec["relations"]) != set(rel_key(r) for r in relations):
#         return False, "relations_mismatch"
#     for r in relations:
#         if r["subject"] not in objets or r["object"] not in objets:
#             return False, "relation_ref_missing"
#     if spec["config"] != "no_relation" and relations:
#         covered = {id_ for r in relations for id_ in (r["subject"], r["object"])}
#         if not all(obj in covered for obj in objets):
#             return False, "object_uncovered"
#     for obj_id in spec["objets"]:
#         if not _object_in_phrase(obj_id, phrase):
#             return False, f"object_not_in_phrase:{obj_id}"
#     spec_relations = {rel_key(r): r for r in spec["relations"]}
#     extracted_relations = {rel_key(r): r for r in relations}
#     for key, spec_rel in spec_relations.items():
#         spec_dist = spec_rel.get("distance")
#         ext_dist  = extracted_relations[key].get("distance")
#         if spec_dist is not None:
#             if ext_dist is None:
#                 return False, "distance_missing_in_json"
#             if abs(float(ext_dist) - float(spec_dist)) > 0.05:
#                 return False, "distance_value_mismatch"
#             phrase_dists = _phrase_distances_meters(phrase)
#             if not any(abs(d - float(spec_dist)) < 0.05 for d in phrase_dists):
#                 return False, "distance_not_in_phrase"
#         else:
#             if ext_dist is not None:
#                 return False, "distance_hallucinated"
#     for r in relations:
#         if r.get("distance") is not None and r["type"] not in RELATION_TYPES_WITH_DISTANCE:
#             return False, "distance_on_invalid_type"
#     def orient_key(o):
#         return (o["id"], o["turn"])
#     spec_orients_set = set(orient_key(o) for o in spec.get("orientations", []))
#     ext_orients_set  = set(orient_key(o) for o in orientations)
#     if spec_orients_set != ext_orients_set:
#         return False, "orientations_mismatch"
#     for o in spec.get("orientations", []):
#         if not _orientation_marker_in_phrase(o["turn"], phrase):
#             return False, f"missing_orientation_marker:{o['id']}:{o['turn']}"
#     phrase_norm = _normalize(phrase)
#     if spec["config"] == "coreference":
#         if not any(_normalize(m) in phrase_norm for m in COREF_MARKERS):
#             return False, "missing_coref_marker"
#     if spec["config"] == "negation":
#         if not any(_normalize(m) in phrase_norm for m in NEGATION_MARKERS):
#             return False, "missing_negation_marker"
#     if spec["config"] == "multi_instance":
#         if not any(_normalize(m) in phrase_norm for m in MULTI_MARKERS):
#             return False, "missing_multi_marker"
#     return True, None


OPPOSITES = {"left_of":"right_of","right_of":"left_of","in_front_of":"behind","behind":"in_front_of"}

RELATION_DESC_FR = {
    "on":          "{subj} est sur {obj}",
    "under":       "{subj} est sous {obj}",
    "left_of":     "{subj} est a gauche de {obj}",
    "right_of":    "{subj} est a droite de {obj}",
    "in_front_of": "{subj} est devant {obj}",
    "behind":      "{subj} est derriere {obj}",
    "against":     "{subj} est contre {obj}",
    "inside":      "{subj} est dans {obj}",
}

RELATION_DESC_EN = {
    "on":          "{subj} is on {obj}",
    "under":       "{subj} is under {obj}",
    "left_of":     "{subj} is to the left of {obj}",
    "right_of":    "{subj} is to the right of {obj}",
    "in_front_of": "{subj} is in front of {obj}",
    "behind":      "{subj} is behind {obj}",
    "against":     "{subj} is against {obj}",
    "inside":      "{subj} is inside {obj}",
}


def phrase_valid(spec, phrase):
    phrase_norm = normalize(phrase)
    if "_" in phrase:
        return False, "raw_id_in_phrase"

    # spec sans relations contradictoires (A derriere B et B derriere A)
    rel_keys = set((r["type"], r["subject"], r["object"]) for r in spec["relations"])
    for r in spec["relations"]:
        if (r["type"], r["object"], r["subject"]) in rel_keys:
            return False, "contradictory_relations_in_spec"

    # tous les objets de la spec doivent etre couverts par au moins une relation normalement
    if spec["config"] != "no_relation":
        covered = set()
        for r in spec["relations"]:
            covered.add(r["subject"])
            covered.add(r["object"])
        for obj_id in spec["objets"]:
            if obj_id not in covered:
                return False, f"object_uncovered_in_spec:{obj_id}"

    # tous les objets doivent apparaitre dans la phrase
    for obj_id in spec["objets"]:
        if not object_in_phrase(obj_id, phrase, config=spec["config"]):
            return False, f"object_not_in_phrase:{obj_id}"

    # chaque distance de la spec doit apparaitre dans la phrase
    for rel in spec["relations"]:
        dist = rel.get("distance")
        if dist is not None:
            phrase_dists = phrase_distances_meters(phrase)
            if not any(abs(d - float(dist)) < 0.05 for d in phrase_dists):
                return False, f"distance_not_in_phrase:{dist}"

    # chaque orientation doit avoir un marqueur dans la phrase
    for o in spec.get("orientations", []):
        if not orientation_marker_in_phrase(o["turn"], phrase):
            return False, f"missing_orientation_marker:{o['id']}:{o['turn']}"

    if spec["config"] == "coreference":
        if not any(normalize(m) in phrase_norm for m in COREF_MARKERS):
            return False, "missing_coref_marker"

    if spec["config"] == "negation":
        if not any(normalize(m) in phrase_norm for m in NEGATION_MARKERS):
            return False, "missing_negation_marker"

    if spec["config"] == "multi_instance":
        if not any(normalize(m) in phrase_norm for m in MULTI_MARKERS):
            return False, "missing_multi_marker"

    return True, None


def _build_hint(reason):
    hints = {
        "raw_id_in_phrase":           "N'ecris jamais d'identifiant technique dans la phrase (pas de underscore, pas de '_2'). Utilise uniquement du langage naturel : 'une autre armoire', 'un deuxieme micro-onde'.",
        "object_not_in_phrase":       "Tous les objets de la spec doivent apparaitre dans la phrase. Tu en as oublie un.",
        "distance_not_in_phrase":     "Mentionne explicitement la distance dans la phrase (ex: 'a 1.5 metres', 'a 50 cm').",
        "missing_orientation_marker": "Decris explicitement l'orientation (ex: 'a l envers', 'couche', 'tourne a gauche', 'incline vers l arriere').",
        "missing_coref_marker":       "Utilise 'cette X' ou 'ce X' pour referer au meme objet la deuxieme fois.",
        "missing_negation_marker":    "Mentionne l'objet absent avec 'sans X' ou 'pas de X'.",
        "missing_multi_marker":       "Utilise 'une autre X' ou 'un deuxieme X' pour le second exemplaire.",
    }
    key = reason.split(":")[0] if ":" in reason else reason
    return hints.get(key, "Respecte exactement la spec fournie.")



# def _call_api(prompt, hint=None):
#     _rate_limit()
#     full_prompt = SYSTEM + "\n\n" + prompt
#     if hint:
#         full_prompt += f"\n\nCorrection requise : {hint}\nRegenere completement."
#     schema_format = { "type": "json_schema", "name": "scene", "strict": True,
#         "schema": { "type": "object", "properties": {
#             "phrase": {"type": "string"},
#             "json": { "type": "object", "properties": {
#                 "objets": {"type": "array", "items": {"type": "string"}},
#                 "root": {"type": "string"},
#                 "relations": {"type": "array", "items": {"type": "object",
#                     "properties": {"type": {"type": "string", "enum": ["on","under","left_of","right_of","in_front_of","behind","against","inside"]},
#                         "subject": {"type": "string"}, "object": {"type": "string"},
#                         "distance": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
#                     "required": ["type","subject","object","distance"], "additionalProperties": False}},
#                 "orientations": {"type": "array", "items": {"type": "object",
#                     "properties": {"id": {"type": "string"}, "turn": {"type": "string",
#                         "enum": ["tip_left","tip_right","tip_forward","tip_backward","upside_down","turn_left","turn_right","turn_around"]}},
#                     "required": ["id","turn"], "additionalProperties": False}}},
#             "required": ["objets","root","relations","orientations"], "additionalProperties": False}},
#         "required": ["phrase","json"], "additionalProperties": False}}
#     create_kwargs = {"model": MODEL, "input": full_prompt, "temperature": 0.3}
#     try:
#         import inspect
#         params = inspect.signature(client.responses.create).parameters
#         if "text" in params:
#             create_kwargs["text"] = {"format": schema_format}
#         elif "response_format" in params:
#             create_kwargs["response_format"] = schema_format
#     except Exception:
#         create_kwargs["text"] = {"format": schema_format}
#     response = client.responses.create(**create_kwargs)
#     return response.output_text



def call_api(prompt, hint=None):
    _rate_limit()

    full_prompt = SYSTEM + "\n\n" + prompt
    if hint:
        full_prompt += f"\n\nCorrection requise : {hint}\nRegenere une nouvelle phrase."

    response = client.responses.create(
        model=MODEL,
        input=full_prompt,
        temperature=0.7,
    )
    return response.output_text


def generate_example(spec, max_retries=2):
    style        = "\n".join(f'- "{e}"' for e in STYLE_EXAMPLES[:6])
    config_instr = CONFIG_INSTRUCTIONS.get(spec["config"], "")

    use_fr = spec["config"] != "anglais"

    def obj_label(obj_id):
        if not use_fr:
            return obj_id
        base = obj_id.rsplit("_", 1)[0] if obj_id[-1].isdigit() else obj_id
        fr   = OBJECT_NAMES_FR.get(base, obj_id)
        return fr + ("_2" if obj_id.endswith("_2") else "")

    def rel_desc(r):
        subj = obj_label(r["subject"])
        obj  = obj_label(r["object"])
        tpl  = RELATION_DESC_FR if use_fr else RELATION_DESC_EN
        desc = tpl[r["type"]].format(subj=subj, obj=obj)
        dist = r.get("distance")
        if dist is not None:
            desc += f" (a {dist} m)" if use_fr else f" (at {dist} m)"
        return desc

    objets_label     = [obj_label(o) for o in spec["objets"]]
    root_label       = obj_label(spec["root"])
    relations_label  = [rel_desc(r) for r in spec["relations"]]
    orient_label     = [{**o, "id": obj_label(o["id"])} for o in spec.get("orientations", [])]

    prompt = (
        f"Exemples de style :\n{style}\n\n"
        f"Spec :\n"
        f"  objets       : {objets_label}\n"
        f"  root         : {root_label}\n"
        f"  relations    : {json.dumps(relations_label, ensure_ascii=False)}\n"
        f"  orientations : {json.dumps(orient_label, ensure_ascii=False)}\n\n"
        f"Consigne : {config_instr}"
    )

    reason = None
    phrase = None
    hint   = None
    for _ in range(max_retries + 1):
        text   = call_api(prompt, hint=hint)
        phrase = parse_response(text)
        if not phrase:
            hint = "Reponds uniquement avec la phrase, sans guillemets ni explication."
            continue
        valid, reason = phrase_valid(spec, phrase)
        if valid:
            return {
                "input":  phrase,
                "config": spec["config"],
                "output": {
                    "objets":       spec["objets"],
                    "root":         spec["root"],
                    "relations":    spec["relations"],
                    "orientations": spec.get("orientations", []),
                },
            }
        hint = _build_hint(reason)

    print(f"\n[REJETE] raison : {reason}", flush=True)
    print(f"  spec   : objets={spec['objets']} root={spec['root']} config={spec['config']}", flush=True)
    print(f"  phrase : {phrase}", flush=True)
    return None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from spec_generator import generate_spec

    for config in ["basic", "with_distance", "with_orientation", "dense", "coreference", "multi_instance", "negation", "anglais"]:
        spec = generate_spec(config=config)
        print(f"\n[{config}] spec : {spec['objets']} | root : {spec['root']}")
        print(f"  relations    : {spec['relations']}")
        print(f"  orientations : {spec['orientations']}")
        ex = generate_example(spec)
        if ex:
            print(f"  phrase  : {ex['input']}")
            print(f"  output  : {json.dumps(ex['output'], ensure_ascii=False)}")
        else:
            print("  -> rejete apres retries")
