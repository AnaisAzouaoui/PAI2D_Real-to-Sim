import json
import re
import threading
import time
import unicodedata
import os
import random
from object_pool import STYLE_EXAMPLES, RELATION_TYPES_WITH_DISTANCE, OBJECT_NAMES_FR
from dotenv import load_dotenv
load_dotenv()

# rate limiter : max 45 appels/min
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

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-mini"

COREF_MARKERS = ["cette", "ce ", "cet ", "la meme", "le meme", "celui", "celle", "this ", "that ", "the same"]
MULTI_MARKERS = ["autre", "deuxieme", "deuxième", "second", "2eme", "2ème", "another", "2nd", "supplementaire", "supplémentaire"]

DISTANCE_REGEX = re.compile(
    r"\b(\d+(?:[\.,]\d+)?)\s*(meters?|m(?:[èe]tres?)?|cm|mm)\b",
    re.IGNORECASE,
)
DISTANCE_REGEX_FR       = re.compile(r"\b(\d+)m(\d{1,2})\b", re.IGNORECASE)
DISTANCE_REGEX_FR_SPACE = re.compile(r"\b(\d+)\s+m\s+(\d{1,2})\b", re.IGNORECASE)

ORIENTATION_MARKERS = {
    "tip_left":     ["sur le cote gauche", "couche a gauche", "couchee a gauche", "incline a gauche", "inclinee a gauche", "incline vers la gauche", "inclinee vers la gauche", "penche a gauche", "penchee a gauche", "penche vers la gauche", "penchee vers la gauche", "tipped left", "tilted left", "leaning left"],
    "tip_right":    ["sur le cote droit", "couche a droite", "couchee a droite", "incline a droite", "inclinee a droite", "incline vers la droite", "inclinee vers la droite", "penche a droite", "penchee a droite", "penche vers la droite", "penchee vers la droite", "tipped right", "tilted right", "leaning right"],
    "tip_forward":  ["couche", "couchee", "allonge", "allongee", "sur le ventre", "face contre terre", "penche en avant", "penchee en avant", "incline en avant", "inclinee en avant", "incline vers l avant", "inclinee vers l avant", "penche vers l avant", "penchee vers l avant", "bascule vers l avant", "basculee vers l avant", "tipped forward", "tilted forward", "leaning forward", "lying", "fallen forward"],
    "tip_backward": ["sur le dos", "renverse", "renversee", "penche en arriere", "penchee en arriere", "incline en arriere", "inclinee en arriere", "incline vers l arriere", "inclinee vers l arriere", "penche vers l arriere", "penchee vers l arriere", "bascule vers l arriere", "basculee vers l arriere", "tipped backward", "tilted backward", "leaning backward", "fallen backward"],
    "upside_down":  ["a l envers", "tete en bas", "retourne", "retournee", "renverse", "renversee", "inverse", "inversee", "a l envers", "upside down", "inverted", "flipped"],
    "turn_left":    ["tourne a gauche", "tournee a gauche", "pivote a gauche", "pivotee a gauche", "turned left", "rotated left", "facing left"],
    "turn_right":   ["tourne a droite", "tournee a droite", "pivote a droite", "pivotee a droite", "turned right", "rotated right", "facing right"],
    "turn_around":  ["fait demi-tour", "demi tour", "dos a", "tourne dans l autre sens", "tourne a 180", "180 degres", "de dos", "turned around", "facing away", "turned 180", "facing the other way", "facing opposite"],
}

CONFIG_INSTRUCTIONS = {
    "basic":            "Ecris une phrase simple et naturelle.",
    "with_distance":    "Mentionne explicitement chaque distance avec son unite (ex: 'a 50 cm', 'a 2 metres', 'a 1m5'). La valeur dans la phrase doit correspondre exactement a la distance dans la spec (en metres).",
    "with_orientation": "Decris explicitement la rotation/orientation de chaque objet concerne (ex: 'a l envers', 'couchee', 'tournee vers la gauche', 'sur le dos'). Utilise un terme naturel coherent avec le 'turn' donne.",
    "dense":            "Scene riche : exprime explicitement TOUTES les distances avec leur unite ET TOUTES les orientations donnees. Phrase fluide en plusieurs propositions.",
    "coreference":      "Un objet apparait dans plusieurs relations. Utilise obligatoirement 'cette X' ou 'ce X' pour le referencer la deuxieme fois (jamais un nouveau X).",
    "multi_instance":   "Il y a deux instances du meme type (ex: banane et banane_2). Introduis le premier normalement, le second avec 'une autre X' ou 'un deuxieme X'.",
    "longue":           "Ecris une description en plusieurs propositions enchainees.",
    "anglais":          "Write the sentence in English. Use 'another X' or 'a second X' for duplicate objects. Express distances in meters/cm naturally.",
    "stacking":         "Decris un empilement d'objets : utilise 'sur' pour chaque niveau de la pile. Ex: 'un mug sur un livre lui-meme pose sur la table'. La chaine de 'sur' doit refleter exactement les relations on.",
    "surface_commune":  "Plusieurs objets sont poses sur la meme surface. Formule naturellement avec 'sur' : 'sur la table : un mug, un livre et une banane' ou 'je veux une table avec dessus un mug et un livre'.",
    "aligned":          "Des objets sont alignes dans une direction. Exprime l'alignement de facon naturelle : 'une chaise, puis une table, puis une lampe alignees de gauche a droite' ou 'trois mugs cote a cote devant le frigo'.",
}

PARAPHRASE_STYLES = ["imperatif", "conversationnel", "nominatif"]
PARAPHRASE_PROB   = 0.30

PARAPHRASE_STYLE_INSTRUCTIONS = {
    "imperatif":      "Reformule la meme scene en style IMPERATIF (ex: 'place le mug sur la table, mets le livre a cote').",
    "conversationnel":"Reformule la meme scene en style CONVERSATIONNEL (ex: 'je voudrais un mug sur la table avec un livre a cote').",
    "nominatif":      "Reformule la meme scene en style NOMINATIF pur, sans sujet (ex: 'un mug sur la table, un livre a sa gauche').",
}

SYSTEM = """Tu es un generateur de donnees d'entrainement pour un modele NLP de scene 3D.
Tu recois une spec (objets, relations, orientations) et tu dois ecrire UNE SEULE phrase naturelle qui decrit exactement cette scene.

Reponds UNIQUEMENT avec la phrase, sans guillemets, sans JSON, sans explication.

STYLE : la phrase doit sonner naturelle, comme ce qu'un humain taperait pour decrire une scene.
- Prefere les formes nominales et conversationnelles : "une chaise sur le canape", "je veux un frigo a cote d'un lave-linge", "place le livre sur la table"
- Evite les formes robotiques du type "La chaise est sur le canape." ou "Le frigo est positionne a gauche de..."
- Varie les formulations : parfois sans sujet ("un bureau avec une lampe dessus"), parfois imperatif ("mets une plante devant la fenetre"), parfois "je veux/voudrais..."

OBLIGATOIRE : tous les objets de la spec doivent apparaitre dans la phrase. N'en oublie aucun.

DISTANCES : ecris toujours les distances en chiffres, JAMAIS en lettres. Correct : "a 1.05 m", "a 1m5", "a 50 cm". Interdit : "a un metre cinq", "a un metre et demi".

ORIENTATIONS — signification exacte de chaque valeur (ne pas confondre) :
- tip_left / tip_right : l objet est incline sur le cote (ex: "penche vers la gauche", "tilted right")
- tip_forward / tip_backward : l objet est incline vers l avant ou l arriere (ex: "penche en avant", "tilted backward")
- upside_down : l objet est retourne completement a l envers, tete en bas (ex: "a l envers", "upside down") — PAS turn_around
- turn_left / turn_right : l objet pivote sur lui-meme (ex: "tourne a gauche", "turned right")
- turn_around : l objet fait face dans la DIRECTION OPPOSEE, il a le dos vers nous (ex: "de dos", "tourne dans l autre sens", "facing away") — PAS a l envers"""

SYSTEM_VERIFY = """Tu es un verificateur de coherence pour des descriptions de scenes 3D.
Tu recois une spec (objets, relations, orientations) et une phrase censee la decrire.
Reponds UNIQUEMENT en JSON valide avec ce format :
{"ok": true/false, "errors": ["..."], "suggestion": "correction courte si ok=false, sinon null"}

Verifie :
1. Tous les objets de la spec sont mentionnes dans la phrase
2. Toutes les relations sont coherentes avec la phrase (pas de contradiction, bon sens)
3. Les distances sont correctement exprimees si presentes
4. Les orientations sont correctement decrites si presentes
5. Pas de relation inventee qui contredit la spec"""


def normalize(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return (text.lower()
            .replace("’", " ")   # ' right single quotation mark
            .replace("‘", " ")   # ' left single quotation mark
            .replace("ʼ", " ")   # modifier letter apostrophe
            .replace("`", " ")   # backtick
            .replace("'", " ")        # ASCII apostrophe
            .replace("-", " ")
            .replace("_", " "))


def display_name(obj_id, config=""):
    if config == "anglais":
        base = obj_id.rsplit("_", 1)[0] if (obj_id[-1].isdigit() and "_" in obj_id) else obj_id
        return base
    base = obj_id.rsplit("_", 1)[0] if (obj_id[-1].isdigit() and "_" in obj_id) else obj_id
    return OBJECT_NAMES_FR.get(base, base)


def object_in_phrase(obj_id, phrase, config=""):
    name = display_name(obj_id, config)
    name_norm   = normalize(name)
    phrase_norm = normalize(phrase)
    if name_norm in phrase_norm:
        return True
    words = name_norm.split()
    return len(words) > 1 and all(w in phrase_norm for w in words)


def parse_response(text):
    phrase = text.strip().strip('"').strip("'")
    return phrase if phrase else None


def phrase_distances_meters(phrase):
    found = []
    consumed_spans = []
    # "1m5", "1m60" (sans espace)
    for match in DISTANCE_REGEX_FR.finditer(phrase):
        meters  = int(match.group(1))
        dec_str = match.group(2)
        decimal = int(dec_str) / (10 ** len(dec_str))
        found.append(round(meters + decimal, 3))
        consumed_spans.append(match.span())
    # "1 m 5", "1 m 60" (avec espaces)
    for match in DISTANCE_REGEX_FR_SPACE.finditer(phrase):
        if any(start <= match.start() < end for start, end in consumed_spans):
            continue
        meters  = int(match.group(1))
        dec_str = match.group(2)
        decimal = int(dec_str) / (10 ** len(dec_str))
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
        found.append(round(val, 3))
    return found


def orientation_marker_in_phrase(turn, phrase):
    phrase_norm = normalize(phrase)
    for marker in ORIENTATION_MARKERS.get(turn, []):
        if normalize(marker) in phrase_norm:
            return True
    return False


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

    rel_keys = set((r["type"], r["subject"], r["object"]) for r in spec["relations"])
    for r in spec["relations"]:
        if (r["type"], r["object"], r["subject"]) in rel_keys:
            return False, "contradictory_relations_in_spec"

    for obj_id in spec["objets"]:
        if not object_in_phrase(obj_id, phrase, config=spec["config"]):
            return False, f"object_not_in_phrase:{obj_id}"

    for rel in spec["relations"]:
        dist = rel.get("distance")
        if dist is not None:
            phrase_dists = phrase_distances_meters(phrase)
            if not any(abs(d - float(dist)) < 0.05 for d in phrase_dists):
                return False, f"distance_not_in_phrase:{dist}"

    for o in spec.get("orientations", []):
        if not orientation_marker_in_phrase(o["turn"], phrase):
            return False, f"missing_orientation_marker:{o['id']}:{o['turn']}"

    if spec["config"] == "coreference":
        if not any(normalize(m) in phrase_norm for m in COREF_MARKERS):
            return False, "missing_coref_marker"

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
        "missing_multi_marker":       "Utilise 'une autre X' ou 'un deuxieme X' pour le second exemplaire.",
    }
    key = reason.split(":")[0] if ":" in reason else reason
    return hints.get(key, "Respecte exactement la spec fournie.")


def call_api(prompt, hint=None, temperature=0.7):
    _rate_limit()
    full_prompt = SYSTEM + "\n\n" + prompt
    if hint:
        full_prompt += f"\n\nCorrection requise : {hint}\nRegenere une nouvelle phrase."
    response = client.responses.create(
        model=MODEL,
        input=full_prompt,
        temperature=temperature,
    )
    return response.output_text


def call_verify(phrase, spec, obj_labels, rel_labels):
    _rate_limit()
    prompt = (
        f"Spec originale :\n"
        f"  objets       : {obj_labels}\n"
        f"  relations    : {json.dumps(rel_labels, ensure_ascii=False)}\n"
        f"  orientations : {json.dumps(spec.get('orientations', []), ensure_ascii=False)}\n\n"
        f"Phrase generee : \"{phrase}\"\n\n"
        f"Verifie la coherence et reponds en JSON."
    )
    response = client.responses.create(
        model=MODEL,
        input=SYSTEM_VERIFY + "\n\n" + prompt,
        temperature=0,
    )
    raw = response.output_text.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"ok": True, "errors": [], "suggestion": None}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"ok": True, "errors": [], "suggestion": None}


def _build_prompt(spec, style_override=None):
    use_fr = spec["config"] != "anglais"

    def obj_label(obj_id):
        if not use_fr:
            base = obj_id.rsplit("_", 1)[0] if (obj_id[-1].isdigit() and "_" in obj_id) else obj_id
            return base + ("_2" if obj_id.endswith("_2") else "")
        base = obj_id.rsplit("_", 1)[0] if (obj_id[-1].isdigit() and "_" in obj_id) else obj_id
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

    objets_label    = [obj_label(o) for o in spec["objets"]]
    relations_label = [rel_desc(r) for r in spec["relations"]]
    orient_label    = [{**o, "id": obj_label(o["id"])} for o in spec.get("orientations", [])]

    style        = "\n".join(f'- "{e}"' for e in STYLE_EXAMPLES[:6])
    config_instr = style_override or CONFIG_INSTRUCTIONS.get(spec["config"], "Ecris une phrase naturelle.")

    prompt = (
        f"Exemples de style :\n{style}\n\n"
        f"Spec :\n"
        f"  objets       : {objets_label}\n"
        f"  relations    : {json.dumps(relations_label, ensure_ascii=False)}\n"
        f"  orientations : {json.dumps(orient_label, ensure_ascii=False)}\n\n"
        f"Consigne : {config_instr}"
    )
    return prompt, objets_label, relations_label


def _generate_one_phrase(spec, prompt, obj_labels, rel_labels, max_retries=2):
    reason = None
    phrase = None
    hint   = None
    for attempt in range(max_retries + 1):
        text   = call_api(prompt, hint=hint)
        phrase = parse_response(text)
        if not phrase:
            hint = "Reponds uniquement avec la phrase, sans guillemets ni explication."
            continue

        valid, reason = phrase_valid(spec, phrase)
        if not valid:
            hint = _build_hint(reason)
            continue

        # verification semantique : second appel
        verify = call_verify(phrase, spec, obj_labels, rel_labels)
        if not verify.get("ok", True) and attempt < max_retries:
            suggestion = verify.get("suggestion")
            hint = suggestion if suggestion else "La phrase ne correspond pas a la spec. Reformule."
            continue

        return phrase

    if reason:
        print(f"\n[REJETE] raison : {reason}", flush=True)
        print(f"  spec   : objets={spec['objets']} config={spec['config']}", flush=True)
        print(f"  phrase : {phrase}", flush=True)
    return None


def generate_example(spec, max_retries=2):
    prompt, obj_labels, rel_labels = _build_prompt(spec)
    phrase = _generate_one_phrase(spec, prompt, obj_labels, rel_labels, max_retries)

    if phrase is None:
        return None

    base_output = {
        "objets":       spec["objets"],
        "relations":    spec["relations"],
        "orientations": spec.get("orientations", []),
    }

    entries = [{"input": phrase, "config": spec["config"], "output": base_output}]

    # paraphrase transversale : 30% de chance, 2 phrases alternatives
    if random.random() < PARAPHRASE_PROB:
        styles = random.sample(PARAPHRASE_STYLES, 2)
        for style_key in styles:
            style_instr = PARAPHRASE_STYLE_INSTRUCTIONS[style_key]
            para_prompt, _, _ = _build_prompt(spec, style_override=style_instr)
            para_phrase = _generate_one_phrase(spec, para_prompt, obj_labels, rel_labels, max_retries=1)
            if para_phrase and para_phrase != phrase:
                entries.append({"input": para_phrase, "config": spec["config"], "output": base_output})

    return entries


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from spec_generator import generate_spec

    for config in ["basic", "with_distance", "stacking", "surface_commune", "aligned", "coreference", "negation"]:
        spec = generate_spec(config=config)
        print(f"\n[{config}] spec : {spec['objets']}")
        print(f"  relations    : {spec['relations']}")
        print(f"  orientations : {spec['orientations']}")
        entries = generate_example(spec)
        if entries:
            for e in entries:
                print(f"  phrase  : {e['input']}")
            print(f"  output  : {json.dumps(entries[0]['output'], ensure_ascii=False)}")
        else:
            print("  -> rejete apres retries")
