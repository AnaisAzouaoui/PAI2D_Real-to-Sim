"""
Remplacement drop-in de extract_labels() dans object_rec_v1_1_embedding.py.

Utilise Ollama en local (modele GGUF Q4_K_M).

Prerequis :
  1. Telecharger le .gguf depuis Colab
  2. Creer le modele Ollama :
    ollama create phi3-scene -f phi3_finetuned_gguf_gguf/Modelfile
  3. Verifier : ollama run phi3-scene "test"

Usage dans object_rec_v1_1_embedding.py :
    from training.inference.extractor import extract_labels
"""
import json
import re
import urllib.request
import urllib.error

OLLAMA_MODEL  = "phi3-scene"
OLLAMA_URL    = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "Tu es un assistant qui extrait les objets, relations spatiales, distances et orientations "
    "d'une description de scene. Reponds uniquement en JSON valide.\n"
    'Format : {"objets": [...], '
    '"relations": [{"type": "...", "subject": "...", "object": "...", "distance": <float optionnel>}], '
    '"orientations": [{"id": "...", "turn": "..."}]}\n'
    "Types de relations : on, under, left_of, right_of, in_front_of, behind, against, inside\n"
    "Champ 'distance' (en metres) autorise UNIQUEMENT sur : left_of, right_of, in_front_of, behind\n"
    "Turns valides : tip_left, tip_right, tip_forward, tip_backward, upside_down, turn_left, turn_right, turn_around\n"
    "Convention : premiere instance = 'banane', deuxieme = 'banane_2'.\n"
    "Si pas de distance mentionnee : ne pas inclure le champ. Si tout est debout : orientations = [].\n"
    "Si plusieurs objets sont mentionnes sans relation explicite, deduis un positionnement lateral par defaut (left_of ou right_of)."
)


def call_ollama(prompt: str) -> str:
    full_prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n"
    )
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 256},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


def parse_output(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def extract_labels(prompt: str) -> dict:
    """
    Extrait objets, relations (avec distances optionnelles) et orientations
    depuis un prompt utilisateur.

    Retourne :
    {
        "objets"       : ["mug", "banane", "lave_linge", "banane_2"],
        "relations"    : [{"type": "left_of", "subject": "mug", "object": "banane", "distance": 2.0}, ...],
        "orientations" : [{"id": "mug", "turn": "upside_down"}, ...]
    }
    """
    try:
        raw = call_ollama(prompt)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama inaccessible ({e}). Lancer 'ollama serve' puis verifier que le modele '{OLLAMA_MODEL}' existe."
        ) from e

    result = parse_output(raw)
    if result is None:
        return {"objets": [], "relations": [], "orientations": []}

    result.pop("root", None)
    result.setdefault("orientations", [])
    result.setdefault("relations", [])
    return result


if __name__ == "__main__":
    test_prompts = [
        "je veux un mug a cote d'une banane",
        "un frigo contre le mur avec une poubelle a sa droite",
        "un mug sur un livre lui-meme pose sur la table",
        "sur le bureau : un ordinateur, une lampe et un mug",
        "une chaise, puis une table, puis une lampe alignees de gauche a droite",
    ]
    for p in test_prompts:
        print(f"\nPrompt : {p}")
        result = extract_labels(p)
        print(f"  objets       : {result['objets']}")
        print(f"  relations    : {result['relations']}")
        print(f"  orientations : {result['orientations']}")
