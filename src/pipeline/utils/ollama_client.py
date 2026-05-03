import json
import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from .catalogue import objects_desc, objets_list

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

URL = "http://localhost:11434/api/generate"

RE_JSON = re.compile(r"\{.*\}", re.DOTALL)

# OpenAI  -> "gpt-4o", "gpt-4o-mini"
# Ollama  -> "llama3.1"
LLM_MODEL = "gpt-4o"

openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return openai_client


def is_openai_model(model):
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


def validate_json_response(payload):
    try:
        response = requests.post(URL, json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Impossible de joindre Ollama.")
    if response.status_code != 200:
        raise RuntimeError(f"Erreur Ollama (HTTP {response.status_code}): {response.text}")
    raw = response.json().get("response", "")
    json_match = RE_JSON.search(raw)
    if not json_match:
        raise ValueError("Le LLM n'a pas retourne de JSON valide")
    return json.loads(json_match.group())


def call_llm(system_prompt, user_prompt):
    """ (OpenAI) ou llama"""
    if is_openai_model(LLM_MODEL):
        print(f"[LLM] OpenAI")
        response = get_openai_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user","content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    else:
        print(f"[LLM] Ollama")
        payload = {
            "model": LLM_MODEL,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        return validate_json_response(payload)


def classify_intent(prompt, scene_summary):
    """Classifie le prompt en 'modify_scene' ou 'new_scene'"""
    system_prompt = f"""You classify user instructions about a 3D scene.

        Current scene: {scene_summary}

        If the user wants to MODIFY the existing scene (move, remove, add, rotate, resize objects), return: {{"intent": "modify_scene"}}
        If the user wants to CREATE a completely new scene (ignoring the current one), return: {{"intent": "new_scene"}}

        Return ONLY raw JSON, no markdown, no comments."""

    result = call_llm(system_prompt, prompt)
    intent = result.get("intent", "new_scene")
    return intent if intent in ("modify_scene", "new_scene") else "new_scene"


def suggest_alternatives(label):
    """Propose 2-3 objets du catalogue les plus proches du label non reconnu."""
    catalog_desc = objects_desc()
    objects_data = objets_list()

    system_prompt = f"""You suggest the closest alternatives from a 3D object catalogue for an unrecognized object.

        CATALOGUE:
        {catalog_desc}

        Return ONLY this JSON: {{"alternatives": ["urdf_key_1", "urdf_key_2"]}}
        - Pick the 2-3 closest matches (same category or similar function)
        - If nothing is remotely close, return {{"alternatives": []}}
        - Each value must be an exact urdf key from the catalogue
        - output raw JSON only, no markdown, no comments"""

    alternatives = call_llm(system_prompt, f'The user asked for: "{label}"')
    return [a for a in alternatives if a in objects_data]
