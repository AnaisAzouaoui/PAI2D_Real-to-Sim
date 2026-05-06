
import os
import base64
import ollama
import openai
import json
from .jsonToSim import validation_physique
from .validation_utils import create_run_dir, clean_reponse, save_iteration_scene, apply_semantic_corrections
from PIL import Image
import xml.etree.ElementTree as ET
import copy


def boucle_vlm_img(user_image_path, jsonFile, max_iter=10):
    '''
    Boucle de validation qui compare la scène générée à une image de référence fournie par l'utilisateur.
    '''
    run_dir = create_run_dir()
    history = []
    fixed_ids = set()

    with open(jsonFile, 'r') as file:
        data = json.load(file)
    itemsList = data if isinstance(data, list) else data.get('objets', [])
    data = itemsList

    for iter in range(max_iter):

        scene_image_path, corrected_objects = validation_physique(data, fixed_ids=fixed_ids)

        save_iteration_scene(scene_image_path, iter, run_dir, data)

        res, data = validation_semantique_img(user_image_path, corrected_objects, scene_image_path)
        print(f"iteration {iter}: valid={res.get('valid')}, feedback={res.get('feedback')}")

        history.append(copy.deepcopy({
            'iteration': iter,
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections', ''),
            'valid': res.get('valid', False),
            'scene': data
        }))

        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        if res.get("valid") == True:
            print("Scène validée")
            return data

        # fixer tous les objets pour la prochaine physique :
        # - ceux qui n'avaient pas de corrections (deja corrects)
        # - ceux qui viennent d'etre corriges (leur position est bonne, ne pas la perdre)
        # only les objets qui seraient encore problematiques a la prochaine iteration restent libres
        corrections = res.get('corrections', [])
        all_ids = {item['id'] for item in data}
        fixed_ids = all_ids  # tout fixer : la verification VLM suffit, pas besoin de physique
        print(f"objets fixes pour iter suivante: {fixed_ids}")

    else:
        print("Nombre maximum d'iterations atteint")
        scene_image_path, corrected_objects = validation_physique(data, fixed_ids=fixed_ids)
        return corrected_objects


def validation_semantique_img(user_image_path, data, scene_image_path):
    """Compare la scene simulee a l'image de reference et retourne des corrections semantiques"""
    pos_and_dims = [
        {'id': item['id'], 'pos': item['pos'], 'dimensions': item['dimensions']}
        for item in data
    ]

    prompt = """You are a 3D scene validator. Compare the REFERENCE image with the CURRENT SIMULATION and identify spatial relation violations.

You receive:
- Image 1: The REFERENCE image (what the scene must look like).
- Image 2: The CURRENT SIMULATION screenshot.
- A JSON list of objects with their current positions and dimensions.

Coordinate system (context only — do NOT output coordinates):
  X = depth (positive toward camera), Y = lateral (positive = right), Z = vertical (positive = up).
  Camera is at (3.5, 0, 2.5) looking at origin, FOV=30.

Supported correction types: "on", "inside", "right_of", "left_of", "in_front_of", "behind"
For "on" and "inside": you may also provide rel_x and rel_y in [0.0, 1.0] to indicate WHERE on the surface
  (rel_x: 0=left side, 0.5=center, 1=right side / rel_y: 0=front side, 0.5=center, 1=back side).
  Use the reference image to estimate rel_x/rel_y when the object is not centered on its support.
  Omit rel_x/rel_y if centering is acceptable.

RULES:
- Set valid=true if the simulation matches the reference image arrangement with no obvious violations.
- Only add a correction if an object is clearly in the wrong place relative to another.
- NEVER output coordinate numbers. Use relation types and optional ratios only.
- "corrections": [] if everything matches.
- Empty corrections list means valid must be true.

You MUST respond with ONLY this JSON, nothing else:
{
  "valid": boolean,
  "feedback": "brief description of what is wrong or why it is valid",
  "corrections": [
    {"subject": "<object_id>", "type": "<relation>", "object": "<reference_id>"},
    {"subject": "<object_id>", "type": "on", "object": "<reference_id>", "rel_x": 0.2, "rel_y": 0.8}
  ]
}"""

    def encode(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
        
    '''
    resultat = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user',
            'content': f"Current Objects and positions: {json.dumps(pos_and_dims)}\nReview the reference image (first) and the current simulation (second).",
            'images': [user_image_path, scene_image_path]}
        ]
    )
    '''

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f"Objects: {json.dumps(pos_and_dims)}\nImage 1 is the reference. Image 2 is the current simulation."},
                {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{encode(user_image_path)}"}},
                {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{encode(scene_image_path)}"}}
            ]}
        ],
        response_format={"type": "json_object"},
    )

    clean = clean_reponse(response.choices[0].message.content)

    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour validation_semantique_img")
        return {"valid": False, "feedback": "json invalide", "corrections": []}, data

    corrections = resultat.get('corrections', [])
    if not corrections:
        resultat['valid'] = True

    if corrections:
        data = apply_semantic_corrections(data, corrections)

    return resultat, data
