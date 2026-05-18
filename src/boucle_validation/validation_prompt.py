
import os
import base64
import ollama
import openai
import json
from .jsonToSim import validation_physique
from .validation_utils import create_run_dir, clean_reponse, save_iteration_scene, apply_semantic_corrections
import copy
import sys
import os
import json



def boucle_vlm_prompt(user_prompt, jsonFile, max_iter=10):

    run_dir = create_run_dir()
    history = []
    fixed_ids = set()

    with open(jsonFile, 'r') as file:
        data = json.load(file)

    for iter in range(max_iter):
        itemsList = data if isinstance(data, list) else data.get('objets', [])
        image_path, corrected_objects = validation_physique(itemsList, fixed_ids=fixed_ids)

        save_iteration_scene(image_path, iter, run_dir, itemsList)
        res, data = validation_semantique_prompt(user_prompt, corrected_objects, image_path)
        
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
            print("scène validée")
            return data

        corrections = res.get('corrections', [])
        all_ids = {item['id'] for item in data}
        corrected_ids = {c['subject'] for c in corrections if c.get('subject')}
        fixed_ids = all_ids - corrected_ids  # on ne fixe que les objets deja bien places
        print(f"objets fixes pour iter suivante: {fixed_ids}")

    else:
        itemsList = data if isinstance(data, list) else data.get('objets', [])
        image_path, corrected_objects = validation_physique(itemsList, fixed_ids=fixed_ids)
        return corrected_objects


def validation_semantique_prompt(original_prompt, data, image_path):
    pos_and_dims = [
        {'id': item['id'], 'pos': item['pos'], 'dimensions': item['dimensions'],
         'lowest_point': item['lowest_point'], 'highest_point': item['highest_point']}
        for item in data
    ]

    prompt = """You are a 3D scene validator. Identify violated spatial relations between the rendered scene and the user's prompt.

            You receive:
            - The user prompt describing the desired scene.
            - A JSON list of objects with their current positions and dimensions.
            - A screenshot of the current simulation.

            Coordinate system (context only — do NOT output coordinates):
            X = depth (positive toward camera), Y = lateral (positive = right), Z = vertical (positive = up).
            Camera is at (3.5, 0, 2.5) looking at origin, FOV=30.

            Supported correction types: "on", "inside", "right_of", "left_of", "in_front_of", "behind"
            For "on" and "inside": you may also provide rel_x and rel_y in [0.0, 1.0] to indicate WHERE on the surface
            (rel_x: 0=left side, 0.5=center, 1=right side / rel_y: 0=front side, 0.5=center, 1=back side).
            Omit rel_x/rel_y if centering on the surface is acceptable.

            RULES:
            - Set valid=true if the scene matches the prompt with no obvious violations.
            - Only add a correction if a spatial relation is clearly violated.
            - NEVER output coordinate numbers. Use relation types and optional ratios only.
            - "corrections": [] if everything is correct.
            - Empty corrections list means valid must be true.

            You MUST respond with ONLY this JSON, nothing else:
            {
            "valid": boolean,
            "feedback": "brief description of what is wrong or why it is valid",
            "corrections": [
                {"subject": "<object_id>", "type": "<relation>", "object": "<reference_id>"},
                {"subject": "<object_id>", "type": "on", "object": "<reference_id>", "rel_x": 0.2, "rel_y": 0.8}
            ]
            }
    """

    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    '''
    resultat = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user',
            'content': f"Prompt: {original_prompt}\nObjects and positions: {json.dumps(pos_and_dims)}",
            'images': [image_path]}
        ]
    )
    '''
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f"Prompt: {original_prompt}\nObjects: {json.dumps(pos_and_dims)}"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{image_b64}"}}
            ]}
        ],
        response_format={"type": "json_object"},
    )
    clean = clean_reponse(response.choices[0].message.content)
    
    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour validation_semantique_prompt")
        return {"valid": False, "feedback": "json invalide", "corrections": []}, data
    corrections = resultat.get('corrections', [])
    if not corrections:
        resultat['valid'] = True
    if corrections:
        data = apply_semantic_corrections(data, corrections)
    return resultat, data
