
import os
import base64
import ollama
import openai
import json
from .jsonToSim import validation_physique
from .validation_utils import create_run_dir, correct_list, clean_reponse, save_iteration_scene
from PIL import Image
import xml.etree.ElementTree as ET
import copy

def boucle_vlm_img(user_image_path, jsonFile, max_iter=5):
    '''
    Boucle de validation qui compare la scène générée à une image de référence fournie par l'utilisateur.
    '''
    run_dir = create_run_dir()
    history = []

    with open(jsonFile, 'r') as file:
        data = json.load(file)
    itemsList = data if isinstance(data, list) else data.get('objets', [])
    data = itemsList

    for iter in range(max_iter):

        scene_image_path, corrected_objects = validation_physique(data)

        print("CORRECTED OBJECTS:", corrected_objects)

        save_iteration_scene(scene_image_path, iter, run_dir, itemsList)

        res, data = validation_semantique_img(user_image_path, corrected_objects, scene_image_path)
        print(f"DEBUG: les res keys sont {res.keys()} et le contenu est {res}")

        history.append(copy.deepcopy({
            'iteration': iter,
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False),
            'scene': data
        }))
        
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        if res.get("valid") == True:
            print("Scène validée")
            return data
        else:
            print("Qcène non validée: ", res.get("feedback"))
            itemsList = data

    else:
        print("Nombre maximum d'itérations atteint.")
        scene_image_path, corrected_objects = validation_physique(data)
        return corrected_objects


def validation_semantique_img(user_image_path, data, scene_image_path):
    '''
    Envoie l'image cible et l'image de la scène au VLM pour obtenir les corrections de coordonnées.
    '''
    pos_and_dims = [{'id': item['id'], 'pos': item['pos'], 'dimensions':item['dimensions']} for item in data]
    
    prompt = """You are a 3D Scene Validator and Spatial Coordinator. Your goal is to ensure the simulated objects match the spatial arrangement seen in the REFERENCE image.

            You will receive:
                - Image 1: The REFERENCE image (what the scene must look like).
                - Image 2: The CURRENT SIMULATIONo
                - A JSON list with the current positions (pos) and dimensions of each object.

            Spatial RULES:
            - GOAL: Adjust the X, Y, Z coordinates of the simulated objects so they match the relative positioning, stacking, and alignment shown in the REFERENCE image.
            - Stacking (On Top): For Object A to be "on" Object B, X and Y must be within the space covered by object B. Object A's lowest point must be 0.001 higher than object B's highest point.
            - Right of, left of, behind, in front of...: for object A to be in these relations to object B, they should be near each other but never touching or overlapping. Their lowest-point should be approximately equal. X and Y may vary depending on if they are at the right, left, in front, behind each other… For example, if the prompt says "Object A is next to Object B", ensure they have the same Z-base but different X or Y.
            - Collisions: Objects must not intersect unless explicitly shown in the reference image. If they overlap in the Top-Down view, they must have different Z-heights to avoid clipping, or different X or Y.
            - Ground Plane: No object's lowest point should be below 0.

            Other RULES:
                - ONLY change pos if needed to match the reference image.
                - Maintain the original id for all objects.
                - 'valid' is true ONLY if the current simulation perfectly matches the reference image arrangement AND has no illogical collisions.
                - ONLY output the JSON object, nothing else.
                - NO markdown, NO backticks, NO explanations before or after.

            IMPORTANT: Valid is True if the scene matches the image.

            You MUST respond with ONLY this JSON format:
            {
                "valid": boolean,
                "feedback": "Reasoning for changes (e.g., 'Moving mug to be on the left of the laptop to match the reference image')",
                "corrections": [{"id": "string", "pos": [x, y, z]}]
            }
   """

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
        model="gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f"Current Objects and positions: {json.dumps(pos_and_dims)}\nReview the reference image (first) and the current simulation (second)."},
                {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{encode(user_image_path)}"}},
                {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{encode(scene_image_path)}"}}
            ]}
        ]
    )

    clean = clean_reponse(response.choices[0].message.content)
    

    #clean = clean_reponse(resultat['message']['content'])
    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour etapes_validation")
        return {"valid": False, "feedback": "json invalide"}
        
    if 'corrections' in resultat:
        corrections = {c['id']: c['pos'] for c in resultat['corrections']}
        data = correct_list(data, corrections, 'pos')
        
    return resultat, data