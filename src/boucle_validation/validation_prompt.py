
import os
import base64
import ollama
import openai
import json
from .jsonToSim import validation_physique
from .validation_utils import create_run_dir, correct_list, clean_reponse, save_iteration_scene
import json
import copy

import sys
import os
import json



def boucle_vlm_prompt(user_prompt, jsonFile, max_iter=5):

    run_dir = create_run_dir()
    history = []

    with open(jsonFile, 'r') as file:
            data = json.load(file)

    for iter in range(max_iter):
        itemsList = data if isinstance(data, list) else data.get('objets', [])
        image_path, corrected_objects = validation_physique(itemsList)

        save_iteration_scene(image_path, iter, run_dir, itemsList)
        res, data = validation_semantique_prompt(user_prompt, corrected_objects, image_path)
        #print(f"DEBUG: les res keys sont {res.keys()} et le contenu est {res}")
        #print(f"DATA2: {data}")

        history.append(copy.deepcopy({
            'iteration': iter,
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False),
            'scene': data
        }))

        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            print("history:", history)
            json.dump(history, f, indent=4)

        if res.get("valid")==True:
            print("scene validée")
            return data
        else:
            print("echec: ", res.get("feedback"))

    else:
        return data



def validation_semantique_prompt(original_prompt, data, image_path):
    pos_and_dims = [{'id': item['id'], 'pos': item['pos'], 'dimensions':item['dimensions'], 'lowest_point':item['lowest_point'],'highest_point':item['highest_point']} for item in data]
    prompt = """ You are a 3D Scene Validator and Spatial Coordinator. Your goal is to ensure objects are logically placed and match the prompt's description.

            You will receive:
                - The current scene as a JSON object
                - The position, dimensions , lowest point and highest point of each object
                - a collage containing THREE images of the same scene:
                    1. PERSPECTIVE VIEW: General context.
                    2. TOP-DOWN VIEW: Best for X, Y coordinates and checking if objects are side-by-side.
                    3. SIDE VIEW: Best for Z coordinate (height) to check if objects are at the right height.
                        
            Spatial RULES:
            - Stacking (On Top): For Object A to be "on" Object B, X and Y must be within the space covered by object B. 
            Object A's lowest point must be 0.001 higher than object B highest point.
            - Next to: for object A to be next to object B, they must have the same height (Z). X and Y may vary depending on if they are at the right, left, in front, behind each other… For example, if the prompt says "Object A is next to Object B", ensure they have the same Z-base but different X or Y.
            - Collisions: Objects must not occupy the same space. If they overlap in the Top-Down view, they must have different Z-heights to avoid clipping. If an object is explicitly inside another, this does not apply. However, meshes still shouldn’t intersect.
            - Ground Plane: No object's lowest point should be below 0.
            - DO NOT throw objects in the air! Objects should be placed on a surface (ground or other objects).

            Other RULES:
                - ONLY change pos if needed
                -  Maintain the original id for all objects.
                - valid is true only if scene matches prompt AND no collisions
                - ONLY output the JSON object, nothing else
                - NO markdown, NO backticks, NO explanations before or after


            You MUST respond with ONLY this JSON format:
            {
                "valid": boolean,
                "feedback":  "Reasoning for changes (e.g., 'Moving mug to be centered on table and fixing ground clipping')",
                "corrections": [{"id": "string", "pos": [x, y, z]}]
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
        model="gpt-4o-mini",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': f"Prompt: {original_prompt}\nObjects and positions: {json.dumps(pos_and_dims)}"},
                {'type': 'image_url', 'image_url': {'url': f"data:image/png;base64,{image_b64}"}}
            ]}
        ]
    )
    clean = clean_reponse(response.choices[0].message.content)
    

    #clean = clean_reponse(resultat['message']['content'])
    #print("CLEAN: ", clean)

    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour etapes_validation")
        return {"valid": False, "feedback": "json invalide"}
    if 'corrections' in resultat:
        corrections = {c['id']: c['pos'] for c in resultat['corrections']}
        data = correct_list(data, corrections, 'pos')
        #print("CORRECTED DATA:", data)
    return resultat, data
