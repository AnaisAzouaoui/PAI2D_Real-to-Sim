
import os
import ollama
import json
import re
import xml.etree.ElementTree as ET
import trimesh
import shutil
from datetime import datetime
from .jsonToSim import create_scene_validation
import json
from scipy.spatial.transform import Rotation as R
import copy

def getFilePath(item):
    '''
    Donne le path pour l'item URDF.

    :param item: le dictionnaire d'item avec id et urdf
    :return path: le path pour l'item
    '''
    items_folder = os.path.join(os.path.dirname(__file__),'..', 'objets')
    base = os.path.join(items_folder, item['urdf'])
    if not os.path.exists(base):
        raise FileNotFoundError(f"Pour {item['id']}, le fichier {item['urdf']} est introuvable.")
    #path = addMass(path)

    #TODO: question: pourquoi on cherche d'autres fichiers que mobility.urdf?
    # si c'est un dossier on cherche un fichier qui existe
    if os.path.isdir(base):
        for name in ["mobility.urdf", "kinbody.xml", "textured.obj", "nontextured.stl", "nontextured.ply"]:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Aucun fichier exploitable trouvé dans {base}")

    return base


def getOriginalDimensions(items): #TODO: attention c'est pas la meme que dans v1, à changer
    '''
    Extrait les dimensions de l'item URDF, qui sont dans '<geometry>'

    :param filepath: le path pour le file urdf
    :return: l'item avec ses dimensions
    '''
    for item in items:
        if item.get('dimensions'):
            continue
        else:
            tree = ET.parse(item['path']) #lecture du fichier urdf en xml
            root = tree.getroot()

            for geometry in root.iter('geometry'):
                box= geometry.find('box')
                cylinder= geometry.find('cylinder')
                sphere= geometry.find('sphere')
                mesh_tag = geometry.find('mesh') #quand les formes sont plus complexes c'est généralement mesh qui est utilisé
                if box is not None:
                    size = box.attrib['size'].split()
                    item['dimensions'] = (float(size[0]),float(size[1]), float(size[2]))
                elif cylinder is not None:
                    r = float(cylinder.attrib['radius'])
                    length = float(cylinder.attrib['length'])
                    item['dimensions'] = (r*2, r*2, length)
                elif sphere is not None:
                    r = float(sphere.attrib['radius'])
                    item['dimensions'] = (r*2, r*2, r*2)
                elif mesh_tag is not None:
                    directory = os.path.dirname(item['path'])
                    path_mesh = os.path.join(directory, mesh_tag.attrib['filename'])
                    mesh = trimesh.load(path_mesh, force='mesh') #force='mesh' est pour ne pas avoir de scene, seulement un mesh unique
                    bounds = mesh.bounding_box.extents
                    item['dimensions'] = (float(bounds[0]),float(bounds[1]),float(bounds[2]))
                else:
                    print("Dimensions non trouvées") #TODO: faire une meilleure erreur
                    item['dimensions'] = (0.1,0.1,0.1)
    return items

def create_run_dir():
    '''un dossier pour chaque run'''
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def checkOverlap(items):
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i >= j:
                continue
            w1, d1, h1 = item1['dimensions']
            w2, d2, h2 = item2['dimensions']
            x1, y1, z1 = item1['pos']
            x2, y2, z2 = item2['pos']
            
            if (abs(x1-x2) < (w1+w2)/2 and abs(y1-y2) < (d1+d2)/2 and abs(z1-z2) < (h1+h2)/2):
                print(f"overlap détecté entre {item1['id']} et {item2['id']}!")
                return True
    return False

#TODO: check 'on'


def save_iteration_scene(image_path, iter, run_dir, phase, itemsList):
    shutil.copy(image_path, os.path.join(run_dir, f"iteration_{phase}_{iter}_image.png"))
    with open(os.path.join(run_dir, f'iter_{iter}_scene.json'), 'w') as f:
        json.dump({'objets': itemsList}, f, indent=4)


def correct_json(jsonFile, corrections, field):
    with open(jsonFile, 'r') as file:
        data = json.load(file)
    existing_id = {item['id'] for item in data['objets']}
    for c_id in corrections.keys():
        if c_id not in existing_id:
            print(f"ATTENTION: l'ID {c_id} n'est pas dans le JSON, skipping correction.")
    for item in data['objets']:
        if item['id'] in corrections:
            old_value = item[field]
            item[field] = corrections[item['id']]
            print(f"CORRECTION: {field} corrigée pour {item['id']}: {old_value} → {item[field]}")
    with open(jsonFile, 'w') as file:
        json.dump(data, file, indent=4)


def clean_reponse(resultat):
    # #resultat = re.sub(r'```json|```', '', resultat) 
    # #resultat = resultat.strip()
    # #resultat = resultat.replace("'", '"') 
    # #resultat = resultat.replace('True', 'true').replace('False', 'false')
    # start = resultat.find('{')
    # end = resultat.rfind('}') + 1
    # if start != -1 and end != 0:
    #     resultat = resultat[start:end]
    # return resultat


    resultat = re.sub(r'```', '', resultat)
    start = resultat.find('{')
    end = resultat.rfind('}') + 1
    if start != -1 and end != 0:
        json_str = resultat[start:end]
        json_str = json_str.replace('True', 'true').replace('False', 'false') 
        return json_str
    
    return ""


def boucle_vlm(user_prompt, jsonFile, max_iter=3):

    run_dir = create_run_dir()
    history = []

    print("Phase 1: fixage sol")
    with open(jsonFile, 'r') as file:
        data = json.load(file)
    itemsList = data.get('objets', [])
    itemsList = getOriginalDimensions(itemsList)
    image_path = create_scene_validation(itemsList, fixed=True)
    save_iteration_scene(image_path, "init", run_dir, "sol", itemsList)

    res = verif_fixage_sol(jsonFile, image_path)

    history.append({
        'iteration': "init",
        'phase': "sol",
        'feedback': res.get('feedback', ''),
        'corrections': res.get('corrections',''),
        'valid': res.get('valid', False)
    })
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    if res.get('valid'):
        print("FIXAGE VALIDE")
    else:
        print("FEEDBACK: ", res.get('feedback', 'pas de feedback'))
        
    for iter in range(max_iter):
        score_parfait = True #le score doit etre parfait dans toutes les évaluations pour que la scène soit valide
        print("ITERATION", iter)
        print("Phase 2: orientation")
        with open(jsonFile, 'r') as file:
            data = json.load(file)
        itemsList = data.get('objets', [])
        itemsList = getOriginalDimensions(itemsList)

        image_path = create_scene_validation(itemsList, fixed=True)
        save_iteration_scene(image_path, iter, run_dir, "orientation", itemsList)

        res = verif_orientation(jsonFile, image_path, user_prompt)

        print(f"DEBUG: les res keys sont {res.keys()} et le contenu est {res}")

        history.append({
            'iteration': iter,
            'phase': "orientation",
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False)
        })
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        if res.get('valid'):
            print("ORIENTATION VALIDE")
        else:
            print("FEEDBACK: ", res.get('feedback', 'pas de feedback'))
            score_parfait = False
    

        print("Phase 3: Sémantique et collisions")
        with open(jsonFile, 'r') as file:
            data = json.load(file)
        itemsList = data.get('objets', [])
        itemsList = getOriginalDimensions(itemsList)

        has_overlap = checkOverlap(itemsList)
        collision_feedback = "overlap detected" if has_overlap else "no overlap"
        image_path = create_scene_validation(itemsList, fixed=False)

        save_iteration_scene(image_path, iter, run_dir, "etapes", itemsList)

        res = etapes_validation(user_prompt,jsonFile,image_path, collision_feedback)

        print(f"DEBUG: les res keys sont {res.keys()} et le contenu est {res}")

        history.append(copy.deepcopy({
            'iteration': iter,
            'phase': "semantique",
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False)
        }))
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        if res.get("valid")==True:
            print("scene validée")
        else:
            score_parfait = False
            print("echec: ", res.get("feedback"))

        print("Phase 1: Fixage sol")
        with open(jsonFile, 'r') as file:
            data = json.load(file)
        itemsList = data.get('objets', [])
        itemsList = getOriginalDimensions(itemsList)
        image_path = create_scene_validation(itemsList, fixed=True)
        save_iteration_scene(image_path, iter, run_dir, "sol", itemsList)

        res = verif_fixage_sol(jsonFile, image_path)
        print(f"DEBUG: les res keys sont {res.keys()} et le contenu est {res}")


        history.append({
            'iteration': iter,
            'phase': "sol",
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False)
        })
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        if res.get('valid'):
            print("Fixage valide")
        else:
            score_parfait = False
            print("FEEDBACK: ", res.get('feedback', 'pas de feedback'))
        
        if score_parfait:
            return res
    else:
        return res


def boucle_vlm_old(user_prompt, jsonFile, image_path, max_iter=3):

    run_dir = create_run_dir()
    history = []

    for iter in range(max_iter):
        print("BOUCLE FIXAGE, ITERATION ", iter)
        with open(jsonFile, 'r') as file:
            data = json.load(file)
        itemsList = data.get('objets', [])
        itemsList = getOriginalDimensions(itemsList)
        image_path = create_scene_validation(itemsList, fixed=True)
        save_iteration_scene(image_path, iter, run_dir, "sol", itemsList)

        res = verif_fixage_sol(jsonFile, image_path)

        history.append({
            'iteration': iter,
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False)
        })
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        if res.get('valid'):
            print("FIXAGE VALIDE")
            break
        else:
            print("FEEDBACK: ", res.get('feedback', 'pas de feedback'))
    else:        
        print("Fixage sol non validé après ", max_iter, " itérations")


    for iter in range(max_iter):
        print("BOUCLE ORIENTATION, ITERATION ", iter)
        with open(jsonFile, 'r') as file:
            data = json.load(file)
        itemsList = data.get('objets', [])
        itemsList = getOriginalDimensions(itemsList)

        image_path = create_scene_validation(itemsList, fixed=True)
        save_iteration_scene(image_path, iter, run_dir, "orientation", itemsList)

        res = verif_orientation(jsonFile, image_path, user_prompt)

        history.append({
            'iteration': iter,
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False)
        })
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        if res.get('valid'):
            print("ORIENTATION VALIDE")
            break
        else:
            print("FEEDBACK: ", res.get('feedback', 'pas de feedback'))
    else:     
        print("Orientation non validée après ", max_iter, " itérations")

    for iter in range(max_iter):
        print("BOUCLE SEMANTIQUE + COLLISIONS, ITERATION ", iter)
        with open(jsonFile, 'r') as file:
            data = json.load(file)
        itemsList = data.get('objets', [])
        itemsList = getOriginalDimensions(itemsList)

        has_overlap = checkOverlap(itemsList)
        collision_feedback = "overlap detected" if has_overlap else "no overlap"
        image_path = create_scene_validation(itemsList, fixed=True)

        save_iteration_scene(image_path, iter, run_dir, "etapes", itemsList)

        res = etapes_validation(user_prompt,jsonFile,image_path, collision_feedback)

        history.append({
            'iteration': iter,
            'feedback': res.get('feedback', ''),
            'corrections': res.get('corrections',''),
            'valid': res.get('valid', False)
        })
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        if res.get("valid")==True:
            print("scene validée")
            return res
        else:
            print("echec: ", res.get("feedback"))
    return res


def verif_fixage_sol(jsonFile, image_path): #TODO: API perception genesis plutot pour verifier la vitesse dans les 100 pas si des objets volent ou tombent
    with open(jsonFile, 'r') as file:
        data = json.load(file)
    z_and_dims = [
        {
            'id': item['id'], 
            'z': item['pos'][2],
            'height': item['dimensions'][2] if 'dimensions' in item else None,
            'suggested_min_z': round(item['dimensions'][2] / 2 + 0.01, 4) if 'dimensions' in item else None
        } 
        for item in data['objets']
    ]
    prompt = """You are a scene corrector. Your task is to verify if the objects in the scene are properly placed above the ground plane, which is at z=0. If any object is clipping into the ground, correct its position by adjusting only the Z coordinate.
    The object should NOT touch the ground plane, it should be slightly above it, around 0.001m.

           You will receive:
               - The current scene's object's z coordinates, their dimensions and the suggested minimum z to be above the ground plane
               - a collage containing FOUR images of the same scene:
                   1. PERSPECTIVE VIEW: General context.
                   2. TOP-DOWN VIEW: Best for X, Y coordinates and checking if objects are side-by-side.
                   3. SIDE VIEW: Best for Z coordinate (height) to check if objects are at the right height.
                   4. SECOND SIDE VIEW: Best for checking if objects are properly placed below the ground plane.


               RULES:
		       - an object is “clipping” if its lowest point is lower than 0.001.
               - THERE SHOULD BE NO OBJECTS CLIPPING INTO THE GROUND PLANE. If there are, 'valid' is FALSE and you MUST provide corrected Z values for those objects.
               - ONLY output the JSON object, nothing else
               - NO markdown, NO backticks, NO explanations before or after
               - Always use double quotes for keys and strings
               - DO NOT change 'id'.
               - DO NOT lower an object if you don't see space between the object and the ground
               - valid: Set to ‘false’ if ANY object required a Z-adjustment. Set to ‘true’ only if all objects were already correctly positioned.
                  
           You MUST respond with ONLY this JSON format:
           {
               "valid": boolean,
               "feedback": "string description of issues found",
               "corrections": [{"id": "string", "z": float}]
           }
           """

    resultat = ollama.chat(
        model="llama3.2-vision:11b",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 
             'content': f"Objects and their current Z: {json.dumps(z_and_dims)}",
             'images': [image_path]}
        ]
    )
    clean = clean_reponse(resultat['message']['content'])
    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour verif fixage sol")
        return {"valid": False, "feedback": "json invalide", "corrections": []}
    if 'corrections' in resultat and resultat['corrections']:
        corrections = {}
        for c in resultat['corrections']:
            if 'z' in c:
                corrections[c['id']] = c['z']
            elif 'pos' in c and len(c['pos']) == 3:
                corrections[c['id']] = c['pos'][2]

        existing_id = {item['id'] for item in data['objets']}

        for c_id in corrections.keys(): #hallucunation
            if c_id not in existing_id:
                print(f"ATTENTION: l'ID {c_id} n'est pas dans le JSON, skipping correction.")

        for item in data['objets']:
            if item['id'] in corrections:
                old_z = item['pos'][2]
                new_z = corrections[item['id']]
                item['pos'][2] = new_z
                print(f"Z corrigé pour {item['id']}: {old_z:.3f} → {new_z:.3f}")
    
        with open(jsonFile, 'w') as file:
            json.dump(data, file, indent=4)
    return resultat


def verif_orientation(jsonFile, image_path, original_prompt): #TODO: nécessaire?
    #TODO: voir si ça change depuis la position initial ou depuis les rotations qu'il y avait déjà    
    with open(jsonFile, 'r') as file:
        data = json.load(file)
    quat_only = [{'id': item['id'], 'quat': item.get('quat', [0,0,0,1])} for item in data['objets']]
    
    prompt = """
        You are a 3D Orientation Specialist. Your task is to analyze a scene and provide corrective Euler rotations (in degrees) for any misaligned objects.

        COORDINATE SYSTEM:
        - Z-Axis: Up (Vertical)
        - X-Axis: Left/Right
        - Y-Axis: Forward/Backward

        OBJECTIVES:
        1. Ensure objects are upright (bottom on the ground) unless the prompt says otherwise.
        2. Ensure objects face the logical direction (e.g., a chair facing a table).
        3. Identify the ABSOLUTE Euler rotation required to reach the correct state.

        RULES:
        - Use Euler angles in degrees: [rx, ry, rz].
        - rx: Rotation around X (Pitch/Tilt forward-back).
        - ry: Rotation around Y (Roll/Tilt side-to-side).
        - rz: Rotation around Z (Yaw/Heading).
        - Set "valid" to false if any object needs a change.
        - Output ONLY raw JSON.

        JSON FORMAT:
        {
            "valid": false,
            "feedback": "The laptop is upside down; rotating 180 degrees on X.",
            "corrections": [
                {"id": "laptop", "euler_deg": [180, 0, 0]}
            ]
        }
    """

    resultat = ollama.chat(
        model="llama3.2-vision:11b",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user',
             'content': f"Prompt: {original_prompt}\nCurrent orientations: {json.dumps(quat_only)}",
             'images': [image_path]}
        ]
    )

    clean = clean_reponse(resultat['message']['content'])
    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour verif orientation")
        return {"valid": True}
    for corr in resultat['corrections']:
            obj_id = corr['id']
            euler = corr['euler_deg'] #rx,ry,rz
            
            rot = R.from_euler('xyz', euler, degrees=True)
            new_quat = rot.as_quat().tolist() # quaternions x, y, z, w

            for item in data['objets']:
                if item['id'] == obj_id:
                    item['quat'] = new_quat
                    print(f"Euler correction applied to {obj_id}: {euler} deg")

    with open(jsonFile, 'w') as file:
        json.dump(data, file, indent=4)
    return resultat


def etapes_validation(original_prompt, jsonFile, image_path, collision_feedback=None):
    with open(jsonFile, 'r') as file:
        data = json.load(file)
    pos_and_dims = [{'id': item['id'], 'pos': item['pos'], 'dimensions':item['dimensions']} for item in data['objets']]
    prompt = """ You are a 3D Scene Validator and Spatial Coordinator. Your goal is to ensure objects are logically placed, match the prompt's description, and do not collide (overlap) with each other or the ground.

            You will receive:
                - The current scene as a JSON object
                - The position and dimensions of each object
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

    resultat = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user',
             'content': f"Prompt: {original_prompt}\nObjects and positions: {json.dumps(pos_and_dims)}\nCollision report: {collision_feedback}",
             'images': [image_path]}
        ]
    )
    clean = clean_reponse(resultat['message']['content'])
    try:
        resultat = json.loads(clean)
    except json.JSONDecodeError:
        print("json invalide pour etapes_validation")
        return {"valid": False, "feedback": "json invalide"}
    if 'corrections' in resultat:
        corrections = {c['id']: c['pos'] for c in resultat['corrections']}
        correct_json(jsonFile, corrections, 'pos')
    return resultat