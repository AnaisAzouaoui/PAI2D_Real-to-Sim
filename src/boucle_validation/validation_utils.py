import os
import sys
from datetime import datetime
import json
import shutil
import re

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from pipeline.sceneBuilding import apply_relation

SURFACE_MARGIN = 0.04
DEFAULT_LATERAL_DISTANCE = 0.2


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

    if os.path.isdir(base):
        for name in ["mobility.urdf", "kinbody.xml", "textured.obj", "nontextured.stl", "nontextured.ply"]:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Aucun fichier exploitable trouvé dans {base}")

    return base

def create_run_dir():
    '''Crée un dossier pour documenter chaque run'''
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_iteration_scene(image_path, iter, run_dir, itemsList):
    '''Sauvegarde l'image et les données de la scène pour une itération donnée'''
    shutil.copy(image_path, os.path.join(run_dir, f"iteration_{iter}.png"))
    with open(os.path.join(run_dir, f'iter_{iter}_scene.json'), 'w') as f:
        json.dump({'objets': itemsList}, f, indent=4)

def correct_list(data, corrections, field):
    '''Applique les corrections à la liste d'items en fonction de l'ID, pour un champ donné (par exemple pos)'''
    existing_id = {item['id'] for item in data}
    for c_id in corrections.keys():
        if c_id not in existing_id:
            print(f"ATTENTION: l'ID {c_id} n'est pas dans la liste, skipping correction.")
    for item in data:
        if item['id'] in corrections:
            old_value = item[field]
            item[field] = corrections[item['id']]
            print(f"CORRECTION: {field} corrigee pour {item['id']}: {old_value} -> {item[field]}")
            if field == 'pos' and 'dimensions' in item:
                h = item['dimensions'][2]
                new_z = item['pos'][2]
                item['lowest_point'] = new_z - h / 2
                item['highest_point'] = new_z + h / 2
    return data


def get_descendants(items_dict, root_id):
    """Retourne tous les IDs qui sont descendants directs ou indirects de root_id."""
    descendants = set()
    to_visit = [root_id]
    while to_visit:
        current = to_visit.pop()
        for item_id, item in items_dict.items():
            if item.get('parent_id') == current and item_id not in descendants:
                descendants.add(item_id)
                to_visit.append(item_id)
    return descendants


def apply_semantic_corrections(data, corrections):
    """Applique les corrections semantiques du VLM en utilisant apply_relation
    Le VLM retourne des types de relations, le code calcule les coordonnees exactes
    sans se fier aux chiffres hallucines par le VLM """
    items_dict = {item['id']: item for item in data}
    for corr in corrections:
        subject_id = corr.get('subject')
        obj_id = corr.get('object')
        rel_type = corr.get('type')
        if not subject_id or not rel_type or subject_id not in items_dict:
            print(f"[apply_semantic_corrections] correction ignoree (subject ou type manquant): {corr}")
            continue
        subject = items_dict[subject_id]
        if not obj_id or obj_id not in items_dict:
            print(f"[apply_semantic_corrections] objet de reference introuvable: {obj_id}")
            continue
        ref = items_dict[obj_id]

        # bloquer les corrections laterales ou la reference est un descendant du sujet :
        # ex: "boite behind conserve" alors que conserve est posee sur boite → ferait flotter la boite
        if rel_type not in ('on', 'inside'):
            descendants = get_descendants(items_dict, subject_id)
            if obj_id in descendants:
                print(f"[apply_semantic_corrections] correction ignoree: {obj_id} est un descendant de {subject_id}, ignorer {rel_type}")
                continue

        if rel_type in ('on', 'inside'):
            # sauvegarder X/Y avant apply_relation qui les randomise
            prev_x, prev_y = subject['pos'][0], subject['pos'][1]
            apply_relation({'type': rel_type}, ref, subject)
            if 'rel_x' in corr or 'rel_y' in corr:
                # position explicite sur la surface demandee par le VLM
                pw, pd, _ = ref['dimensions']
                sw, sd, _ = subject['dimensions']
                margin_x = max(0.0, (pw - sw) / 2 - SURFACE_MARGIN)
                margin_y = max(0.0, (pd - sd) / 2 - SURFACE_MARGIN)
                rel_x = corr.get('rel_x', 0.5)
                rel_y = corr.get('rel_y', 0.5)
                px, py = ref['pos'][0], ref['pos'][1]
                target_x = px - pw / 2 + rel_x * pw
                target_y = py - pd / 2 + rel_y * pd
                subject['pos'][0] = max(px - margin_x, min(target_x, px + margin_x))
                subject['pos'][1] = max(py - margin_y, min(target_y, py + margin_y))
            else:
                # on ne corrige que Z : X/Y deja corrects, ne pas les perdre
                subject['pos'][0] = prev_x
                subject['pos'][1] = prev_y
        else:
            apply_relation({'type': rel_type, 'distance': DEFAULT_LATERAL_DISTANCE}, ref, subject)
        print(f"[apply_semantic_corrections] {rel_type}({subject_id}, {obj_id}) -> pos={subject['pos']}")
    return data


def clean_reponse(resultat):
    resultat = re.sub(r'```', '', resultat)
    start = resultat.find('{')
    end = resultat.rfind('}') + 1
    if start != -1 and end != 0:
        json_str = resultat[start:end]
        json_str = json_str.replace('True', 'true').replace('False', 'false') 
        return json_str
    return ""