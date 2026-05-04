import os
import random
from .itemSpec import getOriginalDimensions, getFilePath
from .jsonParsing import simplifyRelations
import xml.etree.ElementTree as ET
import trimesh
import genesis as gs
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from simulation.simulationGenesis import make_morph


'''
Des liens utiles:
https://wiki.ros.org/urdf/XML/link
https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/conventions.htmlf
'''


def initPosAndQuat(items, scene=None, entites=None):
    '''
    Initialise les positions et orientations des objets, en prenant les infos sur la taille de l'objet de genesis en generant les objets plutot qu'en les parsant
    Donne des orientations et positions de base à tous les items, qui seront modifiées ensuite en fonction des relations.

    :param items: les items
    :return: les items mais avec des positions et orientations de base.
    '''
    for item in items:
        if entites is not None:
            ent = entites[item['id']]
            aabb_min, aabb_max = ent.get_AABB() #on get les limites de la bounding box
            item['dimensions'] = [float(aabb_max[0] - aabb_min[0]), float(aabb_max[1] - aabb_min[1]),float(aabb_max[2] - aabb_min[2])]
            offset = 0.001 - float(aabb_min[2]) #qui est le lowest point
            item['highest_point'] = float(aabb_max[2]) + offset
        else:
            offset = 0.001
            item['highest_point'] = item['dimensions'][2] + offset
        item['pos'] = [0, 0, offset] #fait sortir les objets du sol
        item['lowest_point'] = 0.001
        item['parent_id'] = None
    return items


def build_scene_graph(items_dict, relations):
    """Fait l'arbre des dépendances pour savoir quel objet est sur quel autre objet."""
    #liens explicites
    for rel in relations:
        if rel['type'] in ['on', 'inside']:
            obj_id = rel['object']
            sub_id = rel['subject']
            if sub_id in items_dict and obj_id in items_dict:
                items_dict[sub_id]['parent_id'] = obj_id

    #propagation des liens: si un objet est à coté d'un objet qui 
    changed = True
    for _ in range(10): #il peut y avoir des chaines de relation donc c'est au cas ou ce soit le cas
        changed = False
        for rel in relations:
            if rel['type'] not in ['on', 'inside']:
                obj_id = rel['object']
                sub_id = rel['subject']
                if sub_id in items_dict and obj_id in items_dict:
                    obj_parent = items_dict[obj_id].get('parent_id')
                    sub_parent = items_dict[sub_id].get('parent_id')
                    #héritance des parents;
                    if obj_parent is not None and sub_parent is None:
                        items_dict[sub_id]['parent_id'] = obj_parent
                        changed = True
                    if sub_parent is not None and obj_parent is None:
                        items_dict[obj_id]['parent_id'] = sub_parent
                        changed = True
        if not changed:
            break

def parent_bound(item, parent_item):
    """
    Fait en sorte que l'objet ne dépasse pas les limites de son parent (ex: le mug qui est sur la table a des x,y compris dans les dimensions du plateau de la table)
    """
    if not parent_item: 
        return
    iw, id, ih = item['dimensions']
    pw, pd, ph = parent_item['dimensions']
    margin_x = (pw - iw) / 2
    margin_y = (pd - id) / 2
    px, py, _ = parent_item['pos']
    item['pos'][0] = max(px - margin_x, min(item['pos'][0], px + margin_x))
    item['pos'][1] = max(py - margin_y, min(item['pos'][1], py + margin_y))



def apply_relation(rel, item, subject):
    """calcul de la nouvelle pos"""

    s_item = item.get('scale', 1.0)
    w, d, h = [dim * s_item for dim in item['dimensions']]
    item_highest = item['highest_point']
    item_lowest = item['lowest_point']
    x, y, z = item['pos']
    
    s_sub = subject.get('scale', 1.0)
    sw, sd, sh = [dim * s_sub for dim in subject['dimensions']]
    sub_highest = subject['highest_point']
    sub_lowest = subject['lowest_point']
    sub_z_offset = subject['pos'][2] - subject['lowest_point'] #offset entre l'origine et le bas
    new_x, new_y, new_z = subject['pos']


    isDistance = 'distance' in rel
    distance = rel.get('distance', 0)
    rel_type = rel['type']

    if rel_type == 'on':
        new_z = item_highest + sub_z_offset + 0.001
        #placemnt within les bounds
        margin_x = (w - sw) / 2
        margin_y = (d - sd) / 2
        new_x = x + random.uniform(-max(0, margin_x), max(0, margin_x))
        new_y = y + random.uniform(-max(0, margin_y), max(0, margin_y))
    elif rel_type == 'inside':
        new_x, new_y, new_z = x, y, z+ sub_z_offset + 0.001
    elif rel_type == 'against':
        new_x = x + w/2 + sw/2 + 0.01
        new_y = y
    elif rel_type in ['right_of', 'left_of', 'in_front_of', 'behind']:
        new_z = item_lowest + sub_z_offset  # meme niveau de surface que la reference
        offset = distance if isDistance else random.uniform(0.15, 0.3)
        if rel_type == 'right_of':
            new_x = x
            new_y = y + d/2 + sd/2 + offset
        elif rel_type == 'left_of':
            new_x = x
            new_y = y - d/2 - sd/2 - offset
        elif rel_type == 'in_front_of':
            new_x = x + w/2 + sw/2 + offset
            new_y = y
        elif rel_type == 'behind':
            new_x = x - w/2 - sw/2 - offset
            new_y = y
    else:
        print(f"Relation non traitee: {rel_type}")

    subject['pos'] = [new_x, new_y, new_z]
    subject['lowest_point'] = new_z - sub_z_offset
    subject['highest_point'] = subject['lowest_point'] + sh


_OPPOSITE_DIRECTION = {
    'right_of': 'left_of',
    'left_of': 'right_of',
    'in_front_of': 'behind',
    'behind': 'in_front_of',
    'against': 'against',
}


def processRelations(items, relations):
    items_dict = {item['id']: item for item in items}
    relations = simplifyRelations(relations)
    build_scene_graph(items_dict, relations)

    #  places en chaine, pas mis en root directement 
    directional_types = {'right_of', 'left_of', 'in_front_of', 'behind', 'against'}
    directional_subject_ids = {
        rel['subject'] for rel in relations
        if rel['type'] in directional_types and rel.get('subject') in items_dict
    }

    # roots = objets sans parent ET pas subjets d'une relation directionnelle (left and stuff)
    placed_items = {
        item['id'] for item in items
        if items_dict[item['id']].get('parent_id') is None
        and item['id'] not in directional_subject_ids
    }
    if not placed_items:
        placed_items = {items[0]['id']}
    print(f"[sceneBuilding] roots : {placed_items}")

    # directionnelles d'abord : les roots sont repositionnes avant que leurs enfants soient poses dessus
    relations.sort(key=lambda r: 0 if r['type'] not in ('on', 'inside') else 1)
    reste = [rel for rel in relations if rel.get('object') in items_dict and rel.get('subject') in items_dict]
    while reste:
        progression = False
        for rel in reste[:]:
            obj_id = rel['object']
            sub_id = rel['subject']
            is_vertical = rel['type'] in ('on', 'inside')
            # pour les verticales : ne placer que si le subject n'est pas encore place (evite les conflits)
            # pour les directionnelles : toujours appliquer meme si le subject est deja root
            if obj_id in placed_items and (sub_id not in placed_items or not is_vertical):
                item = items_dict[obj_id]
                subject = items_dict[sub_id]
                apply_relation(rel, item, subject)
                parent_id = subject.get('parent_id')
                if parent_id and parent_id in items_dict:
                    parent_bound(subject, items_dict[parent_id])
                placed_items.add(sub_id)
                reste.remove(rel)
                progression = True
                break  # repart depuis le debut pour que la relation suivante voie les positions a jour
        if not progression:
            # le model inverse des fois right(mug,banne) et right(banane,mug) 
            for rel in reste[:]:
                obj_id = rel['object']
                sub_id = rel['subject']
                if sub_id in placed_items and obj_id not in placed_items:
                    opposite = _OPPOSITE_DIRECTION.get(rel['type'])
                    if opposite:
                        print(f"[sceneBuilding] relation inversee : {rel['type']}({sub_id},{obj_id}) -> {opposite}({obj_id},{sub_id})")
                        inverted = dict(rel, type=opposite, subject=obj_id, object=sub_id)
                        apply_relation(inverted, items_dict[sub_id], items_dict[obj_id])
                        parent_id = items_dict[obj_id].get('parent_id')
                        if parent_id and parent_id in items_dict:
                            parent_bound(items_dict[obj_id], items_dict[parent_id])
                        placed_items.add(obj_id)
                        reste.remove(rel)
                        progression = True
            if not progression:
                print(f"[sceneBuilding] Relations impossibles a resoudre, ignorees : {reste}")
                break
    return items


def changeQuatAndPosFromTurn(turn, item, ent):
    '''
    Change les données d'orientation du sujet
    '''
    current = item.get('quat', [0, 0, 0, 1])
    if not item.get('dimensions'):
        item = getOriginalDimensions(item)
    (width, depth, height) = item['dimensions']
    rotations = {
        'tip_left':[0, 0, 0.707, 0.707],
        'tip_right':[0, 0, -0.707, 0.707],
        'upside_down': [0, 0, 1, 0],
        'tip_forward':[0.707, 0, 0, 0.707],
        'tip_backward':[-0.707, 0, 0, 0.707],
        'turn_right':[0, 0.707, 0, 0.707],
        'turn_left': [0, -0.707, 0, 0.707],
        'turn_around':[1, 0, 0, 0]}
    
    if turn not in rotations:
        print(f"Changement d'orientation non traité: {turn}")
        return item

    change = rotations[turn]
    x1, y1, z1, w1 = current
    x2, y2, z2, w2 = change
    new_quat = [w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2]
    item['quat'] = new_quat

    # #et ensuite on change les dimensions vu que le dessus de l'objet n'et plus le meme etc...
    # if turn == 'turn_left' or turn == 'turn_right':
    #     item['dimensions'] = (depth, width, height)
    # elif turn == 'tip_forward' or turn == 'tip_backward':
    #     item['dimensions'] = (width, height, depth)
    # elif turn == 'tip_right' or turn == 'tip_left':
    #     item['dimensions'] = (height, depth, width)

    # le truc commenté, mais avec genesis instead hihi:

    if ent is not None:
        ent.set_quat(new_quat)
    # scene.build()
    
    # aabb_min, aabb_max = ent.get_AABB()
    # item['dimensions'] = [
    #     float(aabb_max[0] - aabb_min[0]),
    #     float(aabb_max[1] - aabb_min[1]),
    #     float(aabb_max[2] - aabb_min[2])
    # ]
    # z_min_after_rot = float(aabb_min[2])
    # item['z_offset_local'] = 0.001 - z_min_after_rot
    # item['pos'][2] = item['z_offset_local']
    # item['lowest_point'] = 0.001
    # item['highest_point'] = float(aabb_max[2]) + item['z_offset_local']

    return item


def processOrientations(items, orientations, entites=None):
    '''
    Adapte les quaternions des items en fonctions des changements d'orientation.

    :param items: liste de dict d'items
    :param relations: les changements d'orientation
    :return: items, les items avec les bonnes orientations
    '''
    items_dict = {item['id']: item for item in items} #pour pouvoir acceder aux differents items plus facilement
    for ori in orientations:
        id = ori['id']
        turn = ori['turn']
        if id not in items_dict:
            print(f"[processOrientations] ID inconnu ignore : '{id}'")
            continue
        item = items_dict[id]
        ent = entites[id] if entites is not None else None
        changeQuatAndPosFromTurn(turn, item, ent)
    return items




def get_genesis_dimensions(items, orientations=None):
    '''Charge les meshes dans Genesis pour obtenir les vraies AABB et dimensions (c'est pour v3 tkt)'''
    if orientations is None:
        orientations = []
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    entites = {}
    for item in items:
        item['quat'] = [0, 0, 0, 1]
        item['pos'] = [0, 0, 0]
        ent = scene.add_entity(make_morph(item['path'], scale=item.get('scale', 1.0), pos=[0,0,0], fixed=True))
        entites[item['id']] = ent
    scene.build()
    items = processOrientations(items, orientations, entites)
    items = initPosAndQuat(items, scene, entites)
    gs.destroy()
    return items


def buildScene(items, relations, orientations):
    '''
    Ajoute à une liste de dictionnaires (un dict par item), toutes les infos nécessaies à la simulation, donc path et pos
    '''
    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    entites = {}
    for item in items:
        item['path'] = getFilePath(item)
        item['path'] = getFilePath(item)
        item['quat'] = [0, 0, 0, 1]
        item['pos'] = [0, 0, 0]
        ent = scene.add_entity(make_morph(item['path'], scale=item.get('scale', 1.0), pos=[0,0,0], fixed=True))
        entites[item['id']] = ent
    scene.build()       

    items = processOrientations(items, orientations, entites)
    items = initPosAndQuat(items, scene, entites)
    items = processRelations(items, relations)
    
    gs.destroy()
    return items