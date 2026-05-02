import os
import random
from .itemSpec import getOriginalDimensions, getFilePath
from .jsonParsing import simplifyRelations
import xml.etree.ElementTree as ET
import trimesh


'''
Des liens utiles:
https://wiki.ros.org/urdf/XML/link
https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/conventions.htmlf
'''


def findHeightOffset(item):
    urdf_path = os.path.abspath(item['path'])
    if not os.path.exists(urdf_path):
        print(f"[findHeightOffset] fichier introuvable : {urdf_path}")
        return 0

    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError:
        print(f"[findHeightOffset] pas un URDF valide ({os.path.basename(urdf_path)}), offset=0")
        return 0
    root = tree.getroot()
    urdf_dir = os.path.dirname(urdf_path)

    for geometry in root.iter('geometry'):
        box= geometry.find('box')
        if box is not None:
            h = float(box.attrib['size'].split()[2])
            return -h/2
        cylinder = geometry.find('cylinder')
        if cylinder is not None:
            h = float(cylinder.attrib['length'])
            return -h/2
        sphere = geometry.find('sphere')
        if sphere is not None:
            h = float(sphere.attrib['radius'])
            return -h
        mesh = geometry.find('mesh')
        if mesh is not None:
            filename = mesh.attrib['filename']
            if filename.startswith('package://'):
                filename = filename.replace('package://', '')
            mesh_path = os.path.abspath(os.path.join(urdf_dir, filename))
            if not os.path.exists(mesh_path):
                print(f"mesh introuvable : {mesh_path}")
                return 0    
            try:
                mesh_data = trimesh.load(mesh_path)
                if isinstance(mesh_data, trimesh.Scene):
                    min_z = mesh_data.bounds[0][2]
                else:
                    min_z = mesh_data.bounds[0][2]
                return min_z
            except Exception as e:
                print(f"erreur trimesh sur {mesh_path}: {e}")
                return 0
    return 0


def initPosAndQuat(items):
    '''
    Initialise les positions et orientations des objets.
    Donne des orientations et positions de base à tous les items, qui seront modifiées ensuite en fonction des relations.

    :param items: les items
    :return: les items mais avec des positions et orientations de base.
    '''
    for item in items:
        if not item.get('dimensions'):
            item = getOriginalDimensions(item)
        #(_, _, height) = item['dimensions']
        #s = item.get('scale', 1.0)
        #item['pos'] = (0, 0, 0.01 + height * s / 2)
        #if not item.get('quat'):
        #    item['quat'] = (0,0,0,1) #pas de rotation

        height_offset = findHeightOffset(item)
        s = item.get('scale', 1.0)
        item['pos'] = [0, 0, 0.01 + height_offset * s]
        if not item.get('quat'):
            item['quat'] = [0,0,0,1] 
        
        item['parent_id'] = None 
    return items


# def getRoot(items):
#     '''
#     Trouve l'item root et le met dans placed_items.
#
#     :param items: les items
#     :return: les items placés qui ne contient que le root pour l'instant
#     '''
#     placed_items = set()
#
#     for item in items:
#         root_val = item.get('root')
#         if root_val is True or (isinstance(root_val, str) and root_val.lower() == 'true'):
#             placed_items.add(item['id'])
#
#     # fallback : si aucun root explicite, utiliser le premier item
#     if not placed_items and items:
#         print(f"[sceneBuilding] Aucun root explicite, fallback sur '{items[0]['id']}'")
#         items[0]['root'] = True
#         placed_items.add(items[0]['id'])
#
#     if not placed_items:
#         raise ValueError("Aucun objet à placer")
#
#     return placed_items


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
    px, py, pz = parent_item['pos']
    p_scale = parent_item.get('scale', 1.0)
    pw, pd, ph = [d * p_scale for d in parent_item['dimensions']]
    ix, iy, iz = item['pos']
    i_scale = item.get('scale', 1.0)
    iw, id, ih = [d * i_scale for d in item['dimensions']]

    #limites du paren: (TODO: ATTENTION FAUT REVOIR APRES POUR QUAND C'EST PAS AU CENTRE)
    min_x = px - pw/2 + iw/2
    max_x = px + pw/2 - iw/2
    min_y = py - pd/2 + id/2
    max_y = py + pd/2 - id/2
    if min_x > max_x: #cas où l'objet est plus grand que son parent: on le centre sur le parent pour essayer de faire en sorte qu'il ne tombe pas
        min_x = max_x = px
    if min_y > max_y: 
        min_y = max_y = py

    item['pos'][0] = max(min_x, min(ix, max_x))
    item['pos'][1] = max(min_y, min(iy, max_y))

def apply_relation(rel, item, subject):
    """calcul de la nouvelle pos"""
    s_item = item.get('scale', 1.0)
    s_sub = subject.get('scale', 1.0)
    w, d, h = [dim * s_item for dim in item['dimensions']]
    sw, sd, sh = [dim * s_sub for dim in subject['dimensions']]
    x, y, z = item['pos']
    isDistance = 'distance' in rel
    distance = rel.get('distance', 0)
    rel_type = rel['type']
    new_x, new_y, new_z = subject['pos']
    if rel_type == 'on':
        # pos = bottom of object for bottom-origin meshes, so top = z + h
        new_z = z + h + 0.001
        new_x = x + random.uniform(-w/2 + sw/2, w/2 - sw/2)
        new_y = y + random.uniform(-d/2 + sd/2, d/2 - sd/2)
    elif rel_type == 'inside':
        new_x, new_y, new_z = x, y, z
    elif rel_type == 'against':
        new_x = x + w/2 + sw/2 + 0.01
        new_y = y
    elif rel_type in ['right_of', 'left_of', 'in_front_of', 'behind']:
        # new_z keeps subject's initPosAndQuat value (ground-level already set)
        offset = distance if isDistance else random.uniform(0.05, 0.2)
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

    # root = items jamais subjects d'une relation verticale (ne reposent sur rien)
    subjects = {rel['subject'] for rel in relations if rel['type'] in ('on', 'inside')}
    placed_items = {item['id'] for item in items if item['id'] not in subjects}
    if not placed_items:
        placed_items = {items[0]['id']}
    print(f"[sceneBuilding] roots : {placed_items}")

    # directionnelles en premier : les roots sont positionnés avant que leurs enfants soient posés dessus
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
            # second pass: the model sometimes inverts subject/object — if the subject
            # is placed but the object is not, invert the relation direction to recover
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


def changePosFromRel(rel, item, subject):

    processed = True
    if not item.get('dimensions'):
        item = getOriginalDimensions(item)
    if not subject.get('dimensions'):
        subject = getOriginalDimensions(subject)
    s_item = item.get('scale', 1.0)
    s_sub = subject.get('scale', 1.0)
    width, depth, height = [d * s_item for d in item['dimensions']]
    sw, sd, sh = [dim * s_sub for dim in subject['dimensions']]
    x,y,z = item['pos']
    if not subject.get('dimensions'):
        subject = getOriginalDimensions(subject)
    s_sub = subject.get('scale', 1.0)
    subject_w, subject_d, subject_h = [d * s_sub for d in subject['dimensions']]
    (subject_x, subject_y, subject_z) = subject['pos']

    isDistance = False
    distance = 0
    if rel.get('distance'):
        distance = rel['distance']
        isDistance = True

    if rel['type'] == 'on':
        subject_x += random.uniform(x - width/2 + subject_w/2, x + width/2 - subject_w/2) #prise en compte des dims du sujet pour ne pas etre trop au bord
        subject_y += random.uniform(y - depth/2 + subject_d/2, y + depth/2 - subject_d/2)
        subject_z += z + height/2
    elif rel['type'] == 'in_front_of':
        if isDistance:
            subject_x += x + width/2 + subject_w/2 + distance
            subject_z = z
        else:
            subject_x += random.uniform(x + width/2 + subject_w/2, x + width + subject_w/2)
            subject_z = z
    elif rel['type'] == 'behind':
        if isDistance:
            subject_x += x - width/2 - subject_w - distance
            subject_z = z
        else:
            subject_x += random.uniform(x - width/2 - subject_w , x - width - subject_w/2) #temp fix
            subject_z = z
    elif rel['type'] == 'right_of': 
        if isDistance:
            subject_y += y + depth/2 + subject_d/2 + distance
            subject_z = z
        else:
            subject_y += random.uniform(y + depth/2 + subject_d/2, y + depth + subject_d/2)
            subject_z = z
    elif rel['type'] == 'left_of':
        if isDistance:
            subject_y += y - depth/2 - subject_d/2 - distance
            subject_z = z
        else:
            subject_y += random.uniform(y - depth/2 - subject_d/2, y - depth - subject_d/2)
            subject_z = z
    elif rel['type'] == 'against':
        subject_x = x + width/2 + subject_w/2 + 0.01
        subject_y = y
        subject_z = z
    elif rel['type'] == 'inside':
        subject_x = x
        subject_y = y
        subject_z = z
    else:
        print("relation non traitée: ", rel['type'])
    subject['pos'] = (subject_x, subject_y, subject_z)
    


def changeQuatAndPosFromTurn(turn, item):
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
        'upside_down': [1, 0, 0, 0],
        'tip_forward':[0.707, 0, 0, 0.707],
        'tip_backward':[-0.707, 0, 0, 0.707],
        'turn_right':[0, 0.707, 0, 0.707],
        'turn_left': [0, -0.707, 0, 0.707],
        'turn_around':[0, 0, 1, 0]}
    
    if turn not in rotations:
        print(f"chagement d'orientation non traité: {turn}")
        return item
    
    #aja que les quaternions marchent avec des multiplications matricielle. enfin. j'avais déjà appris ça. il a fallu revoir.
    change = rotations[turn]
    x1, y1, z1, w1 = current
    x2, y2, z2, w2 = change
    new_quat = [w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2]
    item['quat'] = new_quat

    #et ensuite on change les dimensions vu que le dessus de l'objet n'et plus le meme etc...
    if turn == 'turn_left' or turn == 'turn_right':
        item['dimensions'] = (depth, width, height)
    elif turn == 'tip_forward' or turn == 'tip_backward':
        item['dimensions'] = (width, height, depth)
    elif turn == 'tip_right' or turn == 'tip_left':
        item['dimensions'] = (height, depth, width)

    return item


# def processRelations(items, relations):
#     '''
#     Adapte les coordonnees x,y,z des items en fonction des relations.

#     :param items: liste de dict d'items
#     :param relations: les relations entre les items
#     :return: items, les items avec les bonnes coordonnées et les bonnes 
#     '''

#     items_dict = {item['id']: item for item in items} #pour pouvoir acceder aux differents items plus facilement

#     placed_items = getRoot(items)
    
#     relations = simplifyRelations(relations)

    # reste = [rel for rel in relations if rel.get('object', '') in items_dict and rel.get('subject', '') in items_dict]

    # while reste:
    #     progression = False
    #     for rel in reste[:]:
    #         item_id = rel['object']
    #         subject_id = rel['subject']
    #         if item_id not in items_dict:
    #             print(f"ATTENTION: ID objet inconnu dans relation: '{item_id}'")
    #         if subject_id not in items_dict:
    #             print(f"ATTENTION: ID objet inconnu dans relation: '{subject_id}'")
    #         item = items_dict[item_id]
    #         subject = items_dict[subject_id]
    #         if item_id not in placed_items:
    #             continue
    #         else:
    #             changePosFromRel(rel, item, subject)
    #             placed_items.add(subject_id)
    #             reste.remove(rel)
    #             progression = True
    #     if not progression:
    #         print(f"[sceneBuilding] Relations impossibles à résoudre, ignorées : {reste}")
    #         break
    # return items


def processOrientations(items,orientations):
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
        changeQuatAndPosFromTurn(turn, item)
    return items


def buildScene(items, relations, orientations): #TODO c'est très moche comme façon de faire
    '''
    Ajoute à une liste de dictionnaires (un dict par item), toutes les infos nécessaies à la simulation, donc path et pos
    '''
    for item in items:
        item['path'] = getFilePath(item)
        if not item.get('dimensions'):
            item = getOriginalDimensions(item)
    items = processOrientations(items,orientations)
    items = initPosAndQuat(items)
    items = processRelations(items,relations)
    return items
